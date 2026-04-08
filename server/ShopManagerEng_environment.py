import random
import uuid
from openenv.core.env_server import Environment

try:
    from ..models import JewelryAction, JewelryObservation, JewelryState, PRODUCT_CATALOG
except ImportError:
    from models import JewelryAction, JewelryObservation, JewelryState, PRODUCT_CATALOG


# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────

STARTING_CASH      = 1000.0
GOLD_PRICE_MIN     = 250.0
GOLD_PRICE_MAX     = 450.0
PRICE_FLUCTUATION  = 0.10       # ±10% per market round
MAX_MARKET_ROUNDS  = 3          # Rounds the agent can wait in market
MAX_NEGOTIATION    = 5          # Showroom counter-offer limit
COUNTER_BUMP       = 1.05       # Customer raises offer by 5% each round
OFFER_MIN_RATIO    = 0.80       # Customer opens at 80-130% of cost basis
OFFER_MAX_RATIO    = 1.30
DEMAND_OFFER_BONUS = 0.20       # High demand adds up to 20% to offer
MAX_PROFIT_MULT    = 2.0        # Normalization ceiling for r3


# ─────────────────────────────────────────────
#  KEYWORD DETECTION  (Phase 3 — Showroom)
# ─────────────────────────────────────────────

ACCEPT_KEYWORDS = ["accept", "deal", "sold", "agreed", "yes", "take it", "i'll take"]
REJECT_KEYWORDS = ["reject", "no deal", "refuse", "walk away", "not interested", "no thanks"]

def detect_intent(message: str) -> str:
    msg = message.lower()
    for kw in ACCEPT_KEYWORDS:
        if kw in msg:
            return "accept"
    for kw in REJECT_KEYWORDS:
        if kw in msg:
            return "reject"
    return "counter"


# ─────────────────────────────────────────────
#  REWARD HELPERS
# ─────────────────────────────────────────────

def compute_r1(buy_price: float, lowest_price: float) -> float:
    """
    Phase 1 reward: did the agent buy near the lowest price seen?
    1.0 if bought at the lowest, decreasing as buy price increases.
    """
    if lowest_price <= 0 or buy_price <= 0:
        return 0.0
    ratio = lowest_price / buy_price  # 1.0 = perfect, <1.0 = overpaid
    return round(min(ratio, 1.0) * 0.5, 4)


def compute_r2(product_choice: str, demand: dict) -> float:
    """
    Phase 2 reward: did the agent pick the highest-demand product?
    0.5 if picked the best, proportionally less for worse choices.
    """
    if not demand or product_choice not in demand:
        return 0.0
    max_demand = max(demand.values())
    if max_demand <= 0:
        return 0.0
    return round((demand[product_choice] / max_demand) * 0.5, 4)


def compute_r3(accepted_price: float, cost_basis: float) -> float:
    """
    Phase 3 reward: normalized profit margin on sale.
    """
    if cost_basis <= 0:
        return 0.0
    profit = accepted_price - cost_basis
    if profit <= 0:
        return 0.0
    max_profit = cost_basis * (MAX_PROFIT_MULT - 1)
    return round(min(profit / max_profit, 1.0), 4)


def combined_reward(r1: float, r2: float, r3: float) -> float:
    """Weighted combination: showroom dominates."""
    return round((0.2 * r1) + (0.2 * r2) + (0.6 * r3), 4)


# ─────────────────────────────────────────────
#  ENVIRONMENT
# ─────────────────────────────────────────────

class JewelryShopEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._state = JewelryState()
        self._r1 = 0.0
        self._r2 = 0.0

    # ── RESET ──────────────────────────────────

    def reset(self, seed=None, episode_id=None, **kwargs) -> JewelryObservation:
        if seed is not None:
            random.seed(seed)

        gold_price = round(random.uniform(GOLD_PRICE_MIN, GOLD_PRICE_MAX), 2)

        # Randomize demand levels for this episode
        demand = {
            "ring":     round(random.uniform(0.4, 1.0), 2),
            "necklace": round(random.uniform(0.2, 0.8), 2),
            "bracelet": round(random.uniform(0.1, 0.6), 2),
        }

        self._state = JewelryState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            cash=STARTING_CASH,
            gold_oz=0.0,
            gold_price=gold_price,
            gold_price_history=[gold_price],
            market_round=0,
            demand=demand,
            inventory={"ring": 0, "necklace": 0, "bracelet": 0},
            phase="market",
            product_for_sale=None,
            cost_basis=0.0,
            negotiation_round=0,
            current_offer=0.0,
            base_offer=0.0,
            lowest_price_seen=gold_price,
        )
        self._r1 = 0.0
        self._r2 = 0.0

        return JewelryObservation(
            done=False,
            reward=None,
            phase="market",
            cash=STARTING_CASH,
            gold_oz=0.0,
            gold_price=gold_price,
            gold_price_history=[gold_price],
            market_round=0,
            max_market_rounds=MAX_MARKET_ROUNDS,
            demand=demand,
            product_catalog=PRODUCT_CATALOG,
            inventory={"ring": 0, "necklace": 0, "bracelet": 0},
            product_for_sale=None,
            cost_basis=0.0,
            current_offer=None,
            negotiation_round=0,
            message=(
                f"Welcome to the Jewelry Shop! Today's gold price is ${gold_price}/oz. "
                f"You have ${STARTING_CASH}. You can 'buy' gold or 'wait' for a better price. "
                f"Market rounds remaining: {MAX_MARKET_ROUNDS}."
            ),
        )

    # ── STEP ───────────────────────────────────

    def step(self, action: JewelryAction, timeout_s=None, **kwargs) -> JewelryObservation:
        self._state.step_count += 1
        phase = self._state.phase

        if phase == "market":
            return self._step_market(action)
        elif phase == "warehouse":
            return self._step_warehouse(action)
        elif phase == "showroom":
            return self._step_showroom(action)
        else:
            raise ValueError(f"Unknown phase: {phase}")

    # ── PHASE 1: MARKET ────────────────────────

    def _step_market(self, action: JewelryAction) -> JewelryObservation:
        s = self._state
        market_action = (action.market_action or "wait").lower().strip()

        if market_action == "buy":
            gold_qty = action.gold_qty or 0.0
            total_cost = gold_qty * s.gold_price

            if gold_qty <= 0 or total_cost > s.cash:
                # Failed transaction — stay in market
                return JewelryObservation(
                    done=False,
                    reward=0.0,
                    phase="market",
                    cash=s.cash,
                    gold_oz=s.gold_oz,
                    gold_price=s.gold_price,
                    gold_price_history=list(s.gold_price_history),
                    market_round=s.market_round,
                    max_market_rounds=MAX_MARKET_ROUNDS,
                    demand=s.demand,
                    product_catalog=PRODUCT_CATALOG,
                    inventory=s.inventory,
                    message=(
                        f"Transaction failed. Tried to buy {gold_qty}oz "
                        f"(${total_cost:.2f}) but you have ${s.cash:.2f}. "
                        f"Try a smaller quantity or wait."
                    ),
                )

            # Successful buy
            s.cash -= total_cost
            s.gold_oz += gold_qty
            self._r1 = compute_r1(s.gold_price, s.lowest_price_seen)

            # Advance to warehouse
            s.phase = "warehouse"

            return JewelryObservation(
                done=False,
                reward=self._r1,
                phase="warehouse",
                cash=s.cash,
                gold_oz=s.gold_oz,
                gold_price=s.gold_price,
                gold_price_history=list(s.gold_price_history),
                market_round=s.market_round,
                max_market_rounds=MAX_MARKET_ROUNDS,
                demand=s.demand,
                product_catalog=PRODUCT_CATALOG,
                inventory=s.inventory,
                message=(
                    f"Bought {gold_qty}oz of gold at ${s.gold_price}/oz "
                    f"for ${total_cost:.2f}. Cash remaining: ${s.cash:.2f}. "
                    f"Now check your warehouse. Which product to craft? "
                    f"Options: ring (1oz gold + $200), necklace (2oz + $300), bracelet (0.5oz + $100)."
                ),
            )

        else:
            # Agent chose to WAIT — advance market round
            s.market_round += 1

            if s.market_round >= MAX_MARKET_ROUNDS:
                # Forced to buy at current price or skip to warehouse with no gold
                s.phase = "warehouse"
                self._r1 = 0.0
                return JewelryObservation(
                    done=False,
                    reward=0.0,
                    phase="warehouse",
                    cash=s.cash,
                    gold_oz=s.gold_oz,
                    gold_price=s.gold_price,
                    gold_price_history=list(s.gold_price_history),
                    market_round=s.market_round,
                    max_market_rounds=MAX_MARKET_ROUNDS,
                    demand=s.demand,
                    product_catalog=PRODUCT_CATALOG,
                    inventory=s.inventory,
                    message=(
                        f"Market closed! You waited too long and didn't buy any gold. "
                        f"Entering warehouse with {s.gold_oz}oz gold and ${s.cash} cash."
                    ),
                )

            # Price fluctuates ±10%
            change = random.uniform(-PRICE_FLUCTUATION, PRICE_FLUCTUATION)
            new_price = round(s.gold_price * (1 + change), 2)
            new_price = max(new_price, 50.0)  # Floor price
            s.gold_price = new_price
            s.gold_price_history.append(new_price)
            s.lowest_price_seen = min(s.lowest_price_seen, new_price)

            trend = "↑" if change > 0 else "↓"
            return JewelryObservation(
                done=False,
                reward=0.0,
                phase="market",
                cash=s.cash,
                gold_oz=s.gold_oz,
                gold_price=new_price,
                gold_price_history=list(s.gold_price_history),
                market_round=s.market_round,
                max_market_rounds=MAX_MARKET_ROUNDS,
                demand=s.demand,
                product_catalog=PRODUCT_CATALOG,
                inventory=s.inventory,
                message=(
                    f"You waited. Gold price moved {trend} to ${new_price}/oz. "
                    f"Price history: {s.gold_price_history}. "
                    f"Rounds left: {MAX_MARKET_ROUNDS - s.market_round}. "
                    f"Buy now or wait?"
                ),
            )

    # ── PHASE 2: WAREHOUSE ─────────────────────

    def _step_warehouse(self, action: JewelryAction) -> JewelryObservation:
        s = self._state
        choice = (action.product_choice or "ring").lower().strip()

        if choice not in PRODUCT_CATALOG:
            choice = "ring"  # Default fallback

        spec = PRODUCT_CATALOG[choice]
        gold_needed = spec["gold_oz"]
        labor_cost = spec["labor"]

        has_gold = s.gold_oz >= gold_needed
        has_cash = s.cash >= labor_cost

        if not has_gold or not has_cash:
            # Cannot craft — skip to showroom with nothing
            self._r2 = 0.0
            s.phase = "showroom"
            reason = (
                f"not enough gold (need {gold_needed}oz, have {s.gold_oz}oz)"
                if not has_gold else
                f"not enough cash for labor (need ${labor_cost}, have ${s.cash:.2f})"
            )
            return JewelryObservation(
                done=False,
                reward=0.0,
                phase="showroom",
                cash=s.cash,
                gold_oz=s.gold_oz,
                gold_price=s.gold_price,
                gold_price_history=list(s.gold_price_history),
                demand=s.demand,
                product_catalog=PRODUCT_CATALOG,
                inventory=s.inventory,
                product_for_sale=None,
                cost_basis=0.0,
                message=f"Cannot craft {choice}: {reason}. Entering showroom with nothing.",
            )

        # Successful craft
        s.cash -= labor_cost
        s.gold_oz -= gold_needed
        s.inventory[choice] = s.inventory.get(choice, 0) + 1
        s.cost_basis = s.gold_price * gold_needed + labor_cost
        s.product_for_sale = choice

        self._r2 = compute_r2(choice, s.demand)

        # Generate customer offer based on demand and cost basis
        demand_factor = s.demand.get(choice, 0.5)
        offer_ratio = random.uniform(OFFER_MIN_RATIO, OFFER_MAX_RATIO) + (demand_factor * DEMAND_OFFER_BONUS)
        base_offer = round(s.cost_basis * offer_ratio, 2)
        s.base_offer = base_offer
        s.current_offer = base_offer
        s.phase = "showroom"
        s.negotiation_round = 0

        return JewelryObservation(
            done=False,
            reward=self._r2,
            phase="showroom",
            cash=s.cash,
            gold_oz=s.gold_oz,
            gold_price=s.gold_price,
            gold_price_history=list(s.gold_price_history),
            demand=s.demand,
            product_catalog=PRODUCT_CATALOG,
            inventory=s.inventory,
            product_for_sale=choice,
            cost_basis=s.cost_basis,
            current_offer=s.current_offer,
            negotiation_round=0,
            message=(
                f"Crafted a {choice}! Cost basis: ${s.cost_basis:.2f} "
                f"(gold ${s.gold_price * gold_needed:.2f} + labor ${labor_cost}). "
                f"Demand for {choice}: {demand_factor:.0%}. "
                f"A customer offers ${s.current_offer:.2f}. Accept, counter, or reject?"
            ),
        )

    # ── PHASE 3: SHOWROOM ──────────────────────

    def _step_showroom(self, action: JewelryAction) -> JewelryObservation:
        s = self._state

        # No product → episode ends immediately
        if s.product_for_sale is None:
            return JewelryObservation(
                done=True,
                reward=combined_reward(self._r1, self._r2, 0.0),
                phase="showroom",
                cash=s.cash,
                gold_oz=s.gold_oz,
                gold_price=s.gold_price,
                demand=s.demand,
                product_catalog=PRODUCT_CATALOG,
                inventory=s.inventory,
                product_for_sale=None,
                cost_basis=0.0,
                message="No products to sell. Episode over.",
            )

        message = action.message or ""
        intent = detect_intent(message)

        # ── ACCEPT ──
        if intent == "accept":
            r3 = compute_r3(s.current_offer, s.cost_basis)
            final_reward = combined_reward(self._r1, self._r2, r3)
            s.cash += s.current_offer
            s.inventory[s.product_for_sale] -= 1
            product_sold = s.product_for_sale
            s.product_for_sale = None

            return JewelryObservation(
                done=True,
                reward=final_reward,
                phase="showroom",
                cash=s.cash,
                gold_oz=s.gold_oz,
                gold_price=s.gold_price,
                demand=s.demand,
                product_catalog=PRODUCT_CATALOG,
                inventory=s.inventory,
                product_for_sale=None,
                cost_basis=s.cost_basis,
                current_offer=s.current_offer,
                negotiation_round=s.negotiation_round,
                message=(
                    f"Deal! Sold {product_sold} for ${s.current_offer:.2f}. "
                    f"Profit: ${s.current_offer - s.cost_basis:.2f}. "
                    f"Final reward: {final_reward}."
                ),
            )

        # ── REJECT ──
        if intent == "reject":
            final_reward = combined_reward(self._r1, self._r2, 0.0)
            return JewelryObservation(
                done=True,
                reward=final_reward,
                phase="showroom",
                cash=s.cash,
                gold_oz=s.gold_oz,
                gold_price=s.gold_price,
                demand=s.demand,
                product_catalog=PRODUCT_CATALOG,
                inventory=s.inventory,
                product_for_sale=s.product_for_sale,
                cost_basis=s.cost_basis,
                current_offer=s.current_offer,
                negotiation_round=s.negotiation_round,
                message=(
                    f"You rejected the offer. Customer left. "
                    f"Final reward: {final_reward}."
                ),
            )

        # ── COUNTER ──
        s.negotiation_round += 1

        if s.negotiation_round >= MAX_NEGOTIATION:
            final_reward = combined_reward(self._r1, self._r2, 0.0)
            return JewelryObservation(
                done=True,
                reward=final_reward,
                phase="showroom",
                cash=s.cash,
                gold_oz=s.gold_oz,
                gold_price=s.gold_price,
                demand=s.demand,
                product_catalog=PRODUCT_CATALOG,
                inventory=s.inventory,
                product_for_sale=s.product_for_sale,
                cost_basis=s.cost_basis,
                current_offer=s.current_offer,
                negotiation_round=s.negotiation_round,
                message=(
                    f"Customer left after {MAX_NEGOTIATION} rounds. "
                    f"Final reward: {final_reward}."
                ),
            )

        # Customer raises offer by 5%
        s.current_offer = round(s.current_offer * COUNTER_BUMP, 2)

        return JewelryObservation(
            done=False,
            reward=0.0,
            phase="showroom",
            cash=s.cash,
            gold_oz=s.gold_oz,
            gold_price=s.gold_price,
            demand=s.demand,
            product_catalog=PRODUCT_CATALOG,
            inventory=s.inventory,
            product_for_sale=s.product_for_sale,
            cost_basis=s.cost_basis,
            current_offer=s.current_offer,
            negotiation_round=s.negotiation_round,
            message=(
                f"Customer raises to ${s.current_offer:.2f} "
                f"(round {s.negotiation_round}/{MAX_NEGOTIATION}). "
                f"Accept, counter, or reject?"
            ),
        )

    # ── STATE PROPERTY ─────────────────────────

    @property
    def state(self) -> JewelryState:
        return self._state