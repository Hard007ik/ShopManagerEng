import random
import uuid
from typing import Optional

from openenv.core.env_server import Environment

try:
    from ..constants import get_market_mode, troy_oz_to_grams
    from ..models import JewelryAction, JewelryObservation, JewelryState, PRODUCT_CATALOG
    from .market_data import last_quote_or_fallback, fetch_gold_spot_usd_per_oz
    from . import sqlite_store
except ImportError:
    # Installed: ShopManagerEng.* — otherwise dev layout: CWD=ShopManagerEng, `import server` (siblings: models, constants)
    from constants import get_market_mode, troy_oz_to_grams
    from models import JewelryAction, JewelryObservation, JewelryState, PRODUCT_CATALOG
    from server.market_data import last_quote_or_fallback, fetch_gold_spot_usd_per_oz
    from server import sqlite_store


# Legacy synthetic market (used when SHOPMANAGER_MARKET_MODE=synthetic)
STARTING_CASH = 10000.0
GOLD_PRICE_MIN = 250.0
GOLD_PRICE_MAX = 450.0
PRICE_FLUCTUATION = 0.10
MAX_MARKET_ROUNDS = 3

MAX_NEGOTIATION = 5
COUNTER_BUMP = 1.05
OFFER_MIN_RATIO = 0.80
OFFER_MAX_RATIO = 1.30
DEMAND_OFFER_BONUS = 0.20
MAX_PROFIT_MULT = 2.0

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
#  REWARD MODEL
#  All r1/r2/r3 are normalized to [0, 1].
#  Each step emits a WEIGHTED PARTIAL reward.
#  Sum of every step's reward over an episode is in [0, 1].
# ─────────────────────────────────────────────

# Per-task phase weights (w_market, w_warehouse, w_showroom). Each row sums to 1.0.
TASK_WEIGHTS = {
    "market_timing":     (0.6, 0.2, 0.2),  # Phase 1 dominates
    "demand_crafter":    (0.2, 0.6, 0.2),  # Phase 2 dominates
    "profit_negotiator": (0.2, 0.2, 0.6),  # Phase 3 dominates; phases 1 & 2 weighted equally
}
DEFAULT_TASK_ID = "profit_negotiator"


def resolve_weights(task_id: Optional[str]) -> tuple:
    tid = (task_id or DEFAULT_TASK_ID).lower().strip()
    if tid not in TASK_WEIGHTS:
        tid = DEFAULT_TASK_ID
    return TASK_WEIGHTS[tid]


def compute_r1(buy_price: float, lowest_price: float) -> float:
    """Phase 1 score in [0, 1]. 1.0 == bought at lowest seen price."""
    if lowest_price <= 0 or buy_price <= 0:
        return 0.0
    ratio = lowest_price / buy_price
    return round(min(ratio, 1.0), 4)


def compute_r2(product_choice: str, demand: dict) -> float:
    """Phase 2 score in [0, 1]. 1.0 == picked the most-demanded product."""
    if not demand or product_choice not in demand:
        return 0.0
    max_demand = max(demand.values())
    if max_demand <= 0:
        return 0.0
    return round(demand[product_choice] / max_demand, 4)


def compute_r3(accepted_price: float, cost_basis: float) -> float:
    """Phase 3 score in [0, 1]. 1.0 == hit the max profit multiple."""
    if cost_basis <= 0:
        return 0.0
    profit = accepted_price - cost_basis
    if profit <= 0:
        return 0.0
    max_profit = cost_basis * (MAX_PROFIT_MULT - 1)
    return round(min(profit / max_profit, 1.0), 4)


def step_reward(weights: tuple, phase_emitted: str, r_value: float) -> float:
    """
    Convert a normalized phase score (in [0, 1]) into the WEIGHTED partial
    reward emitted at that step. Summing these across an episode is in [0, 1].
    Guaranteed to return a Python float (never int / never None).
    """
    if phase_emitted == "market":
        return float(round(float(weights[0]) * float(r_value), 4))
    if phase_emitted == "warehouse":
        return float(round(float(weights[1]) * float(r_value), 4))
    if phase_emitted == "showroom":
        return float(round(float(weights[2]) * float(r_value), 4))
    return 0.0


def _demand_forecast_from(demand: dict) -> dict:
    """
    Noisy "forecast" for the inventory agent to plan against (same scale as demand).
    Deterministic w.r.t. the RNG in reset(seed=...) on the current episode.
    """
    out: dict = {}
    for k, v in demand.items():
        wiggle = random.uniform(-0.12, 0.12)
        out[k] = round(max(0.0, min(1.0, float(v) + wiggle)), 2)
    return out


class JewelryShopEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._state = JewelryState()
        # Normalized per-phase scores in [0, 1] (raw, before weighting)
        self._r1 = 0.0
        self._r2 = 0.0
        self._r3 = 0.0

    def _emit(self, phase_emitted: str, r_value: float) -> float:
        """
        Convert a normalized phase score into the per-step weighted reward,
        update cumulative bookkeeping, and return the value to attach to obs.
        Guaranteed: returned value AND s.cumulative_reward are Python floats.
        """
        s = self._state
        weights = tuple(s.weights) if s.weights else resolve_weights(s.task_id)
        partial = float(step_reward(weights, phase_emitted, r_value))
        s.cumulative_reward = float(round(float(s.cumulative_reward) + partial, 4))
        s.last_phase_emitted_reward = partial
        return partial

    def _apply_action_inventory_fields(self, action: JewelryAction) -> None:
        s = self._state
        if action.inventory_urgent is not None:
            s.inventory_urgent = bool(action.inventory_urgent)
        if action.need_gold_grams is not None:
            s.need_gold_grams = action.need_gold_grams
        if action.buy_deadline_iso is not None:
            s.buy_deadline_iso = action.buy_deadline_iso

    def _mm_line(self) -> str:
        s = self._state
        if s.market_mode == "synthetic" and s.max_market_rounds and s.max_market_rounds > 0:
            return f"Market simulation rounds in this phase: {s.max_market_rounds - s.market_round} (of {s.max_market_rounds})."
        if s.max_market_rounds == 0 or s.max_market_rounds is None:
            return "No round limit: wait to refresh the quote; buy when ready."
        return f"Rounds left: {max(0, s.max_market_rounds - s.market_round)}."

    def _co_market(
        self,
        *,
        done: bool = False,
        reward: float = 0.0,
        msg: str = "",
        keep_phase: Optional[str] = None,
    ) -> dict:
        s = self._state
        ph = keep_phase or s.phase
        max_r = s.max_market_rounds
        g_oz = s.gold_oz
        # Always emit reward as a Python float so it survives JSON serialization
        # as a JSON number with a decimal point (e.g. 0.0, not 0).
        try:
            reward_f = float(reward) if reward is not None else 0.0
        except (TypeError, ValueError):
            reward_f = 0.0
        return dict(
            done=done,
            reward=reward_f,
            phase=ph,
            cash=s.cash,
            gold_oz=g_oz,
            gold_grams=round(troy_oz_to_grams(g_oz), 4),
            gold_price=s.gold_price,
            gold_price_history=list(s.gold_price_history),
            market_round=s.market_round,
            max_market_rounds=max_r,
            market_mode=s.market_mode,
            gold_price_source=s.gold_price_source,
            inventory_urgent=s.inventory_urgent,
            need_gold_grams=s.need_gold_grams,
            buy_deadline_iso=s.buy_deadline_iso,
            cannot_wait=s.inventory_urgent and ph == "market",
            market_reentries=s.market_reentries,
            max_market_reentries=s.max_market_reentries,
            demand=s.demand,
            demand_forecast=getattr(s, "demand_forecast", {}) or {},
            product_catalog=PRODUCT_CATALOG,
            inventory=s.inventory,
            product_for_sale=None if ph == "market" else s.product_for_sale,
            cost_basis=s.cost_basis if ph != "market" else 0.0,
            current_offer=None if ph == "market" else s.current_offer,
            negotiation_round=s.negotiation_round,
            task_id=s.task_id,
            weights=list(s.weights) if s.weights else list(resolve_weights(s.task_id)),
            cumulative_reward=float(s.cumulative_reward),
            message=msg,
        )

    def _obs_from(self, o: dict) -> JewelryObservation:
        try:
            _r = float(o.get("reward", 0.0)) if o.get("reward", 0.0) is not None else 0.0
        except (TypeError, ValueError):
            _r = 0.0
        try:
            _cr = float(o.get("cumulative_reward", 0.0))
        except (TypeError, ValueError):
            _cr = 0.0
        return JewelryObservation(
            done=o.get("done", False),
            reward=_r,
            phase=o.get("phase", "market"),
            cash=o.get("cash", 1000.0),
            gold_oz=o.get("gold_oz", 0.0),
            gold_grams=o.get("gold_grams", 0.0),
            gold_price=o.get("gold_price", 0.0),
            gold_price_history=o.get("gold_price_history", []),
            market_round=o.get("market_round", 0),
            max_market_rounds=o.get("max_market_rounds", 0),
            market_mode=o.get("market_mode", "real"),
            gold_price_source=o.get("gold_price_source", ""),
            inventory_urgent=o.get("inventory_urgent", False),
            need_gold_grams=o.get("need_gold_grams", None),
            buy_deadline_iso=o.get("buy_deadline_iso", None),
            cannot_wait=o.get("cannot_wait", False),
            market_reentries=o.get("market_reentries", 0),
            max_market_reentries=o.get("max_market_reentries", 2),
            demand=o.get("demand", {}),
            demand_forecast=o.get("demand_forecast", {}),
            product_catalog=o.get("product_catalog", PRODUCT_CATALOG),
            inventory=o.get("inventory", {}),
            product_for_sale=o.get("product_for_sale", None),
            cost_basis=o.get("cost_basis", 0.0),
            current_offer=o.get("current_offer", None),
            negotiation_round=o.get("negotiation_round", 0),
            task_id=o.get("task_id", DEFAULT_TASK_ID),
            weights=o.get("weights", list(resolve_weights(DEFAULT_TASK_ID))),
            cumulative_reward=_cr,
            message=o.get("message", ""),
        )

    def reset(self, seed=None, episode_id=None, **kwargs) -> JewelryObservation:
        if seed is not None:
            random.seed(seed)
        eid = episode_id or str(uuid.uuid4())
        try:
            starting_cash = float(kwargs.get("starting_cash", STARTING_CASH))
        except (TypeError, ValueError):
            starting_cash = STARTING_CASH

        inv_urgent = bool(kwargs.get("inventory_urgent", False))
        need_g = kwargs.get("need_gold_grams", None)
        if need_g is not None:
            try:
                need_g = float(need_g)
            except (TypeError, ValueError):
                need_g = None
        deadline = kwargs.get("buy_deadline_iso", None)
        if deadline is not None and not isinstance(deadline, str):
            deadline = str(deadline) if deadline is not None else None

        dem = {
            "ring": round(random.uniform(0.4, 1.0), 2),
            "necklace": round(random.uniform(0.2, 0.8), 2),
            "bracelet": round(random.uniform(0.1, 0.6), 2),
        }
        dem_fc = _demand_forecast_from(dem)
        mode = (kwargs.get("market_mode") or get_market_mode()).lower().strip()

        if mode == "synthetic":
            gp = round(random.uniform(GOLD_PRICE_MIN, GOLD_PRICE_MAX), 2)
            hist = [gp]
            mmode = "synthetic"
            src = "synthetic:random_range"
            maxr = int(kwargs.get("max_market_rounds", MAX_MARKET_ROUNDS))
            use_lots = False
        else:
            mmode = "real"
            maxr = 0
            use_lots = True
            sqlite_store.init_schema()
            try:
                q = fetch_gold_spot_usd_per_oz()
                gp = round(q.usd_per_oz, 2)
                src = q.source
            except Exception:
                gp = 2000.0
                src = "yfinance:error_fallback(2000)"
            hist = [gp]

        max_r0 = int(maxr) if mode == "synthetic" else 0
        task_id = (kwargs.get("task_id") or DEFAULT_TASK_ID).strip().lower()
        weights = resolve_weights(task_id)
        try:
            max_reentries = int(kwargs.get("max_market_reentries", 2))
            if max_reentries < 0:
                max_reentries = 0
        except (TypeError, ValueError):
            max_reentries = 2
        s = self._state = JewelryState(
            episode_id=eid,
            step_count=0,
            cash=starting_cash,
            gold_oz=0.0,
            gold_price=gp,
            gold_price_history=hist,
            market_round=0,
            max_market_rounds=max_r0,
            demand=dem,
            demand_forecast=dem_fc,
            inventory={"ring": 0, "necklace": 0, "bracelet": 0},
            phase="market",
            product_for_sale=None,
            cost_basis=0.0,
            negotiation_round=0,
            current_offer=0.0,
            base_offer=0.0,
            lowest_price_seen=gp,
            inventory_urgent=inv_urgent,
            need_gold_grams=need_g,
            buy_deadline_iso=deadline,
            use_fifo_lots=use_lots,
            gold_price_source=src,
            market_mode=mmode,
            task_id=task_id,
            weights=list(weights),
            cumulative_reward=0.0,
            last_phase_emitted_reward=0.0,
            market_reentries=0,
            max_market_reentries=max_reentries,
        )
        self._r1 = 0.0
        self._r2 = 0.0
        self._r3 = 0.0
        sstep = s.max_market_rounds if s.max_market_rounds else 0
        o = self._co_market(
            msg=(
                f"Welcome. Task='{task_id}' weights(market,warehouse,showroom)={weights}. "
                f"Gold: ${gp}/oz ({s.gold_price_source}). Cash: ${s.cash:.2f}. "
                f"Inventory need-urgent={inv_urgent}."
                f" {self._mm_line()}"
            ),
        )
        o["max_market_rounds"] = sstep
        return self._obs_from(o)

    def step(self, action: JewelryAction, timeout_s=None, **kwargs) -> JewelryObservation:
        self._state.step_count += 1
        if self._state.phase == "market":
            self._apply_action_inventory_fields(action)
        if self._state.phase == "market":
            return self._step_market(action)
        if self._state.phase == "warehouse":
            return self._step_warehouse(action)
        if self._state.phase == "showroom":
            return self._step_showroom(action)
        raise ValueError(f"Unknown phase: {self._state.phase}")

    def _refresh_real_quote(self) -> None:
        s = self._state
        if s.market_mode != "real":
            return
        try:
            q = fetch_gold_spot_usd_per_oz()
            s.gold_price = round(q.usd_per_oz, 2)
            s.gold_price_source = q.source
        except Exception as exc:  # noqa: BLE001
            fb = s.gold_price if s.gold_price > 0 else 2000.0
            q2 = last_quote_or_fallback(fb)
            s.gold_price = round(q2.usd_per_oz, 2)
            s.gold_price_source = f"{q2.source}(err:{type(exc).__name__})"
        s.gold_price_history.append(s.gold_price)
        s.lowest_price_seen = min(s.lowest_price_seen, s.gold_price) if s.lowest_price_seen else s.gold_price

    def _step_market(self, action: JewelryAction) -> JewelryObservation:
        s = self._state
        market_action = (action.market_action or "wait").lower().strip()

        if s.market_mode == "synthetic":
            return self._step_market_synthetic(action, market_action)
        return self._step_market_real(action, market_action)

    def _step_market_synthetic(self, action: JewelryAction, market_action: str) -> JewelryObservation:
        s = self._state
        if market_action == "buy":
            return self._exec_buy_synthetic_common(action, market_action)
        s.market_round += 1
        if s.market_round >= (s.max_market_rounds or MAX_MARKET_ROUNDS) and s.max_market_rounds is not None and s.max_market_rounds > 0:
            s.phase = "warehouse"
            self._r1 = 0.0
            o = self._co_market(keep_phase="warehouse", msg="(Synthetic) Market round limit — entering warehouse with no new purchase.")
            return self._obs_from(o)
        ch = random.uniform(-PRICE_FLUCTUATION, PRICE_FLUCTUATION)
        np = round(s.gold_price * (1 + ch), 2)
        s.gold_price = max(np, 50.0)
        s.gold_price_history.append(s.gold_price)
        s.lowest_price_seen = min(s.lowest_price_seen, s.gold_price) if s.lowest_price_seen else s.gold_price
        o = self._co_market(
            msg=f"(Synthetic) New quote ${s.gold_price}/oz. History (last 5): {s.gold_price_history[-5:]!s}. {self._mm_line()}",
        )
        return self._obs_from(o)

    def _exec_buy_synthetic_common(self, action: JewelryAction, market_action: str) -> JewelryObservation:
        return self._step_market_buy_and_advance(
            action,
            persist_db=False,
        )

    def _step_market_real(self, action: JewelryAction, market_action: str) -> JewelryObservation:
        s = self._state
        self._refresh_real_quote()
        if market_action != "buy":
            if s.inventory_urgent:
                o = self._co_market(
                    msg="Urgent (inventory): you must not wait. Submit market_action=buy with a gold_qty you can afford at the current live quote, or 0.01 if testing.",
                )
                return self._obs_from(o)
            s.market_round += 1
            o = self._co_market(
                msg=f"Quote refreshed. Gold ${s.gold_price}/oz from {s.gold_price_source}. {self._mm_line()} Rounds so far: {s.market_round}.",
            )
            return self._obs_from(o)
        return self._step_market_buy_and_advance(action, persist_db=True)

    def _step_market_buy_and_advance(self, action: JewelryAction, *, persist_db: bool) -> JewelryObservation:
        s = self._state
        market_action = "buy"
        gold_qty = action.gold_qty
        if gold_qty is None or float(gold_qty) <= 0:
            o = self._co_market(
                msg="Buy failed: set gold_qty to a positive number of troy oz.",
            )
            return self._obs_from(o)
        gold_qty = float(gold_qty)
        price = s.gold_price
        total_cost = gold_qty * price
        if total_cost > s.cash:
            o = self._co_market(
                msg=f"Not enough cash: need ${total_cost:.2f} for {gold_qty}oz @ ${price}, have ${s.cash:.2f}.",
            )
            return self._obs_from(o)
        fund_before = s.cash
        s.cash -= total_cost
        s.gold_oz += gold_qty
        s.phase = "warehouse"
        # The bounce signal was satisfied by this purchase; clear it so the next
        # warehouse failure (if any) can emit a fresh urgency.
        s.inventory_urgent = False
        s.need_gold_grams = None
        # Only score r1 on the FIRST market visit; bounce-back buys are loop-recovery,
        # not "good price hunting", so they shouldn't pay phase-1 reward again.
        if s.market_reentries == 0:
            self._r1 = compute_r1(s.gold_price, s.lowest_price_seen) if s.lowest_price_seen else 0.0
            market_partial = self._emit("market", self._r1)
        else:
            self._r1 = 0.0
            market_partial = self._emit("market", 0.0)
        eid = getattr(s, "episode_id", None) or "unknown"
        if persist_db and s.use_fifo_lots and eid != "unknown":
            try:
                sqlite_store.record_gold_purchase(
                    eid,
                    "GOLD",
                    price,
                    gold_qty,
                    round(total_cost, 2),
                    "BUY",
                    action.ai_confidence_pct,
                    action.ai_reasoning,
                    action.target_price_usd,
                    fund_before,
                    s.cash,
                )
            except Exception as exc:  # noqa: BLE001
                s.gold_price_source = f"{s.gold_price_source} | db_log_failed:{type(exc).__name__}"
        o = self._co_market(
            reward=market_partial,
            keep_phase="warehouse",
            msg=(
                f"Bought {gold_qty} troy oz at ${price}/oz ($ {total_cost:.2f}). "
                f"Cash ${s.cash:.2f}. {self._mm_line()} "
                f"Phase reward(r1={self._r1:.4f} * w_market={s.weights[0]})={market_partial:.4f}. "
                f"Cumulative={s.cumulative_reward:.4f}. Choose a product in the warehouse."
            ),
        )
        return self._obs_from(o)

    def _can_afford_smallest_buy(self) -> bool:
        """
        Loop guard: are we even theoretically able to buy *some* useful gold?
        We require cash >= price * smallest product's gold need (i.e. enough
        for at least one bracelet's worth of gold). If not, bouncing back to
        market is wasteful and we should stop the loop.
        """
        s = self._state
        if s.gold_price <= 0:
            return False
        cheapest_gold_oz = min(spec["gold_oz"] for spec in PRODUCT_CATALOG.values())
        return s.cash >= s.gold_price * cheapest_gold_oz

    def _bounce_to_market(self, choice: str, grams_needed: float, reason: str) -> JewelryObservation:
        """
        Inventory -> Market loop: send the agent back to the market phase to
        buy more gold, with urgency flags so the market step won't allow waits.
        Emits 0.0 reward; final episode score still bounded in [0, 1].
        """
        s = self._state
        s.market_reentries += 1
        s.phase = "market"
        s.market_round = 0  # fresh patience counter for this re-entry
        s.inventory_urgent = True
        s.need_gold_grams = round(grams_needed, 4)
        bounce_partial = self._emit("warehouse", 0.0)
        o = self._co_market(reward=bounce_partial, keep_phase="market")
        o["message"] = (
            f"Inventory needs more gold to craft {choice} ({reason}). "
            f"Bouncing back to MARKET (re-entry {s.market_reentries}/{s.max_market_reentries}). "
            f"Need ~{grams_needed:.2f} g. inventory_urgent=True; market_action='wait' will be blocked. "
            f"Cumulative={s.cumulative_reward:.4f}."
        )
        o["product_for_sale"] = None
        o["current_offer"] = None
        o["cost_basis"] = 0.0
        return self._obs_from(o)

    def _step_warehouse(self, action: JewelryAction) -> JewelryObservation:
        s = self._state
        choice = (action.product_choice or "ring").lower().strip()
        if choice not in PRODUCT_CATALOG:
            choice = "ring"
        spec = PRODUCT_CATALOG[choice]
        gold_needed_oz = spec["gold_oz"]
        labor_cost = spec["labor"]
        grams_needed = troy_oz_to_grams(gold_needed_oz)

        has_gold_oz = s.gold_oz + 1e-8 >= gold_needed_oz
        if not has_gold_oz:
            # Inventory -> market loop: try to buy more gold if budget + bounces remain.
            if (
                s.market_reentries < s.max_market_reentries
                and self._can_afford_smallest_buy()
            ):
                return self._bounce_to_market(
                    choice,
                    grams_needed,
                    reason=f"have {s.gold_oz:.4f} oz, need {gold_needed_oz:.4f} oz",
                )
            # Out of bounces or no money: customer leaves, episode ends with no sale.
            self._r2 = 0.0
            s.phase = "showroom"
            o = {**self._co_market(keep_phase="showroom", reward=0.0, msg="")}
            why = "no bounce-backs left" if s.market_reentries >= s.max_market_reentries else "not enough cash to buy any gold"
            o["message"] = (
                f"Cannot craft {choice}: insufficient gold and {why}. "
                f"Customer walks away. Cumulative={s.cumulative_reward:.4f}."
            )
            o["product_for_sale"] = None
            o["current_offer"] = None
            o["cost_basis"] = 0.0
            return self._obs_from(o)
        if s.cash < labor_cost:
            self._r2 = 0.0
            s.phase = "showroom"
            o = {**self._co_market(keep_phase="showroom", reward=0.0, msg="")}
            o["message"] = (
                f"Cannot craft {choice}: have gold but no cash for labor (${labor_cost:.2f}). "
                f"Cumulative={s.cumulative_reward:.4f}."
            )
            o["product_for_sale"] = None
            o["current_offer"] = None
            o["cost_basis"] = 0.0
            return self._obs_from(o)

        s.cash -= labor_cost
        eid = getattr(s, "episode_id", None) or "unknown"
        if s.use_fifo_lots and s.market_mode == "real" and eid != "unknown":
            ok, gold_cost, _d = sqlite_store.fifo_consume_grams(eid, grams_needed)
            if not ok:
                s.cash += labor_cost
                self._r2 = 0.0
                s.phase = "showroom"
                o_ = {**self._co_market(keep_phase="showroom", reward=0.0, msg="")}
                o_["message"] = "FIFO: not enough gold lots in the database for this episode (or oz/gram mismatch)."
                o_["product_for_sale"] = None
                o_["current_offer"] = None
                o_["cost_basis"] = 0.0
                return self._obs_from(o_)
            s.gold_oz -= gold_needed_oz
            s.inventory[choice] = s.inventory.get(choice, 0) + 1
            s.product_for_sale = choice
            s.cost_basis = float(gold_cost) + float(labor_cost)
        else:
            s.gold_oz -= gold_needed_oz
            s.inventory[choice] = s.inventory.get(choice, 0) + 1
            s.product_for_sale = choice
            s.cost_basis = s.gold_price * gold_needed_oz + labor_cost
        self._r2 = compute_r2(choice, s.demand)
        warehouse_partial = self._emit("warehouse", self._r2)
        dmf = s.demand.get(choice, 0.5)
        offer_ratio = random.uniform(OFFER_MIN_RATIO, OFFER_MAX_RATIO) + (dmf * DEMAND_OFFER_BONUS)
        s.base_offer = round(s.cost_basis * offer_ratio, 2)
        s.current_offer = s.base_offer
        s.phase = "showroom"
        s.negotiation_round = 0
        o2 = {**self._co_market(keep_phase="showroom")}
        o2["reward"] = warehouse_partial
        o2["product_for_sale"] = choice
        o2["cost_basis"] = s.cost_basis
        o2["current_offer"] = s.current_offer
        _cost_label = (
            "FIFO (SQLite lots) gold + labor"
            if s.use_fifo_lots and s.market_mode == "real" and eid != "unknown"
            else "market gold + labor"
        )
        o2["message"] = (
            f"Crafted {choice}. Cost ({_cost_label}): ${s.cost_basis:.2f}. "
            f"Phase reward(r2={self._r2:.4f} * w_warehouse={s.weights[1]})={warehouse_partial:.4f}. "
            f"Cumulative={s.cumulative_reward:.4f}. Customer offers ${s.current_offer:.2f}."
        )
        return self._obs_from(o2)

    def _step_showroom(self, action: JewelryAction) -> JewelryObservation:
        s = self._state
        if s.product_for_sale is None:
            self._r3 = 0.0
            showroom_partial = self._emit("showroom", 0.0)
            o3 = {**self._co_market(done=True, reward=showroom_partial, keep_phase="showroom")}
            o3["message"] = (
                "No products to sell. Episode over. "
                f"Phase reward(r3=0 * w_showroom={s.weights[2]})=0.0000. "
                f"Cumulative={s.cumulative_reward:.4f}."
            )
            o3["product_for_sale"] = None
            o3["current_offer"] = s.current_offer
            return self._obs_from(o3)
        message = action.message or ""
        intent = detect_intent(message)
        if intent == "accept":
            self._r3 = compute_r3(s.current_offer, s.cost_basis)
            showroom_partial = self._emit("showroom", self._r3)
            s.cash += s.current_offer
            s.inventory[s.product_for_sale] -= 1
            _ps = s.product_for_sale
            s.product_for_sale = None
            o4 = {**self._co_market(done=True, reward=showroom_partial, keep_phase="showroom")}
            o4["message"] = (
                f"Sold {_ps} for ${s.current_offer:.2f}. "
                f"Phase reward(r3={self._r3:.4f} * w_showroom={s.weights[2]})={showroom_partial:.4f}. "
                f"Cumulative(final)={s.cumulative_reward:.4f}."
            )
            o4["product_for_sale"] = None
            o4["current_offer"] = s.current_offer
            return self._obs_from(o4)
        if intent == "reject":
            self._r3 = 0.0
            showroom_partial = self._emit("showroom", 0.0)
            o5 = {**self._co_market(done=True, reward=showroom_partial, keep_phase="showroom")}
            o5["message"] = (
                f"Rejected. Phase reward(r3=0 * w_showroom={s.weights[2]})=0.0000. "
                f"Cumulative(final)={s.cumulative_reward:.4f}."
            )
            o5["product_for_sale"] = s.product_for_sale
            o5["current_offer"] = s.current_offer
            return self._obs_from(o5)
        s.negotiation_round += 1
        if s.negotiation_round >= MAX_NEGOTIATION:
            self._r3 = 0.0
            showroom_partial = self._emit("showroom", 0.0)
            o6 = {**self._co_market(done=True, reward=showroom_partial, keep_phase="showroom")}
            o6["message"] = (
                f"Max negotiation rounds reached. "
                f"Phase reward(r3=0 * w_showroom={s.weights[2]})=0.0000. "
                f"Cumulative(final)={s.cumulative_reward:.4f}."
            )
            return self._obs_from(o6)
        s.current_offer = round(s.current_offer * COUNTER_BUMP, 2)
        o7 = {**self._co_market(keep_phase="showroom", reward=0.0, msg="")}
        o7["message"] = f"Customer at ${s.current_offer:.2f} (round {s.negotiation_round})."
        o7["current_offer"] = s.current_offer
        o7["product_for_sale"] = s.product_for_sale
        return self._obs_from(o7)

    @property
    def state(self) -> JewelryState:
        return self._state
