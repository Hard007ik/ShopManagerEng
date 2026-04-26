from typing import Optional, Dict, List
from openenv.core.env_server import Action, Observation, State


# ─────────────────────────────────────────────
#  PRODUCT CATALOG (shared constant)
# ─────────────────────────────────────────────

PRODUCT_CATALOG = {
    "ring":     {"gold_oz": 1.0, "labor": 200.0, "base_demand": 0.8},
    "necklace": {"gold_oz": 2.0, "labor": 300.0, "base_demand": 0.5},
    "bracelet": {"gold_oz": 0.5, "labor": 100.0, "base_demand": 0.3},
}


# ─────────────────────────────────────────────
#  ACTION
#  One unified action covers all 3 phases.
# ─────────────────────────────────────────────

class JewelryAction(Action):
    """
    Phase 1 (market)    → market_action ("buy"/"wait") + gold_qty (oz to buy)
    Phase 2 (warehouse) → product_choice ("ring"/"necklace"/"bracelet")
    Phase 3 (showroom)  → message (accept / counter / reject)

    Market (optional): when logging a BUY to SQLite / invoice, the agent may send
    LLM target + reasoning; when coordinating with the inventory side, it may
    update urgency / need-by fields that were also set on reset.
    """
    market_action:  Optional[str]   = None   # "buy" or "wait"
    gold_qty:       Optional[float] = None   # How many oz to buy (market phase)
    product_choice: Optional[str]   = None   # "ring" / "necklace" / "bracelet"
    message:        Optional[str]   = None   # Showroom negotiation text

    target_price_usd:   Optional[float] = None
    ai_confidence_pct:  Optional[float] = None
    ai_reasoning:       Optional[str]   = None
    inventory_urgent:   Optional[bool]  = None
    need_gold_grams:    Optional[float] = None
    buy_deadline_iso:   Optional[str]   = None


# ─────────────────────────────────────────────
#  OBSERVATION
#  Everything the agent can SEE each step.
# ─────────────────────────────────────────────

class JewelryObservation(Observation):
    # Base fields: done, reward (inherited)

    phase:              str                    # "market" | "warehouse" | "showroom"
    cash:               float                  # Agent's current cash ($)
    gold_oz:            float                  # Raw gold in inventory (oz)

    # Market phase
    gold_price:         float                  # Current gold price ($/oz)
    gold_grams:         float = 0.0            # Raw gold in inventory (grams) — troy-oz * GRAMS_PER_TROY_OZ
    gold_price_history: List[float] = []       # Last N prices for trend analysis
    market_round:       int = 0                # "Wait" count in this episode (for analytics; no cap in real mode)
    max_market_rounds:  int = 0                # 0 = no forced round limit (real market); >0 = synthetic only
    market_mode:        str = "real"           # "real" | "synthetic"
    gold_price_source:  str = ""               # e.g. yfinance:GC=F

    # Inventory <-> market coordination (from reset / optional step updates)
    inventory_urgent:     bool = False
    need_gold_grams:   Optional[float] = None
    buy_deadline_iso:  Optional[str] = None
    cannot_wait:       bool = False            # If urgent, "wait" action is rejected

    # Inventory -> Market bounce-back (when warehouse cannot craft due to low gold)
    market_reentries:     int = 0              # How many times warehouse has sent us back to market
    max_market_reentries: int = 2              # Cap on bounce-backs to avoid infinite loops

    # Warehouse phase
    demand:             Dict[str, float] = {}  # "True" per-product demand this episode (0-1)
    demand_forecast:   Dict[str, float] = {}  # Noisy / model-facing signal (inventory "prediction" slot)
    product_catalog:    Dict[str, dict] = {}   # Gold/labor costs per product
    inventory:          Dict[str, int] = {}    # Crafted products in stock

    # Showroom phase
    product_for_sale:   Optional[str] = None   # Which product is being sold
    cost_basis:         float = 0.0            # Total cost to make the product
    current_offer:      Optional[float] = None # Customer's live offer
    negotiation_round:  int = 0                # Counter-offer rounds so far

    # Per-task grading (chosen at reset() from openenv.yaml task_id)
    task_id:            str = "profit_negotiator"
    weights:            List[float] = []       # [w_market, w_warehouse, w_showroom], sums to 1.0
    cumulative_reward:  float = 0.0            # Running sum of per-step rewards in this episode

    message:            str = ""               # Human-readable feedback


# ─────────────────────────────────────────────
#  STATE
#  Full internal state (server-side truth).
# ─────────────────────────────────────────────

class JewelryState(State):
    # Base: episode_id, step_count (inherited)

    cash:               float = 1000.0
    gold_oz:            float = 0.0
    gold_price:         float = 0.0
    gold_price_history: List[float] = []
    market_round:       int = 0
    max_market_rounds:  int = 0  # 0 = no cap (real); >0 only in synthetic mode

    demand:             Dict[str, float] = {}
    demand_forecast:   Dict[str, float] = {}
    inventory:          Dict[str, int] = {}

    phase:              str = "market"
    product_for_sale:   Optional[str] = None
    cost_basis:         float = 0.0
    negotiation_round:  int = 0
    current_offer:      float = 0.0
    base_offer:         float = 0.0            # Hidden from agent
    lowest_price_seen:  float = 0.0            # For r1 scoring

    inventory_urgent:   bool = False
    need_gold_grams:   Optional[float] = None
    buy_deadline_iso:  Optional[str] = None
    use_fifo_lots:      bool = False            # If True, warehouse cost uses per-gram lots in SQLite
    gold_price_source:  str = ""
    market_mode:        str = "real"

    # Inventory -> Market bounce-back loop
    market_reentries:     int = 0
    max_market_reentries: int = 2

    # Per-task grading (selected at reset)
    task_id:            str = "profit_negotiator"
    weights:            List[float] = []        # [w_market, w_warehouse, w_showroom]
    cumulative_reward:  float = 0.0
    last_phase_emitted_reward: float = 0.0      # Reward emitted at the most recent step (debug)