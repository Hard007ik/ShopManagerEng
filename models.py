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
    """
    market_action:  Optional[str]   = None   # "buy" or "wait"
    gold_qty:       Optional[float] = None   # How many oz to buy (market phase)
    product_choice: Optional[str]   = None   # "ring" / "necklace" / "bracelet"
    message:        Optional[str]   = None   # Showroom negotiation text


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
    gold_price_history: List[float] = []       # Last N prices for trend analysis
    market_round:       int = 0                # Current round in market (0-indexed)
    max_market_rounds:  int = 3                # Max rounds before forced decision

    # Warehouse phase
    demand:             Dict[str, float] = {}  # Demand level per product (0-1)
    product_catalog:    Dict[str, dict] = {}   # Gold/labor costs per product
    inventory:          Dict[str, int] = {}    # Crafted products in stock

    # Showroom phase
    product_for_sale:   Optional[str] = None   # Which product is being sold
    cost_basis:         float = 0.0            # Total cost to make the product
    current_offer:      Optional[float] = None # Customer's live offer
    negotiation_round:  int = 0                # Counter-offer rounds so far

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

    demand:             Dict[str, float] = {}
    inventory:          Dict[str, int] = {}

    phase:              str = "market"
    product_for_sale:   Optional[str] = None
    cost_basis:         float = 0.0
    negotiation_round:  int = 0
    current_offer:      float = 0.0
    base_offer:         float = 0.0            # Hidden from agent
    lowest_price_seen:  float = 0.0            # For r1 scoring