from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from .models import JewelryAction, JewelryObservation, JewelryState, PRODUCT_CATALOG


class JewelryShopEnv(EnvClient[JewelryAction, JewelryObservation, JewelryState]):
    """
    Client for the Jewelry Shop RL environment.

    Usage:
        env = JewelryShopEnv(base_url="http://localhost:8000")
        obs = await env.reset()

        # Phase 1 — Market (buy or wait)
        obs = await env.step(JewelryAction(market_action="wait"))
        obs = await env.step(JewelryAction(market_action="buy", gold_qty=2.0))

        # Phase 2 — Warehouse (choose product)
        obs = await env.step(JewelryAction(product_choice="ring"))

        # Phase 3 — Showroom (negotiate)
        obs = await env.step(JewelryAction(message="How about $600?"))
        obs = await env.step(JewelryAction(message="I accept"))
    """

    # ── 1. PACK action → dict (sent TO server) ──────────────────────────────

    def _step_payload(self, action: JewelryAction) -> dict:
        payload = {}

        if action.market_action is not None:
            payload["market_action"] = action.market_action

        if action.gold_qty is not None:
            payload["gold_qty"] = action.gold_qty

        if action.product_choice is not None:
            payload["product_choice"] = action.product_choice

        if action.message is not None:
            payload["message"] = action.message

        return payload

    # ── 2. UNPACK dict → typed observation (received FROM server) ───────────

    def _parse_result(self, payload: dict) -> StepResult:
        obs_data = payload.get("observation", {})

        observation = JewelryObservation(
            # Base fields
            done=payload.get("done", False),
            reward=payload.get("reward", None),

            # Phase info
            phase=obs_data.get("phase", "market"),

            # Finances & inventory
            cash=obs_data.get("cash", 1000.0),
            gold_oz=obs_data.get("gold_oz", 0.0),

            # Market
            gold_price=obs_data.get("gold_price", 0.0),
            gold_price_history=obs_data.get("gold_price_history", []),
            market_round=obs_data.get("market_round", 0),
            max_market_rounds=obs_data.get("max_market_rounds", 3),

            # Warehouse
            demand=obs_data.get("demand", {}),
            product_catalog=obs_data.get("product_catalog", PRODUCT_CATALOG),
            inventory=obs_data.get("inventory", {}),

            # Showroom
            product_for_sale=obs_data.get("product_for_sale", None),
            cost_basis=obs_data.get("cost_basis", 0.0),
            current_offer=obs_data.get("current_offer", None),
            negotiation_round=obs_data.get("negotiation_round", 0),

            # Feedback
            message=obs_data.get("message", ""),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", None),
            done=payload.get("done", False),
        )

    # ── 3. UNPACK dict → typed state (server internal state) ────────────────

    def _parse_state(self, payload: dict) -> JewelryState:
        return JewelryState(
            episode_id=payload.get("episode_id", None),
            step_count=payload.get("step_count", 0),

            cash=payload.get("cash", 1000.0),
            gold_oz=payload.get("gold_oz", 0.0),
            gold_price=payload.get("gold_price", 0.0),
            gold_price_history=payload.get("gold_price_history", []),
            market_round=payload.get("market_round", 0),

            demand=payload.get("demand", {}),
            inventory=payload.get("inventory", {}),

            phase=payload.get("phase", "market"),
            product_for_sale=payload.get("product_for_sale", None),
            cost_basis=payload.get("cost_basis", 0.0),
            negotiation_round=payload.get("negotiation_round", 0),
            current_offer=payload.get("current_offer", 0.0),
            base_offer=payload.get("base_offer", 0.0),
            lowest_price_seen=payload.get("lowest_price_seen", 0.0),
        )