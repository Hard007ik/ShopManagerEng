"""System prompt + per-turn user-prompt builder for the JewelryShop env.

Logic mirrors `inference.py`'s `build_user_prompt` so that whatever the model
saw during inference evaluation it also sees during training. Kept as a plain
sync function (no asyncio) so it composes cleanly with TRL's rollout_func.
"""
from __future__ import annotations

import math
import textwrap
from typing import List


SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert agent running a jewelry shop. The episode runs in 3 phases
    and may loop back to MARKET if the warehouse runs out of gold. The episode
    reward is the SUM of per-step partial rewards across the whole episode and
    is bounded in [0, 1]. Each task weights the phases differently:
      - market_timing     -> phase 1 = 0.6, phase 2 = 0.2, phase 3 = 0.2
      - demand_crafter    -> phase 1 = 0.2, phase 2 = 0.6, phase 3 = 0.2
      - profit_negotiator -> phase 1 = 0.2, phase 2 = 0.2, phase 3 = 0.6

    ## Phase 1: MARKET (buy / wait)
    Two modes:
      - synthetic mode: gold price moves randomly each WAIT step within a round cap.
      - real mode:      gold price comes from a live source (yfinance: GC=F),
                        no round cap; WAIT just refreshes the live quote.
    Coordination from the warehouse:
      - inventory_urgent=True / cannot_wait=True means you MUST buy now;
        WAIT will be blocked. Submit "buy X.XX" with an affordable troy-oz qty.
    Behavior:
      - If you can wait, observe the price trend in gold_price_history before buying.
      - Reserve cash for labor (ring=$200, necklace=$300, bracelet=$100).
      - Respond: "buy X.XX" (troy oz of gold) or "wait".

    ## Phase 2: WAREHOUSE (choose product)
    You see two demand fields:
      - demand          : the TRUE per-product demand for THIS episode (ground truth).
      - demand_forecast : a NOISY signal you can also lean on for planning.
    Products: ring (1oz + $200), necklace (2oz + $300), bracelet (0.5oz + $100).
    If you don't have enough gold to craft your choice, the env may BOUNCE you back
    to MARKET to buy more (up to max_market_reentries times). After max bounces or
    when truly broke, the customer leaves and the episode ends.
    Respond: "ring", "necklace", or "bracelet".

    ## Phase 3: SHOWROOM (negotiate)
    The customer makes an offer; if you counter, they raise it ~5% per round,
    up to 5 rounds. After 5 rounds with no acceptance, the customer leaves
    (no phase-3 reward). Reject also gives 0 phase-3 reward.
    Respond: "I accept" or a counter like "How about $X?". NEVER explicitly reject.

    CRITICAL: Respond with ONLY the action value. No explanations.
    """
).strip()


def build_user_prompt(step: int, obs, last_reward: float, history: List[str]) -> str:
    """Format a single observation into a user prompt the LLM sees this turn.

    Mirrors inference.py:build_user_prompt so the model sees the same input shape
    during training and at evaluation time.
    """
    history_block = "\n".join(history[-4:]) if history else "None"

    if obs.phase == "market":
        prices = getattr(obs, "gold_price_history", []) or []
        trend = ""
        if len(prices) >= 2:
            if prices[-1] < prices[-2]:
                trend = "FALLING (might keep dropping, consider waiting)"
            else:
                trend = "RISING (buy now before it gets more expensive)"

        if getattr(obs, "cannot_wait", False):
            trend = (
                "URGENT: inventory needs gold now — you cannot wait; buy at the current "
                "live quote with an affordable gold_qty (troy oz)."
            )

        max_rounds = getattr(obs, "max_market_rounds", None)
        rounds_left = (max_rounds - getattr(obs, "market_round", 0)) if max_rounds else None

        reserve = 300.0
        gold_price = getattr(obs, "gold_price", 0) or 0
        cash = getattr(obs, "cash", 0) or 0
        if gold_price > 0:
            raw_qty = (cash - reserve) / gold_price
            suggested_qty = max(math.floor(raw_qty * 100) / 100, 0.01)
        else:
            suggested_qty = 1.0

        rl = "unlimited" if rounds_left is None else str(rounds_left)
        phase_hint = (
            f"Price: ${gold_price}/oz ({getattr(obs, 'gold_price_source', '') or 'n/a'}). "
            f"Price history: {prices}. Trend: {trend}. "
            f"Rounds / waits so far: {getattr(obs, 'market_round', 0)}; cap: {rl}. "
            f"Gold on hand: {getattr(obs, 'gold_oz', 0)} troy oz "
            f"(~{getattr(obs, 'gold_grams', 0):.2f} g). "
            f"If buying, suggested qty: {suggested_qty} oz (reserves $300 for labor). "
            f"Respond: 'buy {suggested_qty}' or 'wait'"
        )

    elif obs.phase == "warehouse":
        demand = getattr(obs, "demand", {}) or {}
        forecast = getattr(obs, "demand_forecast", {}) or {}
        best_product = max(demand, key=demand.get) if demand else "ring"
        phase_hint = (
            f"Demand (episode): ring={demand.get('ring', 0):.0%}, "
            f"necklace={demand.get('necklace', 0):.0%}, "
            f"bracelet={demand.get('bracelet', 0):.0%}. "
            f"Forecast (noisy): ring={forecast.get('ring', 0):.0%}, "
            f"necklace={forecast.get('necklace', 0):.0%}, "
            f"bracelet={forecast.get('bracelet', 0):.0%}. "
            f"Highest demand: {best_product}. "
            f"You have {getattr(obs, 'gold_oz', 0)}oz gold and "
            f"${getattr(obs, 'cash', 0)} cash. "
            f"Respond with EXACTLY: {best_product}"
        )

    elif obs.phase == "showroom":
        cost_basis = getattr(obs, "cost_basis", 0) or 0
        current_offer = getattr(obs, "current_offer", 0) or 0
        negotiation_round = getattr(obs, "negotiation_round", 0) or 0

        margin = ""
        if current_offer and cost_basis > 0:
            margin_pct = ((current_offer - cost_basis) / cost_basis) * 100
            margin = f"Margin: {margin_pct:+.1f}%. "

        should_accept = negotiation_round >= 4 or (
            current_offer and cost_basis > 0 and current_offer > cost_basis * 1.3
        )

        if should_accept:
            phase_hint = (
                f"Cost: ${cost_basis}. Offer: ${current_offer}. {margin}"
                f"Round {negotiation_round}/5. "
                f"Respond with EXACTLY: I accept"
            )
        else:
            counter_msgs = [
                "I need a better price for this quality piece",
                "That's too low, this craftsmanship deserves more",
                f"How about ${round(cost_basis * 1.4, 2)}?",
                f"I can't go below ${round(cost_basis * 1.3, 2)}",
            ]
            msg = counter_msgs[min(negotiation_round, len(counter_msgs) - 1)]
            phase_hint = (
                f"Cost: ${cost_basis}. Offer: ${current_offer}. {margin}"
                f"Round {negotiation_round}/5. "
                f"DO NOT ACCEPT. Counter-offer. "
                f"Respond with EXACTLY: {msg}"
            )
    else:
        phase_hint = ""

    return textwrap.dedent(
        f"""
        Step: {step} | Phase: {obs.phase} | Last reward: {last_reward:.2f}
        Cash: ${getattr(obs, 'cash', 0)} | Gold: {getattr(obs, 'gold_oz', 0)}oz | Rings: {getattr(obs, 'inventory', {})}
        Gold Price: ${getattr(obs, 'gold_price', 0)}/oz
        Env Message: {getattr(obs, 'message', '')}

        {phase_hint}

        History: {history_block}
        """
    ).strip()
