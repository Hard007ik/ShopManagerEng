"""Parse free-form model text into a typed JewelryAction.

Mirrors inference.py:get_action_from_text so the action surface during
training matches what was used during evaluation.
"""
from __future__ import annotations

from typing import Tuple

try:
    from ..models import JewelryAction
except ImportError:
    from models import JewelryAction


def parse_model_text_to_action(phase: str, text: str) -> Tuple[JewelryAction, str]:
    """Return (action, normalised_text) for the current phase.

    Robust against typical LLM output noise: backticks, quotes, leading/trailing
    whitespace. Falls back to safe defaults so a single bad token never breaks
    the rollout.
    """
    text = (text or "").strip().replace("`", "").strip(" \t\n\r\"'")

    if phase == "market":
        lower = text.lower()
        if lower.startswith("buy"):
            qty_str = lower.replace("buy", "").strip()
            try:
                qty = float(qty_str)
            except ValueError:
                qty = 1.0
            return JewelryAction(market_action="buy", gold_qty=qty), f"buy {qty}"
        if "wait" in lower:
            return JewelryAction(market_action="wait"), "wait"
        try:
            qty = float(text)
            return JewelryAction(market_action="buy", gold_qty=qty), f"buy {qty}"
        except ValueError:
            return JewelryAction(market_action="wait"), "wait"

    if phase == "warehouse":
        lower = text.lower()
        for product in ("necklace", "bracelet", "ring"):
            if product in lower:
                return JewelryAction(product_choice=product), product
        return JewelryAction(product_choice="ring"), "ring"

    if phase == "showroom":
        return JewelryAction(message=text), text

    return JewelryAction(), text
