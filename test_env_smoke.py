"""Standalone, no-LLM, no-server smoke tests for the JewelryShop env.

Run from inside the ShopManagerEng folder so `models` / `server` import.

Usage:
    cd ShopManagerEng
    SHOPMANAGER_MARKET_MODE=synthetic python test_env_smoke.py            # default: A+B
    SHOPMANAGER_MARKET_MODE=real      python test_env_smoke.py live       # C only
    python test_env_smoke.py all                                          # all three

Why a script: putting `f'gold_price=${o.gold_price}/oz'` inside `bash -c "..."`
makes bash try to expand `${o.gold_price}` as a shell variable and crash with
`bad substitution`. Running it from a .py file removes that whole class of bugs.
"""

from __future__ import annotations

import os
import sys

from server.ShopManagerEng_environment import JewelryShopEnvironment
from models import JewelryAction


def test_reward_path() -> None:
    """A. Synthetic episode end-to-end, prints per-step partial + cumulative."""
    print("\n=== A. reward path (synthetic) ===")
    os.environ["SHOPMANAGER_MARKET_MODE"] = "synthetic"
    e = JewelryShopEnvironment()
    o = e.reset(seed=42, task_id="market_timing", starting_cash=10000.0)
    print(
        f"  reset: phase={o.phase} cash=${o.cash} gold={o.gold_oz}oz "
        f"price=${o.gold_price} weights={o.weights}"
    )

    actions = [
        JewelryAction(market_action="buy", gold_qty=1.0),
        JewelryAction(product_choice="ring"),
        JewelryAction(message="I accept"),
    ]
    for i, a in enumerate(actions, 1):
        o = e.step(a)
        print(
            f"  step {i}: phase={o.phase} reward={o.reward:.4f} "
            f"cum={o.cumulative_reward:.4f} done={o.done}"
        )
    print(f"  FINAL cum={o.cumulative_reward:.4f} (must be in [0, 1])")


def test_bounce_loop() -> None:
    """B. Warehouse cannot craft -> agent is bounced back to MARKET."""
    print("\n=== B. bounce loop (warehouse -> market) ===")
    os.environ["SHOPMANAGER_MARKET_MODE"] = "synthetic"
    e = JewelryShopEnvironment()
    o = e.reset(seed=42, task_id="profit_negotiator", starting_cash=10000.0)

    o = e.step(JewelryAction(market_action="buy", gold_qty=0.2))
    print(f"  bought 0.2oz: phase={o.phase}, gold={o.gold_oz}oz")

    o = e.step(JewelryAction(product_choice="ring"))
    print(
        f"  tried ring: phase={o.phase} reentries={o.market_reentries}"
        f"/{o.max_market_reentries} urgent={o.inventory_urgent} "
        f"cannot_wait={o.cannot_wait}"
    )
    assert o.phase == "market", "expected bounce back to market"
    assert o.inventory_urgent is True, "expected urgent flag"

    o = e.step(JewelryAction(market_action="wait"))
    print(f"  tried wait while urgent: phase={o.phase} (should still be market)")
    assert o.phase == "market", "wait should be blocked when urgent"

    o = e.step(JewelryAction(market_action="buy", gold_qty=1.0))
    print(f"  bought 1.0oz more: phase={o.phase} gold={o.gold_oz}oz")

    o = e.step(JewelryAction(product_choice="ring"))
    print(f"  craft ring: phase={o.phase} cum={o.cumulative_reward:.4f}")

    o = e.step(JewelryAction(message="I accept"))
    print(f"  FINAL cum={o.cumulative_reward:.4f}")


def test_live_quote() -> None:
    """C. Real mode: live yfinance gold price (needs network)."""
    print("\n=== C. live yfinance quote (real mode) ===")
    os.environ["SHOPMANAGER_MARKET_MODE"] = "real"
    e = JewelryShopEnvironment()
    o = e.reset(seed=0, task_id="market_timing", starting_cash=10000.0)
    print(f"  gold_price=${o.gold_price}/oz   source={o.gold_price_source}")
    print(f"  market_mode={o.market_mode}")
    print(f"  history(last 5)={o.gold_price_history[-5:]}")


def main(argv: list[str]) -> None:
    arg = (argv[1] if len(argv) > 1 else "default").lower()

    if arg in ("default", "ab"):
        test_reward_path()
        test_bounce_loop()
    elif arg == "live":
        test_live_quote()
    elif arg == "all":
        test_reward_path()
        test_bounce_loop()
        test_live_quote()
    elif arg == "a":
        test_reward_path()
    elif arg == "b":
        test_bounce_loop()
    elif arg == "c":
        test_live_quote()
    else:
        print(f"unknown arg: {arg}. use one of: a / b / c / ab / live / all")
        sys.exit(2)


if __name__ == "__main__":
    main(sys.argv)
