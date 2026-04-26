#!/usr/bin/env python3
"""
No-TRL baseline: run random vs simple heuristic policies; log mean return.
Run local server first:  uv run server
Then:  SHOPMANAGER_MARKET_MODE=synthetic SHOPMANAGER_TRAIN_BASE_URL=http://127.0.0.1:8000 python rollout_baseline.py
"""
from __future__ import annotations

import argparse
import asyncio
import os
import random
import sys
from pathlib import Path
from statistics import fmean, pstdev
from typing import List, Optional

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from client import JewelryShopEnv
from models import JewelryAction, PRODUCT_CATALOG


def _heuristic_action(obs) -> JewelryAction:
    ph = obs.phase
    if ph == "market":
        g = float(obs.gold_price or 0.0) or 1.0
        need = 1.0
        if obs.cash >= need * g + 10:
            return JewelryAction(
                market_action="buy", gold_qty=need, target_price_usd=obs.gold_price
            )
        return JewelryAction(market_action="wait")
    if ph == "warehouse":
        dem = obs.demand or {"ring": 0.5, "necklace": 0.3, "bracelet": 0.2}
        for name in sorted(dem, key=lambda k: dem.get(k, 0), reverse=True):
            gneed = float(PRODUCT_CATALOG[name]["gold_oz"])
            lab = float(PRODUCT_CATALOG[name]["labor"])
            if obs.gold_oz + 1e-9 >= gneed and obs.cash + 1e-9 >= lab:
                return JewelryAction(product_choice=name)
        return JewelryAction(product_choice="ring")
    if ph == "showroom":
        if (
            obs.current_offer
            and obs.cost_basis > 0
            and (float(obs.current_offer) / float(obs.cost_basis)) >= 1.15
        ) or (getattr(obs, "negotiation_round", 0) and int(obs.negotiation_round) >= 3):
            return JewelryAction(message="I accept")
        off = float(obs.current_offer or 0.0)
        return JewelryAction(
            message=f"How about ${off * 1.08:.2f}?" if off else "I need a better offer"
        )
    return JewelryAction()


def _random_action(obs) -> JewelryAction:
    if obs.phase == "market":
        if random.random() < 0.35:
            return JewelryAction(
                market_action="buy", gold_qty=round(random.uniform(0.1, 1.2), 2)
            )
        return JewelryAction(market_action="wait")
    if obs.phase == "warehouse":
        return JewelryAction(product_choice=random.choice(["ring", "necklace", "bracelet"]))
    return JewelryAction(
        message=random.choice(
            [
                "I accept",
                f"How about ${float(obs.current_offer or 0) * 1.1:.0f}?",
            ]
        )
    )


async def one_episode(base: str, policy: str, seed: Optional[int], max_steps: int) -> float:
    """
    Run one episode under the given policy and return the trajectory return,
    which is the env's cumulative reward (sum of per-step partials, in [0, 1]).
    """
    if seed is not None:
        random.seed(seed)
    env = JewelryShopEnv(base_url=base)
    r = await env.reset(seed=seed, episode_id=None)
    o = r.observation
    for _ in range(max_steps):
        if r.done:
            break
        if policy == "heuristic":
            a = _heuristic_action(o)
        else:
            a = _random_action(o)
        r = await env.step(a)
        o = r.observation
    try:
        await env.close()
    except Exception:  # noqa: BLE001
        pass
    # Authoritative trajectory return from the server (in [0, 1]).
    return float(getattr(o, "cumulative_reward", 0.0))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=25)
    p.add_argument(
        "--base-url",
        default=os.environ.get("SHOPMANAGER_TRAIN_BASE_URL", "http://127.0.0.1:8000"),
    )
    p.add_argument("--policies", nargs="+", default=["heuristic", "random"])
    p.add_argument("--out", type=Path, default=Path("rollout_metrics.txt"))
    args = p.parse_args()
    base = str(args.base_url)
    all_lines: List[str] = [f"base_url={base}", f"episodes={args.episodes} max_steps={args.max_steps}", ""]

    for name in args.policies:
        scores: List[float] = []
        for epi in range(args.episodes):
            sc = asyncio.run(
                one_episode(base, name, seed=epi, max_steps=int(args.max_steps))  # type: ignore[misc]  # noqa: E501
            )
            scores.append(sc)
        m = fmean(scores) if scores else 0.0
        sd = pstdev(scores) if len(scores) > 1 else 0.0
        line = f"{name}: mean={m:.4f} std={sd:.4f} scores={scores!s}"
        all_lines.append(line)
        print(line)
    text = "\n".join(all_lines) + "\n"
    try:
        args.out.write_text(text, encoding="utf-8")
        print(f"Wrote {args.out}", flush=True)
    except OSError as err:
        print("Could not write out file:", err, flush=True)


if __name__ == "__main__":
    main()
