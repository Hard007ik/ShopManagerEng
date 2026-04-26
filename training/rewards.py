"""Reward functions consumed by GRPOTrainer.

Design choice: ONE primary reward (``reward_total``) drives advantages, while
the three per-phase rewards are exposed for *monitoring only* via TRL's logged
reward metrics. Their config weight should be set to 0.0 to avoid double
counting the cumulative phase sum.

If you actually want phase-level shaping in the gradient, change the
GRPOConfig ``reward_weights`` to e.g. [1.0, 0.2, 0.2, 0.2].
"""
from __future__ import annotations

from typing import Any, List


def _pull(kwargs: dict, key: str, n: int) -> List[float]:
    vals = kwargs.get(key)
    if not vals:
        return [0.0] * n
    return [float(v) for v in vals]


def reward_total(completions: List[Any], **kwargs) -> List[float]:
    """Authoritative trajectory return: env's cumulative_reward in [0, 1]."""
    return _pull(kwargs, "total_reward", len(completions))


def reward_market(completions: List[Any], **kwargs) -> List[float]:
    """Sum of per-step partials emitted while phase == 'market'. Monitoring."""
    return _pull(kwargs, "market_reward", len(completions))


def reward_warehouse(completions: List[Any], **kwargs) -> List[float]:
    """Sum of per-step partials emitted while phase == 'warehouse'. Monitoring."""
    return _pull(kwargs, "warehouse_reward", len(completions))


def reward_showroom(completions: List[Any], **kwargs) -> List[float]:
    """Sum of per-step partials emitted while phase == 'showroom'. Monitoring."""
    return _pull(kwargs, "showroom_reward", len(completions))


# Convenience tuple for single-import use
ALL_REWARDS = (reward_total, reward_market, reward_warehouse, reward_showroom)

# Matching weights so only `reward_total` contributes to the GRPO advantage.
# Plug this straight into GRPOConfig(reward_weights=REWARD_WEIGHTS_MONITOR_ONLY).
REWARD_WEIGHTS_MONITOR_ONLY = [1.0, 0.0, 0.0, 0.0]
