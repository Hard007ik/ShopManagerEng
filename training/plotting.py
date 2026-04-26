"""Persist training metrics + loss/reward plots to disk.

Why this exists: the hackathon submission asks for "evidence you actually
trained — at minimum loss and reward plots from a real run." Since we run as
a script (not a notebook), nothing renders automatically. This module:

* Snapshots ``trainer.state.log_history`` every N steps via a TrainerCallback
  (so a crashed run still leaves partial evidence behind), and
* Dumps a final set of artifacts (CSV, JSON, PNGs) after ``trainer.train()``.

All artifacts land in the trainer's ``output_dir`` so they ride back to the
Hugging Face Hub when ``push_to_hub=True``.
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


# Reward keys we track. TRL logs reward functions under "rewards/<func_name>"
# (and a single-scalar "reward" = sum of weighted rewards).
PRIMARY_REWARD_KEY = "rewards/reward_total"
PHASE_REWARD_KEYS = (
    "rewards/reward_market",
    "rewards/reward_warehouse",
    "rewards/reward_showroom",
)
LOSS_KEY = "loss"
STEP_KEY = "step"


def _flatten_log_history(log_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Make sure every row carries a `step` field even when TRL omits it on epoch logs."""
    cleaned: List[Dict[str, Any]] = []
    last_step = 0
    for row in log_history:
        step = row.get("step", row.get("global_step", last_step))
        last_step = step or last_step
        merged = {"step": last_step, **{k: v for k, v in row.items() if k != "step"}}
        cleaned.append(merged)
    return cleaned


def _series(rows: List[Dict[str, Any]], key: str) -> List[tuple]:
    """Return ``[(step, value), ...]`` for the given metric key."""
    out: List[tuple] = []
    for r in rows:
        if key in r and r[key] is not None:
            try:
                out.append((int(r["step"]), float(r[key])))
            except (TypeError, ValueError):
                continue
    return out


def _save_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    columns: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                columns.append(k)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _save_json(rows: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w") as f:
        json.dump(rows, f, indent=2, default=str)


def _try_plot(
    series: Iterable[tuple],
    title: str,
    ylabel: str,
    out_path: Path,
    *,
    label: Optional[str] = None,
) -> bool:
    """Draw a single-series line plot. Silently no-ops if matplotlib is missing."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        logger.warning("matplotlib unavailable, skipping %s (%s)", out_path.name, exc)
        return False

    pts = list(series)
    if not pts:
        logger.warning("no data for %s, skipping plot", out_path.name)
        return False
    xs, ys = zip(*pts)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xs, ys, marker="o", linewidth=1.5, label=label or ylabel)
    ax.set_xlabel("training step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if label:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def _try_plot_multi(
    name_to_series: Dict[str, Iterable[tuple]],
    title: str,
    ylabel: str,
    out_path: Path,
) -> bool:
    """Draw a multi-series line plot."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        logger.warning("matplotlib unavailable, skipping %s (%s)", out_path.name, exc)
        return False

    fig, ax = plt.subplots(figsize=(8.5, 5))
    drew_any = False
    for label, pts in name_to_series.items():
        pts = list(pts)
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", linewidth=1.3, label=label)
        drew_any = True
    if not drew_any:
        plt.close(fig)
        logger.warning("no data for %s, skipping plot", out_path.name)
        return False
    ax.set_xlabel("training step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def _summary_stats(series: List[tuple]) -> Dict[str, float]:
    if not series:
        return {"final": 0.0, "max": 0.0, "min": 0.0, "mean": 0.0, "n": 0}
    ys = [v for _, v in series]
    return {
        "final": float(ys[-1]),
        "max": float(max(ys)),
        "min": float(min(ys)),
        "mean": float(sum(ys) / len(ys)),
        "n": len(ys),
    }


def save_training_artifacts(
    log_history: List[Dict[str, Any]],
    output_dir: str | Path,
    *,
    run_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Write metrics + loss/reward plots into ``output_dir``.

    Returns the summary dict that was also written to ``training_summary.json``.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = _flatten_log_history(log_history)
    _save_csv(rows, out / "metrics.csv")
    _save_json(rows, out / "metrics.json")

    loss_series = _series(rows, LOSS_KEY)
    total_reward_series = _series(rows, PRIMARY_REWARD_KEY)
    # Some TRL versions log a flat "reward" scalar in addition. Prefer the
    # named primary; fall back to "reward" if the named one is empty.
    if not total_reward_series:
        total_reward_series = _series(rows, "reward")

    phase_series = {
        "market": _series(rows, "rewards/reward_market"),
        "warehouse": _series(rows, "rewards/reward_warehouse"),
        "showroom": _series(rows, "rewards/reward_showroom"),
    }

    _try_plot(
        loss_series,
        title="Training loss (GRPO)",
        ylabel="loss",
        out_path=out / "loss_curve.png",
        label="loss",
    )
    _try_plot(
        total_reward_series,
        title="Reward (total) — env cumulative_reward in [0, 1]",
        ylabel="reward",
        out_path=out / "reward_total_curve.png",
        label="reward_total",
    )
    _try_plot_multi(
        {
            "reward_total": total_reward_series,
            **{f"reward_{k}": v for k, v in phase_series.items()},
        },
        title="Rewards over training",
        ylabel="reward",
        out_path=out / "reward_curve.png",
    )

    summary: Dict[str, Any] = {
        "loss": _summary_stats(loss_series),
        "reward_total": _summary_stats(total_reward_series),
        "reward_market": _summary_stats(phase_series["market"]),
        "reward_warehouse": _summary_stats(phase_series["warehouse"]),
        "reward_showroom": _summary_stats(phase_series["showroom"]),
        "n_log_rows": len(rows),
        "output_dir": str(out.resolve()),
    }
    if run_config is not None:
        summary["run_config"] = run_config

    with (out / "training_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("Wrote training artifacts to %s", out.resolve())
    return summary


def build_metrics_callback(output_dir: str | Path, snapshot_every: int = 5):
    """Return a TrainerCallback that snapshots metrics every N steps + on end.

    Imported lazily so this module can be inspected on a machine without
    transformers installed (e.g. for the local --smoke run).
    """
    from transformers.trainer_callback import TrainerCallback

    out = Path(output_dir)

    class MetricsSaverCallback(TrainerCallback):
        """Persist metrics CSV/JSON + plots periodically and at the end."""

        def __init__(self) -> None:
            self._last_snapshot_step = -1

        def _snapshot(self, state) -> None:
            try:
                save_training_artifacts(list(state.log_history or []), out)
            except Exception as exc:  # never let plotting kill training
                logger.warning("metrics snapshot failed: %s", exc)

        def on_log(self, args, state, control, **kwargs):
            step = int(getattr(state, "global_step", 0) or 0)
            if step <= 0:
                return control
            if (step - self._last_snapshot_step) >= max(snapshot_every, 1):
                self._snapshot(state)
                self._last_snapshot_step = step
            return control

        def on_train_end(self, args, state, control, **kwargs):
            self._snapshot(state)
            return control

    return MetricsSaverCallback()
