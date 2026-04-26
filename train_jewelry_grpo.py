"""GRPO training entry point for the JewelryShop OpenEnv (Module 5 style).

Canonical TRL >= 0.17 + OpenEnv pattern:

  1. Persistent sync env client            (one WebSocket reused across rollouts)
  2. rollout_func(prompts, trainer)        (returns prompt_ids/completion_ids/logprobs + rewards)
  3. Pure reward_funcs(completions, ...)   (read kwargs the rollout puts there)
  4. GRPOTrainer(..., rollout_func=...)    (NO `environment_factory`)

Design notes:
- Reward shaping: ONE primary reward (`reward_total = obs.cumulative_reward`)
  drives gradients; per-phase rewards (market/warehouse/showroom) are exposed
  for monitoring with weight 0 in the GRPO advantage.
- Tasks: dataset rows embed [TASK=<task_id>] which the rollout extracts so each
  episode trains against a specific per-phase weight profile from openenv.yaml.
- Smoke: --smoke imports everything, builds the rollout func, opens the env
  and does a single reset() round-trip — no GPU and no model weights needed.
  Use this to validate wiring before paying for a GPU.

Quick local smoke (no GPU, no model load):
    python train_jewelry_grpo.py --smoke

Full local quick check (CPU; slow, but verifies trainer.train() starts):
    TRAIN_MODEL=Qwen/Qwen3-0.6B python train_jewelry_grpo.py --quick

Cloud (HF Jobs) — see README/TRAINING for the exact `hf jobs run` command.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make `from ShopManagerEng...` imports work whether you launch this from inside
# the package directory or one level up.
ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

try:
    from ShopManagerEng.client import JewelryShopEnv
    from ShopManagerEng.training.plotting import (
        build_metrics_callback,
        save_training_artifacts,
    )
    from ShopManagerEng.training.prompts import SYSTEM_PROMPT
    from ShopManagerEng.training.rewards import (
        ALL_REWARDS,
        REWARD_WEIGHTS_MONITOR_ONLY,
    )
    from ShopManagerEng.training.rollout import VALID_TASKS, build_rollout_func
except ImportError:  # script-style invocation from inside the folder
    from client import JewelryShopEnv  # type: ignore
    from training.plotting import (  # type: ignore
        build_metrics_callback,
        save_training_artifacts,
    )
    from training.prompts import SYSTEM_PROMPT  # type: ignore
    from training.rewards import (  # type: ignore
        ALL_REWARDS,
        REWARD_WEIGHTS_MONITOR_ONLY,
    )
    from training.rollout import VALID_TASKS, build_rollout_func  # type: ignore


def _build_dataset(dataset_size: int):
    """Cycle through the 3 graded tasks; embed task id in prompt for the rollout."""
    from datasets import Dataset

    rows = []
    for i in range(dataset_size):
        task_id = VALID_TASKS[i % len(VALID_TASKS)]
        rows.append(
            {
                "prompt": (
                    f"[TASK={task_id}] Manage a jewelry shop episode end-to-end. "
                    f"Maximize the {task_id} task reward."
                ),
                "task_id": task_id,
            }
        )
    return Dataset.from_list(rows)


def _resolve_precision():
    """CPU/GPU autodetect; mirror the well-tested defaults."""
    try:
        import torch
        has_cuda = bool(torch.cuda.is_available())
    except Exception:
        has_cuda = False
    if has_cuda:
        return {"bf16": True}
    return {"use_cpu": True, "bf16": False, "fp16": False}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default=os.environ.get("TRAIN_MODEL", "Qwen/Qwen3-1.7B"),
        help="HF model id (default: Qwen/Qwen3-1.7B; matches openenv-course Module 5).",
    )
    ap.add_argument(
        "--env-url",
        default=os.environ.get(
            "ENV_URL", "https://hard007ik-shopmanagereng.hf.space"
        ),
        help="Base URL of the running OpenEnv server (Space or http://127.0.0.1:8000).",
    )
    ap.add_argument(
        "--output-dir",
        default=os.environ.get("TRAIN_OUTPUT_DIR", "shopmanager-grpo-out"),
    )
    ap.add_argument("--dataset-size", type=int, default=300)
    ap.add_argument("--num-generations", type=int, default=2)
    ap.add_argument("--per-device-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=32)
    ap.add_argument("--max-completion-length", type=int, default=64)
    ap.add_argument("--max-prompt-length", type=int, default=2048)
    ap.add_argument("--max-turns", type=int, default=15)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--warmup-steps", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=-1, help="-1 = epoch-bounded.")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument(
        "--vllm-gpu-mem",
        type=float,
        default=0.3,
        help="Fraction of GPU mem reserved for vLLM. Lower if OOM.",
    )
    ap.add_argument("--push-to-hub", action="store_true")
    ap.add_argument(
        "--report-to",
        default=os.environ.get("TRAIN_REPORT_TO", "trackio"),
        help="trackio | wandb | none",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="No training. Imports, env connect, one reset() — validates wiring.",
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="CPU-friendly tiny run (1 step, num_generations=2, max_completion=32).",
    )
    args = ap.parse_args()

    if args.smoke:
        # No transformers / vllm load: just import + connect + reset, then
        # also exercise the plotting pipeline on a fake log_history so the
        # submission-artifact path is proven before we burn a GPU on it.
        env = JewelryShopEnv(base_url=args.env_url)
        sync_env = env.sync()
        sync_env.connect()
        try:
            r = sync_env.reset(task_id=VALID_TASKS[0])
            print(f"[SMOKE] connected to {args.env_url}")
            print(
                f"[SMOKE] reset OK: phase={r.observation.phase}, "
                f"done={r.done}, cumulative_reward="
                f"{getattr(r.observation, 'cumulative_reward', 0)}"
            )
            print(f"[SMOKE] system prompt loaded ({len(SYSTEM_PROMPT)} chars)")
            print("[SMOKE] reward funcs:", [f.__name__ for f in ALL_REWARDS])
            print("[SMOKE] reward weights:", REWARD_WEIGHTS_MONITOR_ONLY)
        finally:
            try:
                sync_env.close()
            except Exception:
                pass

        # Prove the plotting pipeline works (writes PNG + CSV + JSON to
        # output_dir using a synthetic, monotonic log). This is what the
        # real run will produce — same code path.
        fake_history = []
        for step in range(1, 21):
            fake_history.append(
                {
                    "step": step,
                    "loss": 1.0 / step,
                    "reward": 0.05 * step,
                    "rewards/reward_total": min(0.05 * step, 1.0),
                    "rewards/reward_market": min(0.02 * step, 0.6),
                    "rewards/reward_warehouse": min(0.015 * step, 0.6),
                    "rewards/reward_showroom": min(0.018 * step, 0.6),
                }
            )
        summary = save_training_artifacts(
            fake_history,
            args.output_dir,
            run_config={"smoke": True, "model": args.model, "env_url": args.env_url},
        )
        print(
            f"[SMOKE] plot pipeline OK -> {args.output_dir}/loss_curve.png, "
            f"{args.output_dir}/reward_curve.png, "
            f"{args.output_dir}/reward_total_curve.png"
        )
        print(
            f"[SMOKE] summary peak_reward_total={summary['reward_total']['max']:.3f} "
            f"final_loss={summary['loss']['final']:.4f}"
        )
        print("[SMOKE] OK — wiring is sane.")
        return

    # ── Heavy imports only past the smoke gate ──
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    env = JewelryShopEnv(base_url=args.env_url)
    sync_env = env.sync()
    sync_env.connect()
    print(f"[TRAIN] env: {args.env_url}")
    print(f"[TRAIN] model: {args.model}")

    rollout_func = build_rollout_func(
        sync_env=sync_env,
        tokenizer=tokenizer,
        system_prompt=SYSTEM_PROMPT,
        max_turns=args.max_turns,
        model_name=args.model,
    )

    if args.quick:
        # Override to tiny CPU-friendly numbers.
        args.dataset_size = 4
        args.num_generations = 2
        args.per_device_batch = 2
        args.grad_accum = 1
        args.max_completion_length = 32
        args.max_steps = 1
        args.warmup_steps = 0

    dataset = _build_dataset(args.dataset_size)
    precision = _resolve_precision()
    use_cpu = precision.get("use_cpu", False)

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.grad_accum,
        per_device_train_batch_size=args.per_device_batch,
        warmup_steps=args.warmup_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        # vLLM is the canonical generation backend on GPU; turn off on CPU smoke.
        use_vllm=not use_cpu,
        vllm_mode="colocate" if not use_cpu else None,
        vllm_gpu_memory_utilization=args.vllm_gpu_mem if not use_cpu else None,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        reward_weights=REWARD_WEIGHTS_MONITOR_ONLY,
        report_to=args.report_to if args.report_to != "none" else "none",
        logging_steps=1,
        save_steps=20,
        push_to_hub=args.push_to_hub,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        **precision,
    )

    print(f"[TRAIN] device={'cpu' if use_cpu else 'gpu'}  precision={precision}")
    print(f"[TRAIN] dataset_size={args.dataset_size}  num_generations={args.num_generations}")

    trainer = GRPOTrainer(
        model=args.model,
        processing_class=tokenizer,
        reward_funcs=list(ALL_REWARDS),
        train_dataset=dataset,
        args=grpo_config,
        rollout_func=rollout_func,
    )

    # Persist loss + reward plots into output_dir every N steps + at the end.
    # This is the hackathon "evidence you actually trained" artifact set.
    trainer.add_callback(build_metrics_callback(args.output_dir, snapshot_every=5))

    run_config = {
        "model": args.model,
        "env_url": args.env_url,
        "dataset_size": args.dataset_size,
        "num_generations": args.num_generations,
        "per_device_batch": args.per_device_batch,
        "grad_accum": args.grad_accum,
        "max_completion_length": args.max_completion_length,
        "max_prompt_length": args.max_prompt_length,
        "max_turns": args.max_turns,
        "lr": args.lr,
        "warmup_steps": args.warmup_steps,
        "max_steps": args.max_steps,
        "epochs": args.epochs,
        "vllm_gpu_mem": args.vllm_gpu_mem,
        "reward_weights": REWARD_WEIGHTS_MONITOR_ONLY,
        "precision": precision,
    }

    try:
        trainer.train()
    finally:
        try:
            sync_env.close()
        except Exception:
            pass
        # Always persist whatever metrics we have, even on a crash mid-run.
        try:
            summary = save_training_artifacts(
                list(trainer.state.log_history or []),
                args.output_dir,
                run_config=run_config,
            )
            print(
                f"[ARTIFACTS] wrote loss/reward plots + metrics to {args.output_dir}\n"
                f"[ARTIFACTS] final loss={summary['loss']['final']:.4f} "
                f"max_reward_total={summary['reward_total']['max']:.4f} "
                f"final_reward_total={summary['reward_total']['final']:.4f}"
            )
        except Exception as exc:
            print(f"[ARTIFACTS] failed to save metrics: {exc}")

    trainer.save_model(args.output_dir)
    if args.push_to_hub:
        trainer.push_to_hub()
    print(f"[DONE] saved to {args.output_dir}")


if __name__ == "__main__":
    main()
