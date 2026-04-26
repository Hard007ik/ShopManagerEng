"""Module 5 (TRL OpenEnv Wordle) style rollout for ShopManagerEng.

Two public symbols:

* ``rollout_once(...)``  — plays a single multi-turn jewelry-shop episode
  against an already-connected sync env client and returns the per-episode
  signals TRL/GRPO needs.
* ``build_rollout_func(...)`` — closure factory that returns the
  ``rollout_func(prompts, trainer=None)`` callable handed to ``GRPOTrainer``.

The pattern (canonical for OpenEnv + TRL >= 0.17):

    sync_env = env.sync(); sync_env.connect()      # one persistent WS
    trainer = GRPOTrainer(..., rollout_func=rollout_func)
    trainer.train()
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional

try:
    from .parse_action import parse_model_text_to_action
    from .prompts import build_user_prompt
except ImportError:
    from training.parse_action import parse_model_text_to_action
    from training.prompts import build_user_prompt


# Set of valid task ids supported by openenv.yaml; first one is the default.
VALID_TASKS = ("market_timing", "demand_crafter", "profit_negotiator")
_TASK_RE = re.compile(r"\[TASK=(\w+)\]")


def extract_task_id(prompt_text: str, default: str = VALID_TASKS[0]) -> str:
    """Pull the [TASK=...] tag the dataset embeds, or fall back to the default."""
    m = _TASK_RE.search(prompt_text or "")
    if not m:
        return default
    candidate = m.group(1)
    return candidate if candidate in VALID_TASKS else default


def _apply_chat_template(tokenizer, messages, model_name: str = "") -> str:
    """Apply chat template, opting out of Qwen3 'thinking' mode when applicable."""
    template_kwargs: Dict[str, Any] = {
        "add_generation_prompt": True,
        "tokenize": False,
    }
    # Qwen3 family supports the `enable_thinking` switch — disable it for short
    # action outputs. Other models silently ignore unknown kwargs in newer
    # transformers; older ones may raise, hence the lower() guard.
    if "qwen3" in (model_name or "").lower():
        template_kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(messages, **template_kwargs)


def rollout_once(
    *,
    trainer,
    sync_env,
    tokenizer,
    dataset_prompt: str,
    system_prompt: str,
    max_turns: int,
    model_name: str = "",
) -> Dict[str, Any]:
    """Play one full jewelry-shop episode and return per-episode signals.

    Returns the dict shape TRL's GRPO loop expects: ``prompt_ids``,
    ``completion_ids``, ``logprobs`` (concatenated across turns of the episode)
    plus reward signals consumed by reward functions (``total_reward``,
    ``market_reward``, ``warehouse_reward``, ``showroom_reward``).
    """
    # Late import: trl.experimental.openenv only exists for trl >= 0.17.
    from trl.experimental.openenv import generate_rollout_completions

    task_id = extract_task_id(dataset_prompt)
    result = sync_env.reset(task_id=task_id)
    obs = result.observation

    prompt_ids: List[int] = []
    completion_ids: List[int] = []
    logprobs: List[float] = []

    history: List[str] = []
    last_reward = 0.0
    phase_rewards = {"market": 0.0, "warehouse": 0.0, "showroom": 0.0}

    for turn in range(1, max_turns + 1):
        if result.done:
            break

        user_prompt = build_user_prompt(turn, obs, last_reward, history)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = _apply_chat_template(tokenizer, messages, model_name=model_name)

        rollout_outputs = generate_rollout_completions(trainer, [prompt_text])[0]
        prompt_ids.extend(rollout_outputs["prompt_ids"])
        completion_ids.extend(rollout_outputs["completion_ids"])
        logprobs.extend(rollout_outputs["logprobs"])

        completion_text = rollout_outputs.get("text") or tokenizer.decode(
            rollout_outputs["completion_ids"], skip_special_tokens=True
        )

        current_phase = obs.phase
        action, raw_action_str = parse_model_text_to_action(current_phase, completion_text)

        result = sync_env.step(action)
        obs = result.observation
        step_reward = float(result.reward or 0.0)
        last_reward = step_reward

        if current_phase in phase_rewards:
            phase_rewards[current_phase] += step_reward

        history.append(
            f"Step {turn} ({current_phase}): {raw_action_str!r} -> reward {step_reward:+.2f}"
        )

    total_reward = float(getattr(obs, "cumulative_reward", sum(phase_rewards.values())))
    total_reward = max(0.0, min(total_reward, 1.0))

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "total_reward": total_reward,
        "market_reward": float(phase_rewards["market"]),
        "warehouse_reward": float(phase_rewards["warehouse"]),
        "showroom_reward": float(phase_rewards["showroom"]),
    }


def build_rollout_func(
    *,
    sync_env,
    tokenizer,
    system_prompt: str,
    max_turns: int = 15,
    model_name: str = "",
) -> Callable[..., Dict[str, List]]:
    """Return ``rollout_func(prompts, trainer=None)`` closing over the env client.

    A fresh episode is run for each prompt; the same persistent ``sync_env``
    is reused across all prompts (single WebSocket session — matches Module 5).
    """

    def rollout_func(prompts: List[str], trainer=None) -> Dict[str, List]:
        episode_prompt_ids: List[List[int]] = []
        episode_completion_ids: List[List[int]] = []
        episode_logprobs: List[List[float]] = []
        total_rewards: List[float] = []
        market_rewards: List[float] = []
        warehouse_rewards: List[float] = []
        showroom_rewards: List[float] = []

        for prompt_text in prompts:
            ep = rollout_once(
                trainer=trainer,
                sync_env=sync_env,
                tokenizer=tokenizer,
                dataset_prompt=prompt_text,
                system_prompt=system_prompt,
                max_turns=max_turns,
                model_name=model_name,
            )
            episode_prompt_ids.append(ep["prompt_ids"])
            episode_completion_ids.append(ep["completion_ids"])
            episode_logprobs.append(ep["logprobs"])
            total_rewards.append(ep["total_reward"])
            market_rewards.append(ep["market_reward"])
            warehouse_rewards.append(ep["warehouse_reward"])
            showroom_rewards.append(ep["showroom_reward"])

        return {
            "prompt_ids": episode_prompt_ids,
            "completion_ids": episode_completion_ids,
            "logprobs": episode_logprobs,
            "total_reward": total_rewards,
            "market_reward": market_rewards,
            "warehouse_reward": warehouse_rewards,
            "showroom_reward": showroom_rewards,
        }

    return rollout_func
