import asyncio
import math
import os
import sys
import textwrap
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

# Add parent directory to path so ShopManagerEng is importable as a package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ShopManagerEng.client import JewelryShopEnv
from ShopManagerEng.models import JewelryAction

load_dotenv()

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

# ── LLM API ─────────────────────────────────────────────────────────────────
# HuggingFace Inference Router (needs HF_TOKEN in .env)
API_BASE_URL = "https://router.huggingface.co/v1"

# ── MODEL ───────────────────────────────────────────────────────────────────
# Pick one — comment out the other
MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
# MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
# MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
# MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
# ─────────────────────────────────────────────────────────────────────────────
TASK_NAME = os.getenv("JEWELRY_ENV_TASK", "jewelry-shop")
BENCHMARK = os.getenv("JEWELRY_ENV_BENCHMARK", "jewelry_shop_benchmark")
MAX_STEPS = 15
TEMPERATURE = 0.7
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.01


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


# ── LOGGING ────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ── PROMPT BUILDING ────────────────────────────

def build_user_prompt(step: int, obs, last_reward: float, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"

    if obs.phase == "market":
        prices = obs.gold_price_history
        trend = ""
        if len(prices) >= 2:
            if prices[-1] < prices[-2]:
                trend = "FALLING ↓ (might keep dropping, consider waiting)"
            else:
                trend = "RISING ↑ (buy now before it gets more expensive)"

        if getattr(obs, "cannot_wait", False):
            trend = "URGENT: inventory needs gold now — you cannot wait; buy at the current live quote with an affordable gold_qty (troy oz)."

        rounds_left = (obs.max_market_rounds - obs.market_round) if obs.max_market_rounds else None
        # Suggest buy quantity that reserves $300 for labor (max labor cost)
        reserve = 300.0
        if obs.gold_price > 0:
            raw_qty = (obs.cash - reserve) / obs.gold_price
            suggested_qty = math.floor(raw_qty * 100) / 100
            suggested_qty = max(suggested_qty, 0.01)
        else:
            suggested_qty = 1.0

        _rl = "unlimited" if rounds_left is None else str(rounds_left)
        phase_hint = (
            f"Price: ${getattr(obs, 'gold_price', 0)}/oz ({getattr(obs, 'gold_price_source', '') or 'n/a'}). "
            f"Price history: {prices}. Trend: {trend}. "
            f"Rounds / waits so far: {getattr(obs, 'market_round', 0)}; cap: {_rl}. "
            f"Gold on hand: {getattr(obs, 'gold_oz', 0)} troy oz (~{getattr(obs, 'gold_grams', 0):.2f} g). "
            f"If buying, suggested qty: {suggested_qty} oz (reserves $300 for labor). "
            f"Respond: 'buy {suggested_qty}' or 'wait'"
        )

    elif obs.phase == "warehouse":
        demand = obs.demand
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
            f"You have {obs.gold_oz}oz gold and ${obs.cash} cash. "
            f"Respond with EXACTLY: {best_product}"
        )

    elif obs.phase == "showroom":
        margin = ""
        if obs.current_offer and obs.cost_basis > 0:
            margin_pct = ((obs.current_offer - obs.cost_basis) / obs.cost_basis) * 100
            margin = f"Margin: {margin_pct:+.1f}%. "

        should_accept = False
        if obs.negotiation_round >= 4:
            should_accept = True
        if obs.current_offer and obs.cost_basis > 0 and obs.current_offer > obs.cost_basis * 1.3:
            should_accept = True

        if should_accept:
            phase_hint = (
                f"Cost: ${obs.cost_basis}. Offer: ${obs.current_offer}. {margin}"
                f"Round {obs.negotiation_round}/5. "
                f"Respond with EXACTLY: I accept"
            )
        else:
            # Vary counter-offers per round
            counter_msgs = [
                "I need a better price for this quality piece",
                "That's too low, this craftsmanship deserves more",
                f"How about ${round(obs.cost_basis * 1.4, 2)}?",
                f"I can't go below ${round(obs.cost_basis * 1.3, 2)}",
            ]
            msg = counter_msgs[min(obs.negotiation_round, len(counter_msgs) - 1)]
            phase_hint = (
                f"Cost: ${obs.cost_basis}. Offer: ${obs.current_offer}. {margin}"
                f"Round {obs.negotiation_round}/5. "
                f"DO NOT ACCEPT. Counter-offer. "
                f"Respond with EXACTLY: {msg}"
            )
    else:
        phase_hint = ""

    return textwrap.dedent(
        f"""
        Step: {step} | Phase: {obs.phase} | Last reward: {last_reward:.2f}
        Cash: ${obs.cash} | Gold: {obs.gold_oz}oz | Rings: {obs.inventory}
        Gold Price: ${obs.gold_price}/oz
        Env Message: {obs.message}

        {phase_hint}

        History: {history_block}
        """
    ).strip()


# ── ACTION PARSING ─────────────────────────────

def get_action_from_text(phase: str, text: str) -> tuple[JewelryAction, str]:
    text = text.strip().replace("`", "").strip(' \t\n\r"\'')

    if phase == "market":
        lower = text.lower()
        if lower.startswith("buy"):
            # Extract quantity from "buy 2.5" or "buy2.5"
            qty_str = lower.replace("buy", "").strip()
            try:
                qty = float(qty_str)
            except ValueError:
                qty = 1.0
            return JewelryAction(market_action="buy", gold_qty=qty), f"buy {qty}"
        elif "wait" in lower:
            return JewelryAction(market_action="wait"), "wait"
        else:
            # Try to parse as a number (assumed buy)
            try:
                qty = float(text)
                return JewelryAction(market_action="buy", gold_qty=qty), f"buy {qty}"
            except ValueError:
                return JewelryAction(market_action="wait"), "wait"

    elif phase == "warehouse":
        lower = text.lower()
        for product in ["necklace", "bracelet", "ring"]:
            if product in lower:
                return JewelryAction(product_choice=product), product
        return JewelryAction(product_choice="ring"), "ring"

    elif phase == "showroom":
        return JewelryAction(message=text), text

    return JewelryAction(), text


def get_model_action(client: OpenAI, step: int, obs, last_reward: float, history: List[str]) -> tuple[JewelryAction, str]:
    user_prompt = build_user_prompt(step, obs, last_reward, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return get_action_from_text(obs.phase, text)
    except Exception as exc:
        # print(f"[DEBUG] Model request failed: {exc}", flush=True)
        # Fallback actions
        if obs.phase == "market":
            return JewelryAction(market_action="buy", gold_qty=1.0), "buy 1.0"
        elif obs.phase == "warehouse":
            return JewelryAction(product_choice="ring"), "ring"
        else:
            return JewelryAction(message="I accept"), "I accept"



# ── SINGLE EPISODE RUNNER ──────────────────────

async def run_episode(client: OpenAI, task_name: str, env_name: str, base_url: str) -> float:
    """Run a single episode and return the final score."""
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=env_name, model=MODEL_NAME)

    try:
        env = JewelryShopEnv(base_url=base_url)

        # Pass task_id so the env applies that task's per-phase weights.
        result = await env.reset(task_id=task_name)
        obs = result.observation
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action, raw_action_str = get_model_action(client, step, obs, last_reward, history)
            current_phase = obs.phase

            result = await env.step(action)
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(step=step, action=raw_action_str.replace('\n', ' '), reward=reward, done=done, error=error)
            history.append(f"Step {step} ({current_phase}): {raw_action_str!r} -> reward {reward:+.2f}")

            if done:
                break

        # Trajectory return = env's authoritative cumulative reward (sum of per-step
        # partials, in [0, 1]). Falls back to summing locally if the field is missing.
        score = float(getattr(obs, "cumulative_reward", sum(rewards) if rewards else 0.0))
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            pass
            # print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ── MAIN ───────────────────────────────────────

TASKS = [
    {"id": "market_timing", "env": "jewelry_shop_benchmark"},
    {"id": "demand_crafter", "env": "jewelry_shop_benchmark"},
    {"id": "profit_negotiator", "env": "jewelry_shop_benchmark"},
]

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # ── ENV SERVER URL ──────────────────────────────────────────────────────
    # LOCAL:  start server with `uv run --project . server`, then use localhost
    # REMOTE: comment the localhost line and uncomment the HF Space line
    # base_url = "http://localhost:8000"
    base_url = "https://hard007ik-shopmanagereng.hf.space"
    # ───────────────────────────────────────────────────────────────────────

    # print(f"[CONFIG] base_url={base_url}  model={MODEL_NAME}", flush=True)

    for task in TASKS:
        await run_episode(client, task["id"], task["env"], base_url)


if __name__ == "__main__":
    asyncio.run(main())

