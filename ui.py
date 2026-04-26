"""
Streamlit UI for ShopManagerEng — Interactive Jewelry Shop Demo.

An AI heuristic agent automatically plays through each episode.
Users press "New Episode" and watch the agent navigate all 3 phases.
"""

import os
import sys
import random
import time
from pathlib import Path

import streamlit as st

# ── Ensure imports resolve ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("SHOPMANAGER_MARKET_MODE", "synthetic")

from server.ShopManagerEng_environment import JewelryShopEnvironment
from models import JewelryAction, PRODUCT_CATALOG


# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ShopManagerEng — Jewelry Shop RL",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS: clean light-theme styling ──────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }

.hero-title {
    font-size: 36px; font-weight: 800;
    background: linear-gradient(135deg, #7c3aed, #3b82f6, #10b981);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 2px;
}
.hero-sub { font-size: 15px; color: #64748b; margin-bottom: 20px; }

.metric-card {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 14px 18px; text-align: center; min-height: 90px;
}
.metric-card .label { font-size: 12px; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
.metric-card .value { font-size: 24px; font-weight: 800; color: #1e293b; margin-top: 4px; }

.phase-box {
    background: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 12px;
    padding: 18px; margin-bottom: 12px;
}
.phase-box.active { border: 2px solid #7c3aed; background: #f5f3ff; }
.phase-box h4 { margin: 0 0 6px; color: #1e293b; }
.phase-box p { margin: 0; color: #475569; font-size: 14px; }

.env-msg {
    background: #eff6ff; border-left: 4px solid #3b82f6;
    border-radius: 0 10px 10px 0; padding: 12px 16px;
    margin: 10px 0; font-size: 13px; color: #1e40af; line-height: 1.5;
    word-wrap: break-word;
}

.step-row {
    padding: 8px 12px; margin: 4px 0; border-radius: 8px;
    font-size: 13px; color: #334155;
}
.step-row.market   { background: #fef3c7; border-left: 3px solid #f59e0b; }
.step-row.warehouse { background: #dbeafe; border-left: 3px solid #3b82f6; }
.step-row.showroom  { background: #d1fae5; border-left: 3px solid #10b981; }

.reward-big {
    text-align: center; padding: 24px;
    background: linear-gradient(135deg, #f5f3ff, #eff6ff);
    border: 2px solid #7c3aed; border-radius: 16px;
}
.reward-big .score { font-size: 52px; font-weight: 800; color: #7c3aed; }
.reward-big .label { font-size: 14px; color: #64748b; }

.catalog-card {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 16px; text-align: center;
}
.catalog-card h4 { margin: 0 0 8px; color: #1e293b; }
.catalog-card p { margin: 2px 0; color: #475569; font-size: 13px; }

.demand-bar {
    background: #e2e8f0; border-radius: 6px; height: 10px; overflow: hidden; margin-top: 4px;
}
.demand-fill { height: 100%; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)


# ── Heuristic Agent ─────────────────────────────────────────────────────────
def heuristic_action(obs):
    """Simple rule-based agent that plays through all 3 phases."""
    if obs.phase == "market":
        price = float(obs.gold_price or 300)
        # Buy if we have enough cash for at least 1 oz
        if obs.cash >= price + 10:
            return JewelryAction(
                market_action="buy",
                gold_qty=1.0,
                target_price_usd=obs.gold_price,
            )
        return JewelryAction(market_action="wait")

    if obs.phase == "warehouse":
        # Pick the highest-demand product we can afford
        demand = obs.demand or {"ring": 0.5, "necklace": 0.3, "bracelet": 0.2}
        for name in sorted(demand, key=lambda k: demand.get(k, 0), reverse=True):
            spec = PRODUCT_CATALOG[name]
            if obs.gold_oz + 1e-8 >= spec["gold_oz"] and obs.cash >= spec["labor"]:
                return JewelryAction(product_choice=name)
        return JewelryAction(product_choice="ring")

    if obs.phase == "showroom":
        # Accept if margin > 15% or after round 3
        if (
            obs.current_offer
            and obs.cost_basis > 0
            and float(obs.current_offer) / float(obs.cost_basis) >= 1.15
        ) or (obs.negotiation_round and int(obs.negotiation_round) >= 3):
            return JewelryAction(message="I accept")
        offer = float(obs.current_offer or 0)
        if offer:
            return JewelryAction(message=f"How about ${offer * 1.08:.2f}?")
        return JewelryAction(message="I need a better offer")

    return JewelryAction()


# ── Session state ───────────────────────────────────────────────────────────
if "episode_steps" not in st.session_state:
    st.session_state.episode_steps = None
    st.session_state.final_reward = None
    st.session_state.episode_count = 0


def run_episode(task_id):
    """Run a full episode with the heuristic agent and return step logs."""
    env = JewelryShopEnvironment()
    seed = random.randint(0, 99999)
    obs = env.reset(seed=seed, market_mode="synthetic", task_id=task_id)

    steps = [{
        "step": 0, "phase": obs.phase, "action": "reset",
        "msg": obs.message, "reward": 0.0,
        "cash": obs.cash, "gold_oz": obs.gold_oz,
        "gold_price": obs.gold_price,
        "cumulative": float(obs.cumulative_reward),
    }]

    for i in range(1, 20):
        if obs.done:
            break
        action = heuristic_action(obs)

        # Describe the action in human terms
        if obs.phase == "market":
            act_str = f"BUY {action.gold_qty} oz" if action.market_action == "buy" else "WAIT"
        elif obs.phase == "warehouse":
            act_str = f"CRAFT {action.product_choice}"
        else:
            act_str = action.message or "..."

        obs = env.step(action)
        steps.append({
            "step": i, "phase": obs.phase, "action": act_str,
            "msg": obs.message, "reward": float(obs.reward),
            "cash": obs.cash, "gold_oz": obs.gold_oz,
            "gold_price": getattr(obs, "gold_price", 0),
            "product": getattr(obs, "product_for_sale", None),
            "offer": float(obs.current_offer) if obs.current_offer else None,
            "cost_basis": float(obs.cost_basis) if obs.cost_basis else None,
            "cumulative": float(obs.cumulative_reward),
        })

    return steps, float(obs.cumulative_reward)


# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 💎 ShopManagerEng")
    st.markdown("---")

    task = st.selectbox(
        "🎯 Task Profile",
        ["profit_negotiator", "market_timing", "demand_crafter"],
        index=0,
    )
    weights = {
        "profit_negotiator": "Showroom 60% · Market 20% · Warehouse 20%",
        "market_timing": "Market 60% · Warehouse 20% · Showroom 20%",
        "demand_crafter": "Warehouse 60% · Market 20% · Showroom 20%",
    }
    st.caption(f"**Weights:** {weights[task]}")

    st.markdown("---")
    if st.button("🚀 New Episode", use_container_width=True, type="primary"):
        steps, reward = run_episode(task)
        st.session_state.episode_steps = steps
        st.session_state.final_reward = reward
        st.session_state.episode_count += 1
        st.rerun()

    st.markdown("---")
    st.markdown("#### How It Works")
    st.markdown("""
    An AI **heuristic agent** automatically plays
    through all 3 phases of the jewelry shop:

    1. 📈 **Market** — Buy gold at the right price
    2. 🏭 **Warehouse** — Craft the most demanded product
    3. 🤝 **Showroom** — Negotiate the best sale price

    Press **🚀 New Episode** to watch the agent play!
    """)

    if st.session_state.final_reward is not None:
        st.markdown("---")
        reward = st.session_state.final_reward
        color = "#10b981" if reward >= 0.6 else "#f59e0b" if reward >= 0.4 else "#ef4444"
        st.markdown(f"**Final Score:** :{'green' if reward >= 0.6 else 'orange' if reward >= 0.4 else 'red'}[{reward:.4f}]")
        st.metric("Episodes Played", st.session_state.episode_count)


# ── Main area ───────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">💎 Jewelry Shop Manager</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">An RL environment for training LLMs on multi-step business decisions</p>', unsafe_allow_html=True)


# ── No episode yet — show welcome ──────────────────────────────────────────
if st.session_state.episode_steps is None:
    st.info("👋 **Welcome!** Press **🚀 New Episode** in the sidebar to watch the AI agent play through the jewelry shop simulation.")

    st.markdown("### 📦 Product Catalog")
    cols = st.columns(3)
    items = [("💍 Ring", "ring"), ("📿 Necklace", "necklace"), ("⌚ Bracelet", "bracelet")]
    for i, (icon_name, key) in enumerate(items):
        spec = PRODUCT_CATALOG[key]
        with cols[i]:
            st.markdown(f"""
            <div class="catalog-card">
                <h4>{icon_name}</h4>
                <p>🪙 Gold: {spec['gold_oz']} oz</p>
                <p>🔧 Labor: ${spec['labor']:.0f}</p>
                <p>📊 Base demand: {spec['base_demand']:.0%}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("### 🧠 Three Business Phases")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="phase-box">
            <h4>📈 Phase 1: Market</h4>
            <p>Buy raw gold at the best price. Prices fluctuate — time your purchase wisely!</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="phase-box">
            <h4>🏭 Phase 2: Warehouse</h4>
            <p>Craft a product (ring, necklace, bracelet) that matches market demand.</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="phase-box">
            <h4>🤝 Phase 3: Showroom</h4>
            <p>Negotiate with a customer over 5 rounds to maximize your selling price.</p>
        </div>
        """, unsafe_allow_html=True)


# ── Episode results ─────────────────────────────────────────────────────────
else:
    steps = st.session_state.episode_steps
    reward = st.session_state.final_reward

    # ── Score banner ────────────────────────────────────────────────────────
    if reward >= 0.8:
        grade, grade_emoji = "Excellent", "🏆"
    elif reward >= 0.6:
        grade, grade_emoji = "Good", "👍"
    elif reward >= 0.4:
        grade, grade_emoji = "Fair", "😐"
    else:
        grade, grade_emoji = "Poor", "😬"

    st.markdown(f"""
    <div class="reward-big">
        <div class="label">{grade_emoji} Episode #{st.session_state.episode_count} — {grade}</div>
        <div class="score">{reward:.4f}</div>
        <div class="label">cumulative reward (out of 1.0)</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # ── Summary metrics ─────────────────────────────────────────────────────
    last = steps[-1]
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""<div class="metric-card"><div class="label">Steps</div><div class="value">{len(steps)-1}</div></div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="metric-card"><div class="label">Final Cash</div><div class="value">${last['cash']:,.0f}</div></div>""", unsafe_allow_html=True)
    with m3:
        gold_price = steps[0].get("gold_price", 0)
        st.markdown(f"""<div class="metric-card"><div class="label">Gold Price</div><div class="value">${gold_price:,.0f}/oz</div></div>""", unsafe_allow_html=True)
    with m4:
        product = None
        for s in steps:
            if s.get("product"):
                product = s["product"]
        st.markdown(f"""<div class="metric-card"><div class="label">Product</div><div class="value">{(product or 'N/A').title()}</div></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Step-by-step log ────────────────────────────────────────────────────
    st.markdown("### 📋 Agent Decision Log")

    for s in steps:
        phase = s.get("phase", "market")
        icon = {"market": "📈", "warehouse": "🏭", "showroom": "🤝"}.get(phase, "⬜")
        step_num = s["step"]
        action = s.get("action", "")
        rw = s.get("reward", 0)
        cum = s.get("cumulative", 0)

        rw_badge = f" · reward: `{rw:.4f}`" if rw else ""
        cum_str = f" · cumulative: `{cum:.4f}`" if cum else ""

        # Use pure markdown — no raw HTML divs
        st.markdown(f"**Step {step_num}** {icon} **{action}**{rw_badge}{cum_str}")

        # Show environment message
        if s.get("msg"):
            st.caption(s["msg"])

        st.divider()

    # ── Reward progression chart ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Cumulative Reward Over Steps")
    chart_data = [s["cumulative"] for s in steps]
    st.line_chart(chart_data, use_container_width=True, height=200)

    st.info("Press **🚀 New Episode** in the sidebar to run another episode with a different seed!")
