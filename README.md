---
title: Shopmanagereng Environment Server
emoji: 🎖️
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Jewelry Shop Manager — RL Environment

A reinforcement learning environment simulating a **jewelry shop management** pipeline. An AI agent navigates three sequential phases — buying raw materials, selecting products to craft based on demand, and negotiating sales — to maximize profit.

## Environment Overview

### Phase 1: Market (Buy / Wait)

- Gold prices **fluctuate ±10% each round** (up to 3 rounds).
- The agent analyzes price trends and decides to **buy** gold or **wait** for a better price.
- Goal: Buy gold at the lowest possible price while reserving cash for crafting labor.

### Phase 2: Warehouse (Product Selection)

- The agent sees **demand levels** for each product type:

| Product   | Gold (oz) | Labor ($) | Demand Range |
|-----------|-----------|-----------|--------------|
| Ring      | 1.0       | $200      | 40-100%      |
| Necklace  | 2.0       | $300      | 20-80%       |
| Bracelet  | 0.5       | $100      | 10-60%       |

- The agent picks the **highest-demand product** it can afford to craft.
- Goal: Match production to market demand.

### Phase 3: Showroom (Negotiation)

- A customer makes an initial offer based on cost basis and product demand.
- The agent can **accept**, **counter-offer**, or **reject**.
- Each counter raises the customer's offer by **5%** (up to 5 rounds).
- Goal: Sell at maximum profit through smart negotiation.

### Reward Structure

| Component | Weight | Description |
|-----------|--------|-------------|
| R1 (Market) | 20% | How close to the lowest price did the agent buy? |
| R2 (Warehouse) | 20% | Did the agent pick the highest-demand product? |
| R3 (Showroom) | 60% | Normalized profit margin on the sale |

**Final Score** = `0.2 × R1 + 0.2 × R2 + 0.6 × R3` (range [0, 1])

## Quick Start

```python
from ShopManagerEng import JewelryAction, JewelryShopEnv

async def run():
    env = JewelryShopEnv(base_url="http://localhost:8000")

    result = await env.reset()
    print(f"Gold price: ${result.observation.gold_price}/oz")

    # Phase 1 — Market: wait for better price
    result = await env.step(JewelryAction(market_action="wait"))

    # Phase 1 — Market: buy gold
    result = await env.step(JewelryAction(market_action="buy", gold_qty=2.0))

    # Phase 2 — Warehouse: choose product
    result = await env.step(JewelryAction(product_choice="ring"))

    # Phase 3 — Showroom: negotiate
    result = await env.step(JewelryAction(message="How about $600?"))
    result = await env.step(JewelryAction(message="I accept"))

    print(f"Final reward: {result.reward}, Cash: {result.observation.cash}")
    await env.close()

import asyncio
asyncio.run(run())
```

## Action Space

```python
class JewelryAction:
    market_action:  str   # "buy" or "wait" (Phase 1)
    gold_qty:       float # Ounces to buy (Phase 1)
    product_choice: str   # "ring", "necklace", or "bracelet" (Phase 2)
    message:        str   # Negotiation text (Phase 3)
```

## Observation Space

```python
class JewelryObservation:
    phase:              str          # "market" | "warehouse" | "showroom"
    cash:               float        # Current cash balance
    gold_oz:            float        # Raw gold in inventory
    gold_price:         float        # Current gold price ($/oz)
    gold_price_history: List[float]  # Price trend for analysis
    market_round:       int          # Current market round
    demand:             Dict[str, float]  # Demand per product (0-1)
    product_catalog:    Dict[str, dict]   # Specs per product
    inventory:          Dict[str, int]    # Crafted products in stock
    product_for_sale:   str          # Product being sold (showroom)
    cost_basis:         float        # Total manufacturing cost
    current_offer:      float        # Customer's current offer
    negotiation_round:  int          # Counter-offer round
    message:            str          # Environment feedback
```

## Running the Inference Script

```bash
# Terminal 1: Start the server
cd ShopManagerEng
uv run server

# Terminal 2: Run inference (from parent directory or inside ShopManagerEng)
python inference.py
```

Required environment variables (set in `.env`):
- `HF_TOKEN` — Hugging Face API token
- `MODEL_NAME` — LLM model (default: `meta-llama/Llama-3.3-70B-Instruct`)

## Deploying to Hugging Face Spaces

```bash
openenv push
```

## Project Structure

```
ShopManagerEng/
├── __init__.py            # Module exports
├── README.md              # This file
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Dependencies
├── models.py              # Action, Observation, State definitions
├── client.py              # JewelryShopEnv client
├── inference.py           # LLM-based agent inference script
└── server/
    ├── __init__.py
    ├── ShopManagerEng_environment.py  # Core environment logic
    ├── app.py             # FastAPI application
    └── Dockerfile         # Container image
```
