# 🏬 ShopManagerEng: Training LLMs to Run a Business

Welcome to **ShopManagerEng**, a Reinforcement Learning (RL) environment designed to test and train Large Language Models (LLMs) on complex, multi-step business operations. 

While LLMs have become incredibly adept at chatting and writing code, evaluating their ability to make strategic, long-term decisions in a dynamic environment remains a challenge. ShopManagerEng tackles this by putting the AI in the shoes of a jewelry shop manager. Dont get into name 'Eng',just typos mistake of 'Env'.

This blog post breaks down how the environment works, its architecture, and how it can be used to train smarter, more strategic agents.

---

## 🎮 The Environment: A 3-Phase Business Simulation

ShopManagerEng is not a simple Q&A benchmark. It is a continuous simulation where the agent must manage inventory, respond to market fluctuations, and negotiate with customers. Every "episode" (a full game loop) consists of three distinct phases:

### Phase 1: 📈 The Market (Supply)
The agent starts with a limited budget and must decide when to buy raw gold. 
*   **The Catch:** Gold prices fluctuate. The agent can observe price trends and decide to "wait" for a better price or "buy" immediately. 
*   **Real-World Noise:** The environment supports a "real" market mode that pulls live gold prices from Yahoo Finance, testing the agent against real-world economic volatility.

### Phase 2: 🏭 The Warehouse (Production)
Once gold is acquired, the agent must decide what to craft: Rings, Necklaces, or Bracelets.
*   **The Catch:** The agent must balance raw material costs, labor costs, and market demand. It receives a "demand forecast" (which includes noise) and must make production choices that maximize potential profit without running out of cash or gold.

### Phase 3: 🤝 The Showroom (Sales)
The final phase tests the agent's negotiation skills.
*   **The Catch:** Customers make initial offers. The agent can accept, reject, or counter-offer over a maximum of 5 rounds. Push too hard, and the customer walks away (resulting in zero profit). Settle too early, and the agent leaves money on the table.

The agent's final score is a cumulative reward based on how well it navigated all three phases.

---

## 🏗️ Architecture Under the Hood

ShopManagerEng is built on a robust Client-Server architecture, making it highly scalable for Reinforcement Learning tasks.

1.  **The Server (`/server/`):** The core simulation engine runs as a standalone web application (often deployed via Docker to a Hugging Face Space). It tracks the hidden "true" state of the world, handles state transitions, fetches real market data, and manages an SQLite database to log inventory and invoices.
2.  **The Client (`client.py` & `models.py`):** Provides a clean, typed Python interface. Agents use this client to send actions (e.g., `{"market_action": "buy", "gold_qty": 2.0}`) and receive structured observations in return.
3.  **The AI Player (`inference.py`):** A script demonstrating how to plug an LLM (like Llama 3) into the environment. It dynamically builds text prompts explaining the current state (e.g., *"Gold is $2000/oz and trending down. You have $1000. Buy or wait?"*) and parses the LLM's text output back into valid game actions.
4.  **The Training Suite (`/training/`):** The real power of this project lies in its RL training capabilities. Using algorithms like GRPO (Group Relative Policy Optimization), the framework can run massive batches of episodes, evaluate the agent using custom reward functions, and iteratively update the model's weights to forge a master negotiator and supply-chain manager.

### 🛠️ Technical Stack & Data Flow

*   **FastAPI & OpenEnv:** The server is powered by FastAPI, leveraging the `openenv` framework to standardize interactions. This ensures the environment can be easily hosted (e.g., on Hugging Face Spaces) and queried by agents remotely.
*   **Pydantic Models:** All actions and observations are strongly typed using Pydantic (`models.py`). This guarantees that agents send properly formatted JSON payloads (like `target_price_usd` or `inventory_urgent` flags) and receive structured state data.
*   **State Persistence:** A built-in SQLite store (`sqlite_store.py`) tracks the complex state across episodes, maintaining ledgers for cash, gold (in troy ounces and grams), and specific inventory items (rings, necklaces, bracelets).
*   **yfinance Integration:** For the "real" market mode, the environment dynamically fetches live GC=F (Gold Futures) ticker data via `market_data.py`, injecting true market volatility into the simulation rather than relying purely on synthetic random walks.
*   **GRPO Training:** The training pipeline (`train_jewelry_grpo.py`) utilizes Group Relative Policy Optimization, an advanced RL technique designed to stabilize training and improve sample efficiency when teaching LLMs complex, multi-step tasks. Custom reward functions (`rewards.py`) evaluate and guide the model's behavior.

---

## 🚀 Why This Matters

As we move towards autonomous AI agents, we need benchmarks that go beyond static knowledge retrieval. ShopManagerEng evaluates an agent's ability to:
*   **Plan Long-Term:** A bad purchase in Phase 1 ruins the negotiation in Phase 3.
*   **Handle Uncertainty:** Dealing with noisy demand forecasts and volatile live markets.
*   **Negotiate:** Understanding margins and customer psychology.

Whether you are testing the out-of-the-box reasoning of a new foundational model or fine-tuning a specialized RL agent, ShopManagerEng provides a rich, complex sandbox to push AI capabilities forward.
