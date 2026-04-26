"""
SQLite persistence: market gold purchase invoices (troy oz) and per-gram lots (FIFO for warehouse).
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

try:
    from ..constants import GRAMS_PER_TROY_OZ, default_sqlite_path, get_sqlite_path
except ImportError:
    # `python -c` / `import server` from the ShopManagerEng/ folder: `server` is a top module,
    # so `..constants` is invalid. Parent package constants.py lives as a sibling of `server/`.
    from constants import GRAMS_PER_TROY_OZ, default_sqlite_path, get_sqlite_path


def _db_path() -> str:
    p = get_sqlite_path()
    return p if p else default_sqlite_path()


def _connect() -> sqlite3.Connection:
    path = _db_path()
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False, timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_schema() -> None:
    with _connect() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS gold_purchases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id TEXT NOT NULL,
                product_name TEXT NOT NULL,
                buy_price_usd REAL NOT NULL,
                quantity_oz REAL NOT NULL,
                cost_usd REAL NOT NULL,
                ai_decision TEXT NOT NULL,
                ai_confidence_pct REAL,
                ai_reasoning TEXT,
                target_price_usd REAL,
                bought_at TEXT NOT NULL,
                fund_before_usd REAL NOT NULL,
                fund_after_usd REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_gold_purchases_episode
                ON gold_purchases (episode_id);
            CREATE TABLE IF NOT EXISTS gold_grams_lots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                purchase_id INTEGER NOT NULL,
                episode_id TEXT NOT NULL,
                product_name TEXT NOT NULL,
                buy_price_usd_per_gram REAL NOT NULL,
                quantity_grams_total REAL NOT NULL,
                quantity_grams_remaining REAL NOT NULL,
                bought_at TEXT NOT NULL,
                FOREIGN KEY (purchase_id) REFERENCES gold_purchases (id)
            );
            CREATE INDEX IF NOT EXISTS idx_lots_episode_bought
                ON gold_grams_lots (episode_id, bought_at, id);
            """
        )
        c.commit()


@dataclass
class PurchaseRow:
    id: int
    buy_price_usd: float
    quantity_oz: float
    cost_usd: float
    target_price_usd: Optional[float]
    fund_before_usd: float
    fund_after_usd: float
    bought_at: str


def record_gold_purchase(
    episode_id: str,
    product_name: str,
    buy_price_usd: float,
    quantity_oz: float,
    cost_usd: float,
    ai_decision: str,
    ai_confidence_pct: Optional[float],
    ai_reasoning: Optional[str],
    target_price_usd: Optional[float],
    fund_before_usd: float,
    fund_after_usd: float,
) -> Tuple[int, int]:
    """
    Inserts into gold_purchases and gold_grams_lots. Returns (purchase_id, lot_id).
    """
    init_schema()
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    g_total = round(quantity_oz * GRAMS_PER_TROY_OZ, 6)
    ppg = round(buy_price_usd / GRAMS_PER_TROY_OZ, 8) if GRAMS_PER_TROY_OZ > 0 else 0.0
    ai_r = (ai_reasoning or "").strip() or None
    with _connect() as c:
        cur = c.execute(
            """
            INSERT INTO gold_purchases (
                episode_id, product_name, buy_price_usd, quantity_oz, cost_usd,
                ai_decision, ai_confidence_pct, ai_reasoning, target_price_usd,
                bought_at, fund_before_usd, fund_after_usd
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                episode_id,
                product_name,
                buy_price_usd,
                quantity_oz,
                cost_usd,
                ai_decision,
                ai_confidence_pct,
                ai_r,
                target_price_usd,
                now,
                fund_before_usd,
                fund_after_usd,
            ),
        )
        purchase_id = int(cur.lastrowid)
        cur2 = c.execute(
            """
            INSERT INTO gold_grams_lots (
                purchase_id, episode_id, product_name, buy_price_usd_per_gram,
                quantity_grams_total, quantity_grams_remaining, bought_at
            ) VALUES (?,?,?,?,?,?,?)
            """,
            (
                purchase_id,
                episode_id,
                product_name,
                ppg,
                g_total,
                g_total,
                now,
            ),
        )
        lot_id = int(cur2.lastrowid)
        c.commit()
    return purchase_id, lot_id


def fifo_consume_grams(
    episode_id: str, grams_needed: float
) -> Tuple[bool, float, List[dict]]:
    """
    Uses oldest lots first. Returns (ok, total_usd_cost, details).
    """
    if grams_needed <= 0:
        return True, 0.0, []
    init_schema()
    rem = float(grams_needed)
    total_usd = 0.0
    details: List[dict] = []
    with _connect() as c:
        cur = c.execute(
            """
            SELECT id, quantity_grams_remaining, buy_price_usd_per_gram
            FROM gold_grams_lots
            WHERE episode_id = ? AND quantity_grams_remaining > 0.0000001
            ORDER BY bought_at ASC, id ASC
            """,
            (episode_id,),
        )
        rows = cur.fetchall()
        for row in rows:
            if rem <= 1e-9:
                break
            lot_id = int(row["id"])
            qrem = float(row["quantity_grams_remaining"])
            ppg = float(row["buy_price_usd_per_gram"])
            take = min(qrem, rem)
            cost = take * ppg
            new_rem = round(qrem - take, 6)
            c.execute(
                "UPDATE gold_grams_lots SET quantity_grams_remaining = ? WHERE id = ?",
                (new_rem, lot_id),
            )
            rem -= take
            total_usd += cost
            details.append(
                {
                    "lot_id": lot_id,
                    "grams": take,
                    "cost_usd": round(cost, 4),
                }
            )
        c.commit()
    if rem > 1e-5:
        return False, 0.0, []
    return True, round(total_usd, 4), details


def ensure_schema_once() -> None:
    try:
        init_schema()
    except Exception:
        pass
