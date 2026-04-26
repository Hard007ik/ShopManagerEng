"""
Live gold (USD / troy oz) via yfinance, aligned with api_key_test.py (GC=F).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class GoldPriceQuote:
    usd_per_oz: float
    source: str


def os_gold_symbol() -> str:
    import os

    return (os.environ.get("SHOPMANAGER_GOLD_SYMBOL", "GC=F") or "GC=F").strip()


def _fetch_yfinance_gold() -> Tuple[float, str, List[float]]:
    import yfinance as yf

    sym = (os_gold_symbol() or "GC=F").strip() or "GC=F"
    ticker = yf.Ticker(sym)
    hist = ticker.history(period="60d", interval="1d")
    if hist is None or hist.empty:
        raise ValueError("No price history for gold symbol")
    closes = hist["Close"].dropna().astype(float).tolist()
    if not closes:
        raise ValueError("Empty close series for gold")
    return float(closes[-1]), f"yfinance:{sym}", [float(c) for c in closes[-30:]]


def fetch_gold_spot_usd_per_oz() -> GoldPriceQuote:
    usd, src, _ = _fetch_yfinance_gold()
    if usd <= 0:
        raise ValueError("Invalid non-positive gold price")
    return GoldPriceQuote(usd_per_oz=usd, source=src)


def recent_close_history(max_points: int = 30) -> List[float]:
    try:
        _, _, hist = _fetch_yfinance_gold()
    except Exception:
        return []
    if max_points and len(hist) > max_points:
        return hist[-max_points:]
    return list(hist)


def last_quote_or_fallback(fallback: float) -> GoldPriceQuote:
    try:
        return fetch_gold_spot_usd_per_oz()
    except Exception:
        return GoldPriceQuote(usd_per_oz=fallback, source="fallback")
