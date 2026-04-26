"""
Shared physical and market constants (troy oz ↔ grams, config keys).
"""
import os
from typing import Final

# Troy ounce (XAU convention in this project) to grams, per user / pricing spec
GRAMS_PER_TROY_OZ: Final[float] = 31.1035


def troy_oz_to_grams(oz: float) -> float:
    return round(oz * GRAMS_PER_TROY_OZ, 6)


def grams_to_troy_oz(grams: float) -> float:
    if GRAMS_PER_TROY_OZ <= 0:
        return 0.0
    return round(grams / GRAMS_PER_TROY_OZ, 8)


def get_market_mode() -> str:
    """'real' uses live GC=F + DB; 'synthetic' uses legacy random market (for offline tests)."""
    return (os.environ.get("SHOPMANAGER_MARKET_MODE", "real") or "real").lower().strip()


def get_sqlite_path() -> str:
    return os.environ.get("SHOPMANAGER_SQLITE_PATH", "").strip() or ""


def default_sqlite_path() -> str:
    from pathlib import Path

    here = Path(__file__).resolve().parent
    return str(here / "data" / "shop_manager.db")
