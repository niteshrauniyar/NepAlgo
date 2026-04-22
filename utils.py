"""
Shared utilities for NEPSE Trading Intelligence App
"""

import os
import json
import logging
import pickle
import hashlib
from pathlib import Path
from datetime import datetime, date
from typing import Any, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ── Cache configuration ────────────────────────────────────────────────────────
CACHE_DIR = Path("cache")
CACHE_FILE = CACHE_DIR / "last_market_data.pkl"
CACHE_META_FILE = CACHE_DIR / "cache_meta.json"
CACHE_MAX_AGE_HOURS = 24


def ensure_cache_dir():
    CACHE_DIR.mkdir(exist_ok=True)


# ── Persistence helpers ────────────────────────────────────────────────────────

def save_cache(df: pd.DataFrame, source: str):
    """Persist a DataFrame and metadata to disk."""
    try:
        ensure_cache_dir()
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(df, f)
        meta = {
            "source": source,
            "timestamp": datetime.now().isoformat(),
            "rows": len(df),
            "columns": list(df.columns),
        }
        with open(CACHE_META_FILE, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Cache saved: {len(df)} rows from {source}.")
    except Exception as e:
        logger.warning(f"Cache save failed: {e}")


def load_cache() -> tuple[Optional[pd.DataFrame], Optional[dict]]:
    """Load cached DataFrame and metadata. Returns (df, meta) or (None, None)."""
    try:
        if not CACHE_FILE.exists():
            return None, None

        with open(CACHE_FILE, "rb") as f:
            df = pickle.load(f)

        meta = {}
        if CACHE_META_FILE.exists():
            with open(CACHE_META_FILE) as f:
                meta = json.load(f)

        # Validate age
        if "timestamp" in meta:
            cached_at = datetime.fromisoformat(meta["timestamp"])
            age_hours = (datetime.now() - cached_at).total_seconds() / 3600
            meta["age_hours"] = round(age_hours, 1)
            if age_hours > CACHE_MAX_AGE_HOURS:
                logger.warning(f"Cache is {age_hours:.1f}h old — stale but usable as fallback.")

        logger.info(f"Cache loaded: {len(df)} rows.")
        return df, meta

    except Exception as e:
        logger.warning(f"Cache load failed: {e}")
        return None, None


# ── Safe numeric conversion ────────────────────────────────────────────────────

def safe_to_numeric(series: pd.Series) -> pd.Series:
    """
    Robustly convert a Series to numeric.
    Handles: commas, parentheses (negatives), percentage signs, em-dashes.
    """
    try:
        cleaned = (
            series.astype(str)
            .str.strip()
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.replace("\u2014", "0", regex=False)   # em dash
            .str.replace("\u2013", "0", regex=False)   # en dash
            .str.replace("−", "-", regex=False)        # unicode minus
        )
        # Handle parentheses as negatives: (123.4) → -123.4
        mask = cleaned.str.startswith("(") & cleaned.str.endswith(")")
        cleaned[mask] = "-" + cleaned[mask].str[1:-1]

        return pd.to_numeric(cleaned, errors="coerce")
    except Exception:
        return pd.to_numeric(series, errors="coerce")


# ── Data validation ────────────────────────────────────────────────────────────

def validate_dataframe(df: pd.DataFrame, required_cols: list[str]) -> bool:
    """Return True only if df is non-empty and has all required columns."""
    if df is None or df.empty:
        return False
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.warning(f"DataFrame missing columns: {missing}")
        return False
    return True


def is_empty_or_none(df) -> bool:
    if df is None:
        return True
    if isinstance(df, pd.DataFrame):
        return df.empty
    return True


# ── JSON serialisation helpers ─────────────────────────────────────────────────

def to_serializable(obj: Any) -> Any:
    """Recursively convert numpy / pandas objects to Python native types."""
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(i) for i in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj) if not np.isnan(obj) else None
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return str(obj)
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


# ── Formatting ─────────────────────────────────────────────────────────────────

def fmt_number(val, decimals=2, suffix="") -> str:
    """Format a number for display. Returns '—' for None/NaN."""
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "—"
        if abs(val) >= 1_000_000_000:
            return f"{val/1_000_000_000:.{decimals}f}B{suffix}"
        if abs(val) >= 1_000_000:
            return f"{val/1_000_000:.{decimals}f}M{suffix}"
        if abs(val) >= 1_000:
            return f"{val/1_000:.{decimals}f}K{suffix}"
        return f"{val:.{decimals}f}{suffix}"
    except Exception:
        return str(val)


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
