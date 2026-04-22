"""
NEPSE Data Engine
─────────────────
Handles: data fetching pipeline, normalization, analytics,
         order flow proxies, smart money scoring.
"""

import logging
import os
import sys
from typing import Optional
import numpy as np
import pandas as pd

# Guarantee project root is importable regardless of CWD (Streamlit Cloud, etc.)
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fetchers import fetch_from_api, fetch_from_sharesansar, fetch_from_nepsealpha
from utils import (
    save_cache, load_cache, safe_to_numeric,
    is_empty_or_none, to_serializable
)

logger = logging.getLogger(__name__)

# ── Column name mapping (all known variants → standard name) ───────────────────
COLUMN_MAP = {
    # Symbol / ticker
    "symbol": "symbol", "ticker": "symbol", "scrip": "symbol",
    "stock": "symbol", "company": "symbol", "name": "symbol",
    "stocksymbol": "symbol",

    # % change
    "pct_change": "pct_change", "percent_change": "pct_change",
    "change_percent": "pct_change", "pctchange": "pct_change",
    "% change": "pct_change", "%change": "pct_change",
    "change%": "pct_change", "percentchange": "pct_change",
    "changepercent": "pct_change", "pointchange": "pct_change",
    "diff%": "pct_change", "diff": "pct_change",

    # LTP / close price
    "ltp": "ltp", "lasttradedprice": "ltp", "last_traded_price": "ltp",
    "closeprice": "ltp", "close_price": "ltp", "close": "ltp",
    "price": "ltp", "lastprice": "ltp", "last_price": "ltp",

    # Open
    "open": "open", "openprice": "open", "open_price": "open",

    # High / Low
    "high": "high", "highprice": "high", "high_price": "high",
    "dayhigh": "high", "52weekhigh": "week52_high",
    "low": "low", "lowprice": "low", "low_price": "low",
    "daylow": "low", "52weeklow": "week52_low",

    # Volume
    "volume": "volume", "qty": "volume", "quantity": "volume",
    "tradedquantity": "volume", "traded_quantity": "volume",
    "totaltraded": "volume", "total_traded_quantity": "volume",
    "sharesqty": "volume",

    # Turnover
    "turnover": "turnover", "totalturnovers": "turnover",
    "total_turnover": "turnover", "traded_value": "turnover",
    "tradedvalue": "turnover", "amount": "turnover", "value": "turnover",

    # Transactions
    "transactions": "transactions", "transaction": "transactions",
    "nooftransactions": "transactions", "no_of_transactions": "transactions",

    # Previous close
    "previousclose": "prev_close", "prev_close": "prev_close",
    "previous_close": "prev_close", "previouscloseprice": "prev_close",
}

NUMERIC_COLS = ["ltp", "open", "high", "low", "pct_change",
                "volume", "turnover", "transactions", "prev_close"]

REQUIRED_COLS = ["symbol"]  # Minimum viability


# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCH PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def get_market_data() -> tuple[Optional[pd.DataFrame], str, str]:
    """
    Attempt to fetch market data from all sources with fallback.
    Returns: (DataFrame | None, source_name, status_message)
    """
    sources = [
        ("NEPSE Official API", fetch_from_api),
        ("ShareSansar",        fetch_from_sharesansar),
        ("NepseAlpha",         fetch_from_nepsealpha),
    ]

    for name, fetcher in sources:
        try:
            logger.info(f"Trying source: {name}")
            raw = fetcher()
            if is_empty_or_none(raw):
                logger.warning(f"{name}: returned empty data.")
                continue

            df = normalize_market_data(raw)
            if is_empty_or_none(df) or "symbol" not in df.columns:
                logger.warning(f"{name}: normalization produced unusable data.")
                continue

            save_cache(df, name)
            logger.info(f"SUCCESS — {name}: {len(df)} stocks loaded.")
            return df, name, "live"

        except Exception as e:
            logger.error(f"{name}: pipeline error — {e}")

    # All live sources failed — try cache
    logger.warning("All live sources failed. Attempting cache fallback.")
    cached_df, meta = load_cache()
    if not is_empty_or_none(cached_df):
        age = meta.get("age_hours", "?") if meta else "?"
        src = meta.get("source", "Cache") if meta else "Cache"
        return cached_df, src, f"cached ({age}h ago)"

    return None, "None", "unavailable"


# ══════════════════════════════════════════════════════════════════════════════
# DATA NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def normalize_market_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robustly normalise raw scraped/API data into a standard schema.
    NEVER crashes — returns whatever it can salvage.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    try:
        df = df.copy()

        # 1. Flatten column names
        df.columns = (
            df.columns.astype(str)
            .str.strip()
            .str.lower()
            .str.replace(r"[\s\-/]+", "_", regex=True)
            .str.replace(r"[^a-z0-9_%]", "", regex=True)
        )

        # 2. Map to standard names
        df.rename(columns={c: COLUMN_MAP[c] for c in df.columns if c in COLUMN_MAP},
                  inplace=True)

        # 3. Deduplicate columns (keep first occurrence)
        df = df.loc[:, ~df.columns.duplicated()]

        # 4. Ensure 'symbol' exists — try to infer from first text column
        if "symbol" not in df.columns:
            text_cols = df.select_dtypes(include="object").columns.tolist()
            if text_cols:
                df.rename(columns={text_cols[0]: "symbol"}, inplace=True)
                logger.warning(f"Inferred 'symbol' from column '{text_cols[0]}'.")

        if "symbol" not in df.columns:
            logger.error("Cannot find a symbol column — data unusable.")
            return pd.DataFrame()

        # 5. Clean symbol column
        df["symbol"] = (
            df["symbol"].astype(str)
            .str.strip()
            .str.upper()
            .replace({"NAN": np.nan, "NONE": np.nan, "": np.nan})
        )
        df.dropna(subset=["symbol"], inplace=True)
        df = df[df["symbol"] != ""]

        # 6. Convert numeric columns safely
        for col in NUMERIC_COLS:
            if col in df.columns:
                df[col] = safe_to_numeric(df[col])

        # 7. Derive pct_change if missing but LTP + prev_close exist
        if "pct_change" not in df.columns or df["pct_change"].isna().all():
            if "ltp" in df.columns and "prev_close" in df.columns:
                df["pct_change"] = (
                    (df["ltp"] - df["prev_close"]) / df["prev_close"].replace(0, np.nan) * 100
                ).round(2)
                logger.info("Derived pct_change from LTP and prev_close.")

        # 8. Ensure pct_change column exists with fill
        if "pct_change" not in df.columns:
            df["pct_change"] = 0.0
        else:
            df["pct_change"] = df["pct_change"].fillna(0.0)

        # 9. Fill other numeric cols with 0
        for col in ["volume", "turnover", "transactions"]:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)

        # 10. Reset index
        df.reset_index(drop=True, inplace=True)

        logger.info(f"Normalization complete: {len(df)} rows, cols: {list(df.columns)}")
        return df

    except Exception as e:
        logger.error(f"normalize_market_data crashed unexpectedly: {e}")
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICS ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def market_summary(df: pd.DataFrame) -> dict:
    """
    Compute market-wide summary statistics.
    Returns plain Python dict (no pandas objects).
    """
    result = {
        "total_stocks": 0, "advances": 0, "declines": 0, "unchanged": 0,
        "total_volume": 0, "total_turnover": 0,
        "avg_pct_change": 0.0, "breadth": 0.0,
        "top_gainer": {}, "top_loser": {}, "most_active": {},
        "market_sentiment": "neutral",
    }

    try:
        if is_empty_or_none(df):
            return result

        result["total_stocks"] = int(len(df))

        if "pct_change" in df.columns:
            pct = df["pct_change"].fillna(0)
            result["advances"]  = int((pct > 0).sum())
            result["declines"]  = int((pct < 0).sum())
            result["unchanged"] = int((pct == 0).sum())
            result["avg_pct_change"] = round(float(pct.mean()), 3)
            total = result["advances"] + result["declines"]
            result["breadth"] = (
                round((result["advances"] - result["declines"]) / total, 4) if total else 0.0
            )

        if "volume" in df.columns:
            result["total_volume"] = int(df["volume"].fillna(0).sum())

        if "turnover" in df.columns:
            result["total_turnover"] = float(round(df["turnover"].fillna(0).sum(), 2))

        # Top gainer
        if "pct_change" in df.columns and "symbol" in df.columns:
            idx = df["pct_change"].idxmax()
            row = df.loc[idx]
            result["top_gainer"] = {
                "symbol": str(row.get("symbol", "—")),
                "pct_change": float(row.get("pct_change", 0)),
                "ltp": float(row.get("ltp", 0)) if "ltp" in df.columns else None,
            }

        # Top loser
        if "pct_change" in df.columns:
            idx = df["pct_change"].idxmin()
            row = df.loc[idx]
            result["top_loser"] = {
                "symbol": str(row.get("symbol", "—")),
                "pct_change": float(row.get("pct_change", 0)),
                "ltp": float(row.get("ltp", 0)) if "ltp" in df.columns else None,
            }

        # Most active by volume
        if "volume" in df.columns:
            idx = df["volume"].idxmax()
            row = df.loc[idx]
            result["most_active"] = {
                "symbol": str(row.get("symbol", "—")),
                "volume": float(row.get("volume", 0)),
                "pct_change": float(row.get("pct_change", 0)) if "pct_change" in df.columns else None,
            }

        # Market sentiment
        b = result["breadth"]
        if b > 0.3:
            result["market_sentiment"] = "bullish"
        elif b < -0.3:
            result["market_sentiment"] = "bearish"
        else:
            result["market_sentiment"] = "neutral"

    except Exception as e:
        logger.error(f"market_summary error: {e}")

    return to_serializable(result)


# ── Order Flow Proxy ───────────────────────────────────────────────────────────

def order_flow_signal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Approximate order flow using price+volume behaviour.
    Adds: buy_pressure, sell_pressure, persistence_score
    """
    if is_empty_or_none(df):
        return df

    try:
        out = df.copy()
        pct = out.get("pct_change", pd.Series(dtype=float)).fillna(0)
        vol = out.get("volume",     pd.Series(dtype=float)).fillna(0)

        # Normalise volume 0-1 using robust scaler
        vol_range = vol.max() - vol.min()
        vol_norm = (vol - vol.min()) / vol_range if vol_range > 0 else pd.Series(0, index=vol.index)

        # Buy / sell pressure
        pos_mask = pct > 0
        neg_mask = pct < 0
        out["buy_pressure"]  = (vol_norm * pos_mask.astype(float) * pct.clip(lower=0)).round(4)
        out["sell_pressure"] = (vol_norm * neg_mask.astype(float) * pct.clip(upper=0).abs()).round(4)

        # Persistence: magnitude of pct_change weighted by volume rank
        vol_rank = vol.rank(pct=True)
        out["persistence_score"] = (pct.abs() * vol_rank).round(4)

    except Exception as e:
        logger.error(f"order_flow_signal error: {e}")

    return out


# ── Large Activity Detection ───────────────────────────────────────────────────

def detect_large_activity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag stocks with abnormally high volume vs the market median.
    Adds: volume_ratio, large_activity_flag
    """
    if is_empty_or_none(df) or "volume" not in df.columns:
        return df

    try:
        out = df.copy()
        vol = out["volume"].fillna(0)
        median_vol = vol.median()
        std_vol    = vol.std()

        # z-score based spike detection
        out["volume_ratio"] = (vol / median_vol.clip(min=1)).round(3)
        threshold = median_vol + 2 * std_vol
        out["large_activity_flag"] = vol > threshold

    except Exception as e:
        logger.error(f"detect_large_activity error: {e}")

    return out


# ── Liquidity Metrics ─────────────────────────────────────────────────────────

def liquidity_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute volatility and price impact proxies.
    Adds: volatility_rank, price_impact, turnover_spike_flag
    """
    if is_empty_or_none(df):
        return df

    try:
        out = df.copy()

        if "pct_change" in out.columns:
            pct = out["pct_change"].fillna(0).abs()
            # Relative volatility rank (percentile)
            out["volatility_rank"] = pct.rank(pct=True).round(4)

        if "pct_change" in out.columns and "volume" in out.columns:
            vol = out["volume"].fillna(0).replace(0, np.nan)
            out["price_impact"] = (out["pct_change"].abs() / vol).round(8)

        if "turnover" in out.columns:
            to = out["turnover"].fillna(0)
            to_mean = to.mean()
            to_std  = to.std()
            out["turnover_spike_flag"] = to > (to_mean + 2 * to_std)

    except Exception as e:
        logger.error(f"liquidity_metrics error: {e}")

    return out


# ── Smart Money Score ─────────────────────────────────────────────────────────

def smart_money_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Composite smart money / institutional activity proxy.
    Score 0–100 derived from:
      - price momentum (pct_change percentile)
      - volume spike magnitude
      - persistence score
      - liquidity/price impact (inverted)
    Adds: smart_money_score, smart_money_label
    """
    if is_empty_or_none(df):
        return df

    try:
        out = df.copy()

        def _percentile_rank(s: pd.Series) -> pd.Series:
            s = pd.to_numeric(s, errors="coerce").fillna(0)
            rng = s.max() - s.min()
            return ((s - s.min()) / rng * 100) if rng > 0 else pd.Series(50, index=s.index)

        components = []
        weights = []

        # 1. Price momentum
        if "pct_change" in out.columns:
            components.append(_percentile_rank(out["pct_change"]))
            weights.append(0.30)

        # 2. Volume spike
        if "volume_ratio" in out.columns:
            components.append(_percentile_rank(out["volume_ratio"]))
            weights.append(0.30)
        elif "volume" in out.columns:
            components.append(_percentile_rank(out["volume"]))
            weights.append(0.30)

        # 3. Persistence
        if "persistence_score" in out.columns:
            components.append(_percentile_rank(out["persistence_score"]))
            weights.append(0.25)

        # 4. Price impact (lower impact = easier for big money to move → invert)
        if "price_impact" in out.columns:
            inv_impact = 100 - _percentile_rank(out["price_impact"].fillna(0))
            components.append(inv_impact)
            weights.append(0.15)

        if not components:
            out["smart_money_score"] = 0.0
        else:
            # Normalise weights
            total_w = sum(weights)
            norm_w  = [w / total_w for w in weights]
            score = sum(c * w for c, w in zip(components, norm_w))
            out["smart_money_score"] = score.round(2)

        # Label
        def label(s):
            if s >= 75:  return "🔥 Strong"
            if s >= 55:  return "📈 Moderate"
            if s >= 35:  return "➡️  Neutral"
            return "📉 Weak"

        out["smart_money_label"] = out["smart_money_score"].apply(label)

    except Exception as e:
        logger.error(f"smart_money_score error: {e}")

    return out


# ── Master enrichment pipeline ────────────────────────────────────────────────

def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Run all analytics in sequence. Safe — returns best-effort result."""
    if is_empty_or_none(df):
        return pd.DataFrame()
    try:
        df = order_flow_signal(df)
        df = detect_large_activity(df)
        df = liquidity_metrics(df)
        df = smart_money_score(df)
    except Exception as e:
        logger.error(f"enrich_dataframe pipeline error: {e}")
    return df
