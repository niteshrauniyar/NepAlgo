"""
NEPSE Trading Intelligence Dashboard
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Production-grade Streamlit app with multi-source data pipeline,
smart money scoring, and institutional activity detection.
"""

import os
import sys
import logging
import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# Guarantee the project root is importable regardless of CWD.
# os.path.dirname(__file__) can be empty string when Streamlit Cloud
# sets CWD to the repo root, so we always resolve to an absolute path.
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data_engine import (
    get_market_data, market_summary, enrich_dataframe,
)
from charts import (
    volume_bar_chart, pct_change_distribution, smart_money_heatmap,
    price_volume_impact_chart, market_breadth_gauge,
    top_movers_chart, smart_money_bar,
)
from utils import fmt_number, setup_logging

setup_logging(logging.WARNING)   # Quiet in Streamlit

# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NEPSE Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Syne:wght@400;600;700;800&display=swap');

  :root {
    --bg:       #0D1117;
    --card:     #161B22;
    --border:   #21262D;
    --accent:   #00D4FF;
    --green:    #00E676;
    --red:      #FF5252;
    --amber:    #FFB300;
    --purple:   #B388FF;
    --text:     #E6EDF3;
    --dim:      #8B949E;
  }

  html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text);
    font-family: 'JetBrains Mono', monospace;
  }

  /* Header */
  .nepse-header {
    background: linear-gradient(135deg, #0D1117 0%, #161B22 50%, #0D1117 100%);
    border-bottom: 1px solid var(--accent);
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
  }
  .nepse-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
      90deg,
      transparent,
      transparent 40px,
      rgba(0,212,255,0.03) 40px,
      rgba(0,212,255,0.03) 41px
    );
  }
  .nepse-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(90deg, var(--accent), var(--purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
  }
  .nepse-subtitle {
    color: var(--dim);
    font-size: 0.8rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 0.25rem;
  }

  /* Metric cards */
  .metric-grid { display: grid; gap: 1rem; }
  .metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    transition: border-color 0.2s;
  }
  .metric-card:hover { border-color: var(--accent); }
  .metric-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--dim);
    margin-bottom: 0.4rem;
  }
  .metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text);
    line-height: 1;
  }
  .metric-delta {
    font-size: 0.75rem;
    margin-top: 0.3rem;
  }
  .up   { color: var(--green); }
  .down { color: var(--red);   }
  .flat { color: var(--amber); }

  /* Section headers */
  .section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: 0.05em;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem;
  }

  /* Status badges */
  .badge {
    display: inline-block;
    padding: 0.15rem 0.6rem;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  .badge-live    { background: rgba(0,230,118,0.15); color: var(--green); border: 1px solid var(--green); }
  .badge-cached  { background: rgba(255,179,0,0.15); color: var(--amber); border: 1px solid var(--amber); }
  .badge-error   { background: rgba(255,82,82,0.15);  color: var(--red);   border: 1px solid var(--red);   }
  .badge-demo    { background: rgba(179,136,255,0.15);color: var(--purple);border: 1px solid var(--purple);}

  /* Smart money table */
  .sm-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
  .sm-table th {
    text-align: left;
    color: var(--dim);
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 0.4rem 0.6rem;
    border-bottom: 1px solid var(--border);
  }
  .sm-table td {
    padding: 0.4rem 0.6rem;
    border-bottom: 1px solid rgba(33,38,45,0.6);
    vertical-align: middle;
  }
  .sm-table tr:hover td { background: rgba(0,212,255,0.04); }

  /* Score bar */
  .score-bar-outer {
    background: var(--border);
    border-radius: 4px;
    height: 6px;
    width: 80px;
    display: inline-block;
    vertical-align: middle;
  }
  .score-bar-inner {
    height: 100%;
    border-radius: 4px;
  }

  /* Source status */
  .source-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.6rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.82rem;
  }
  .source-name { flex: 1; color: var(--text); }
  .dot { width: 8px; height: 8px; border-radius: 50%; }
  .dot-green  { background: var(--green); box-shadow: 0 0 6px var(--green); }
  .dot-red    { background: var(--red);   box-shadow: 0 0 6px var(--red);   }
  .dot-amber  { background: var(--amber); box-shadow: 0 0 6px var(--amber); }

  /* Sidebar */
  .css-1d391kg, [data-testid="stSidebar"] {
    background: var(--card) !important;
    border-right: 1px solid var(--border);
  }

  /* Plotly charts background fix */
  .js-plotly-plot .plotly .main-svg { background: transparent !important; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

  /* Streamlit overrides */
  .stDataFrame { background: var(--card); }
  div[data-testid="stMetric"] { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 1rem; }
  .stButton > button {
    background: transparent;
    border: 1px solid var(--accent);
    color: var(--accent);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.1em;
    padding: 0.4rem 1.2rem;
    border-radius: 6px;
    transition: all 0.2s;
  }
  .stButton > button:hover {
    background: var(--accent);
    color: var(--bg);
  }
  .stSelectbox > div, .stMultiSelect > div {
    background: var(--card) !important;
    border-color: var(--border) !important;
  }
  h1, h2, h3 { font-family: 'Syne', sans-serif !important; }
  .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING (CACHED)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def load_data():
    """Fetch, normalize, and enrich market data. Cached for 5 minutes."""
    df, source, status = get_market_data()
    if df is None or df.empty:
        return None, source, status
    df = enrich_dataframe(df)
    return df, source, status


def generate_demo_data() -> pd.DataFrame:
    """
    Generate realistic NEPSE-style demo data when all sources fail.
    Used ONLY as last resort for UI demonstration.
    """
    np.random.seed(42)
    symbols = [
        "NABIL","NICA","SBI","NMB","SANIMA","GBIME","KBL","MBL","PCBL","PRVU",
        "BOKL","CBL","CCBL","CZBIL","EBL","HBL","HIDCL","JBNL","MEGA","NBL",
        "ADBL","NGADI","NHPC","NIB","NIMB","NLG","NLIC","NLICL","NTC","ORTC",
        "PBBL","PLIC","PMHPL","RBBI","RBBL","SADBL","SAPDBL","SCB","SIFC","SIGS2",
        "SHL","SICL","SLBSL","SRBL","TRH","UAIL","UPPER","VLBS","WOMI","YETI",
    ]
    n = len(symbols)
    ltp       = np.random.uniform(100, 3000, n).round(2)
    pct_chg   = np.random.normal(0.5, 3.5, n).round(2).clip(-10, 10)
    prev_cl   = (ltp / (1 + pct_chg / 100)).round(2)
    volume    = np.random.exponential(50000, n).astype(int) + 1000
    turnover  = (ltp * volume * np.random.uniform(0.8, 1.2, n)).round(0)
    trans     = (volume / np.random.uniform(10, 50, n)).astype(int)

    return pd.DataFrame({
        "symbol":       symbols,
        "ltp":          ltp,
        "prev_close":   prev_cl,
        "pct_change":   pct_chg,
        "volume":       volume,
        "turnover":     turnover,
        "transactions": trans,
        "open":         (ltp * np.random.uniform(0.97, 1.03, n)).round(2),
        "high":         (ltp * np.random.uniform(1.00, 1.08, n)).round(2),
        "low":          (ltp * np.random.uniform(0.92, 1.00, n)).round(2),
    })


# ══════════════════════════════════════════════════════════════════════════════
# HELPER RENDERERS
# ══════════════════════════════════════════════════════════════════════════════

def render_metric_card(label: str, value: str, delta: str = "", delta_class: str = "flat"):
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      {"<div class='metric-delta " + delta_class + "'>" + delta + "</div>" if delta else ""}
    </div>
    """, unsafe_allow_html=True)


def score_color(score: float) -> str:
    if score >= 75:  return "#00E676"
    if score >= 55:  return "#69F0AE"
    if score >= 35:  return "#FFB300"
    return "#FF5252"


def render_smart_money_table(df: pd.DataFrame, top_n: int = 30):
    if "smart_money_score" not in df.columns:
        st.warning("Smart money scores not available.")
        return

    cols_show = ["symbol", "smart_money_score", "smart_money_label",
                 "pct_change", "volume", "buy_pressure", "sell_pressure"]
    cols_avail = [c for c in cols_show if c in df.columns]

    top = (
        df[cols_avail]
        .dropna(subset=["smart_money_score"])
        .nlargest(top_n, "smart_money_score")
        .reset_index(drop=True)
    )

    rows_html = ""
    for _, row in top.iterrows():
        score = float(row.get("smart_money_score", 0))
        color = score_color(score)
        bar_w = int(score * 0.8)   # max 80px

        pct   = row.get("pct_change", 0)
        pct_c = "up" if pct > 0 else ("down" if pct < 0 else "flat")
        pct_s = f"+{pct:.2f}%" if pct > 0 else f"{pct:.2f}%"

        vol   = fmt_number(row.get("volume", 0), 1)
        bp    = f"{row.get('buy_pressure', 0):.3f}"  if "buy_pressure"  in cols_avail else "—"
        sp    = f"{row.get('sell_pressure', 0):.3f}" if "sell_pressure" in cols_avail else "—"
        label = row.get("smart_money_label", "—")

        rows_html += f"""
        <tr>
          <td><b style="color:#E6EDF3">{row['symbol']}</b></td>
          <td>
            <span style="color:{color};font-weight:700">{score:.1f}</span>
            <span class="score-bar-outer">
              <span class="score-bar-inner" style="width:{bar_w}%;background:{color}"></span>
            </span>
          </td>
          <td>{label}</td>
          <td class="{pct_c}">{pct_s}</td>
          <td style="color:#8B949E">{vol}</td>
          <td class="up">{bp}</td>
          <td class="down">{sp}</td>
        </tr>"""

    st.markdown(f"""
    <table class="sm-table">
      <thead>
        <tr>
          <th>Symbol</th><th>Score</th><th>Signal</th>
          <th>% Chg</th><th>Volume</th><th>Buy ↑</th><th>Sell ↓</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar(df, source, status, summary):
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:1rem 0 0.5rem">
          <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;
                      background:linear-gradient(90deg,#00D4FF,#B388FF);
                      -webkit-background-clip:text;-webkit-text-fill-color:transparent">
            NEPSE Intel
          </div>
          <div style="color:#8B949E;font-size:0.65rem;letter-spacing:0.2em;
                      text-transform:uppercase;margin-top:0.2rem">
            Trading Dashboard
          </div>
        </div>
        <hr style="border-color:#21262D;margin:0.8rem 0">
        """, unsafe_allow_html=True)

        # Refresh button
        if st.button("⟳  Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

        # Source status
        st.markdown("<div style='font-size:0.7rem;text-transform:uppercase;letter-spacing:0.12em;color:#8B949E;margin-bottom:0.5rem'>Data Sources</div>", unsafe_allow_html=True)

        sources_info = [
            ("NEPSE Official API", "amber" if status != "live" or source != "NEPSE Official API" else "green",
             "Primary"),
            ("ShareSansar",        "green" if source == "ShareSansar" and status == "live" else "red",
             "Secondary"),
            ("NepseAlpha",         "green" if source == "NepseAlpha"  and status == "live" else "red",
             "Tertiary"),
        ]
        for sname, color, priority in sources_info:
            is_active = sname == source
            st.markdown(f"""
            <div class="source-row">
              <div class="dot dot-{color}"></div>
              <div class="source-name" style="{'font-weight:600' if is_active else ''}">{sname}</div>
              <div style="color:#8B949E;font-size:0.65rem">{priority}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        # Market quick stats
        if summary and summary.get("total_stocks", 0) > 0:
            st.markdown("<div style='font-size:0.7rem;text-transform:uppercase;letter-spacing:0.12em;color:#8B949E;margin-bottom:0.5rem'>Market Pulse</div>", unsafe_allow_html=True)

            sent = summary.get("market_sentiment", "neutral")
            sent_color = "#00E676" if sent == "bullish" else ("#FF5252" if sent == "bearish" else "#FFB300")

            st.markdown(f"""
            <div style="background:#161B22;border:1px solid #21262D;border-radius:8px;padding:0.8rem;font-size:0.8rem">
              <div style="display:flex;justify-content:space-between;margin-bottom:0.4rem">
                <span style="color:#8B949E">Sentiment</span>
                <span style="color:{sent_color};font-weight:700;text-transform:capitalize">{sent}</span>
              </div>
              <div style="display:flex;justify-content:space-between;margin-bottom:0.4rem">
                <span style="color:#8B949E">Advances</span>
                <span style="color:#00E676">{summary.get('advances', 0)}</span>
              </div>
              <div style="display:flex;justify-content:space-between;margin-bottom:0.4rem">
                <span style="color:#8B949E">Declines</span>
                <span style="color:#FF5252">{summary.get('declines', 0)}</span>
              </div>
              <div style="display:flex;justify-content:space-between">
                <span style="color:#8B949E">Avg Chg</span>
                <span style="color:#E6EDF3">{summary.get('avg_pct_change', 0):.2f}%</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        # Filters
        st.markdown("<div style='font-size:0.7rem;text-transform:uppercase;letter-spacing:0.12em;color:#8B949E;margin-bottom:0.5rem'>Filters</div>", unsafe_allow_html=True)

        min_vol = st.number_input("Min Volume", min_value=0, value=0, step=1000)
        pct_range = st.slider("% Change Range", -15.0, 15.0, (-15.0, 15.0), 0.5)

        # Timestamp
        st.markdown(f"""
        <hr style="border-color:#21262D;margin:1rem 0 0.5rem">
        <div style="color:#8B949E;font-size:0.65rem;text-align:center">
          Last updated: {datetime.now().strftime('%H:%M:%S')}<br>
          Source: <span style="color:#00D4FF">{source}</span>
        </div>
        """, unsafe_allow_html=True)

        return min_vol, pct_range


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Header
    st.markdown("""
    <div class="nepse-header">
      <h1 class="nepse-title">NEPSE Trading Intelligence</h1>
      <div class="nepse-subtitle">Real-Time Market Data · Smart Money Detection · Institutional Activity</div>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    with st.spinner("Fetching market data…"):
        df, source, status = load_data()

    is_demo = False
    if df is None or df.empty:
        st.warning(
            "⚠️ All live data sources failed and no cache is available. "
            "Showing **demo data** for UI demonstration."
        )
        from data_engine import enrich_dataframe
        from utils import save_cache
        df = generate_demo_data()
        df = enrich_dataframe(df)
        source = "Demo (Synthetic)"
        status = "demo"
        is_demo = True

    summary = market_summary(df)

    # Sidebar (returns filters)
    min_vol, pct_range = render_sidebar(df, source, status, summary)

    # Apply filters
    filtered = df.copy()
    if "volume" in filtered.columns:
        filtered = filtered[filtered["volume"] >= min_vol]
    if "pct_change" in filtered.columns:
        filtered = filtered[
            filtered["pct_change"].between(pct_range[0], pct_range[1])
        ]

    # Status bar
    badge_class = {"live": "badge-live", "demo": "badge-demo"}.get(
        status.split(" ")[0], "badge-cached"
    )
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:1.5rem;
                padding:0.6rem 1rem;background:#161B22;border-radius:8px;
                border:1px solid #21262D">
      <span class="badge {badge_class}">{status}</span>
      <span style="color:#8B949E;font-size:0.8rem">Source: <b style="color:#E6EDF3">{source}</b></span>
      <span style="color:#8B949E;font-size:0.8rem">·</span>
      <span style="color:#8B949E;font-size:0.8rem">{len(df)} stocks loaded</span>
      <span style="color:#8B949E;font-size:0.8rem">·</span>
      <span style="color:#8B949E;font-size:0.8rem">{len(filtered)} after filters</span>
    </div>
    """, unsafe_allow_html=True)

    # ── TAB LAYOUT ─────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "📊 Market Overview",
        "🧠 Smart Money",
        "📈 Top Movers",
        "🔥 Most Active",
        "💧 Liquidity",
        "⚙️ Data Source",
    ])

    # ══════════════════════════════════════════════════════════════════════
    # TAB 1 — Market Overview
    # ══════════════════════════════════════════════════════════════════════
    with tabs[0]:
        st.markdown("<div class='section-header'>Market Overview</div>", unsafe_allow_html=True)

        # KPI row
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            render_metric_card("Total Stocks",  str(summary.get("total_stocks", 0)))
        with c2:
            render_metric_card("Advances",  str(summary.get("advances", 0)),  delta="↑ Positive", delta_class="up")
        with c3:
            render_metric_card("Declines",  str(summary.get("declines", 0)),  delta="↓ Negative", delta_class="down")
        with c4:
            render_metric_card("Unchanged", str(summary.get("unchanged", 0)), delta="→ Flat",      delta_class="flat")
        with c5:
            avg = summary.get("avg_pct_change", 0)
            dc  = "up" if avg >= 0 else "down"
            render_metric_card("Avg % Chg", f"{avg:+.2f}%", delta_class=dc)
        with c6:
            render_metric_card("Breadth", f"{summary.get('breadth', 0):.3f}")

        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

        # Second row
        c1, c2, c3 = st.columns(3)
        with c1:
            render_metric_card(
                "Total Volume",
                fmt_number(summary.get("total_volume", 0), 2),
            )
        with c2:
            render_metric_card(
                "Total Turnover",
                "Rs " + fmt_number(summary.get("total_turnover", 0), 2),
            )
        with c3:
            sent = summary.get("market_sentiment", "neutral").upper()
            sc   = "up" if sent == "BULLISH" else ("down" if sent == "BEARISH" else "flat")
            render_metric_card("Market Sentiment", sent, delta_class=sc)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        # Charts row
        col_a, col_b = st.columns([1, 1])
        with col_a:
            adv = summary.get("advances", 0)
            dec = summary.get("declines", 0)
            unc = summary.get("unchanged", 0)
            st.plotly_chart(
                market_breadth_gauge(adv, dec, unc),
                use_container_width=True, config={"displayModeBar": False}
            )
        with col_b:
            st.plotly_chart(
                pct_change_distribution(filtered),
                use_container_width=True, config={"displayModeBar": False}
            )

        # Top gainer / loser / most active highlight
        st.markdown("<div class='section-header'>Market Highlights</div>", unsafe_allow_html=True)
        h1, h2, h3 = st.columns(3)

        tg = summary.get("top_gainer", {})
        tl = summary.get("top_loser", {})
        ma = summary.get("most_active", {})

        with h1:
            st.markdown(f"""
            <div class="metric-card" style="border-color:#00E676">
              <div class="metric-label">🏆 Top Gainer</div>
              <div class="metric-value" style="color:#00E676">{tg.get('symbol','—')}</div>
              <div class="metric-delta up">+{tg.get('pct_change',0):.2f}%
                {"· Rs " + fmt_number(tg.get('ltp',0),2) if tg.get('ltp') else ""}
              </div>
            </div>""", unsafe_allow_html=True)

        with h2:
            st.markdown(f"""
            <div class="metric-card" style="border-color:#FF5252">
              <div class="metric-label">📉 Top Loser</div>
              <div class="metric-value" style="color:#FF5252">{tl.get('symbol','—')}</div>
              <div class="metric-delta down">{tl.get('pct_change',0):.2f}%
                {"· Rs " + fmt_number(tl.get('ltp',0),2) if tl.get('ltp') else ""}
              </div>
            </div>""", unsafe_allow_html=True)

        with h3:
            st.markdown(f"""
            <div class="metric-card" style="border-color:#00D4FF">
              <div class="metric-label">🔥 Most Active</div>
              <div class="metric-value" style="color:#00D4FF">{ma.get('symbol','—')}</div>
              <div class="metric-delta flat">Vol: {fmt_number(ma.get('volume',0),1)}</div>
            </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 2 — Smart Money
    # ══════════════════════════════════════════════════════════════════════
    with tabs[1]:
        st.markdown("<div class='section-header'>Smart Money & Institutional Signals ⭐</div>",
                    unsafe_allow_html=True)

        if is_demo:
            st.info("📌 Showing synthetic demo data. Connect to a live source for real signals.")

        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.plotly_chart(
                smart_money_bar(filtered, top_n=20),
                use_container_width=True, config={"displayModeBar": False}
            )
        with col_b:
            st.plotly_chart(
                smart_money_heatmap(filtered, top_n=50),
                use_container_width=True, config={"displayModeBar": False}
            )

        st.markdown("<div class='section-header'>Top Smart Money Candidates</div>",
                    unsafe_allow_html=True)
        render_smart_money_table(filtered, top_n=30)

        # Score methodology expander
        with st.expander("📐 Score Methodology"):
            st.markdown("""
            The **Smart Money Score (0–100)** is a composite proxy for institutional / informed-money activity.
            Since NEPSE doesn't expose full order book data, the score approximates activity via:

            | Component | Weight | Signal |
            |-----------|--------|--------|
            | Price Momentum | 30% | % change percentile rank |
            | Volume Spike | 30% | Volume vs market median (z-score) |
            | Persistence | 25% | abs(% change) × volume rank |
            | Liquidity Impact | 15% | Inverted price impact proxy |

            **Interpretation:**
            - 🔥 75–100: Strong institutional interest
            - 📈 55–74: Moderate smart money activity
            - ➡️ 35–54: Neutral / retail-driven
            - 📉 0–34: Low conviction / weak signals

            *This is an approximation — always combine with fundamental analysis.*
            """)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 3 — Top Movers
    # ══════════════════════════════════════════════════════════════════════
    with tabs[2]:
        st.markdown("<div class='section-header'>Top Gainers & Losers</div>", unsafe_allow_html=True)

        st.plotly_chart(
            top_movers_chart(filtered, top_n=15),
            use_container_width=True, config={"displayModeBar": False}
        )

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("<div class='section-header'>🟢 Top 15 Gainers</div>", unsafe_allow_html=True)
            if "pct_change" in filtered.columns:
                gainers = (
                    filtered[["symbol", "pct_change", "ltp", "volume"] if all(c in filtered.columns for c in ["ltp","volume"]) else ["symbol","pct_change"]]
                    .dropna(subset=["pct_change"])
                    .nlargest(15, "pct_change")
                    .reset_index(drop=True)
                )
                gainers.index += 1
                st.dataframe(gainers.style.format(
                    {c: "{:.2f}" for c in gainers.select_dtypes("number").columns}
                ), use_container_width=True)

        with col_b:
            st.markdown("<div class='section-header'>🔴 Top 15 Losers</div>", unsafe_allow_html=True)
            if "pct_change" in filtered.columns:
                losers = (
                    filtered[["symbol", "pct_change", "ltp", "volume"] if all(c in filtered.columns for c in ["ltp","volume"]) else ["symbol","pct_change"]]
                    .dropna(subset=["pct_change"])
                    .nsmallest(15, "pct_change")
                    .reset_index(drop=True)
                )
                losers.index += 1
                st.dataframe(losers.style.format(
                    {c: "{:.2f}" for c in losers.select_dtypes("number").columns}
                ), use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 4 — Most Active
    # ══════════════════════════════════════════════════════════════════════
    with tabs[3]:
        st.markdown("<div class='section-header'>Most Active Stocks</div>", unsafe_allow_html=True)

        st.plotly_chart(
            volume_bar_chart(filtered, top_n=25),
            use_container_width=True, config={"displayModeBar": False}
        )

        if "large_activity_flag" in filtered.columns:
            flagged = filtered[filtered["large_activity_flag"] == True]
            if not flagged.empty:
                st.markdown(
                    f"<div class='section-header'>⚡ Unusual Volume Detected ({len(flagged)} stocks)</div>",
                    unsafe_allow_html=True
                )
                show_cols = [c for c in ["symbol", "volume", "volume_ratio", "pct_change", "turnover"] if c in flagged.columns]
                st.dataframe(
                    flagged[show_cols]
                    .sort_values("volume", ascending=False)
                    .reset_index(drop=True)
                    .style.format({c: "{:.2f}" for c in flagged[show_cols].select_dtypes("number").columns}),
                    use_container_width=True,
                )

    # ══════════════════════════════════════════════════════════════════════
    # TAB 5 — Liquidity
    # ══════════════════════════════════════════════════════════════════════
    with tabs[4]:
        st.markdown("<div class='section-header'>Liquidity & Market Impact Analysis</div>",
                    unsafe_allow_html=True)

        col_a, col_b = st.columns([1.2, 0.8])
        with col_a:
            st.plotly_chart(
                price_volume_impact_chart(filtered, top_n=30),
                use_container_width=True, config={"displayModeBar": False}
            )

        with col_b:
            # Volatility summary
            if "volatility_rank" in filtered.columns:
                st.markdown("<div class='section-header'>⚡ Top Volatile Stocks</div>", unsafe_allow_html=True)
                vol_top = (
                    filtered[["symbol", "volatility_rank", "pct_change"]]
                    .dropna()
                    .nlargest(15, "volatility_rank")
                    .reset_index(drop=True)
                )
                vol_top.index += 1
                st.dataframe(
                    vol_top.style.format({c: "{:.4f}" for c in vol_top.select_dtypes("number").columns}),
                    use_container_width=True,
                )

        # Turnover spikes
        if "turnover_spike_flag" in filtered.columns:
            spikes = filtered[filtered["turnover_spike_flag"] == True]
            if not spikes.empty:
                st.markdown(
                    f"<div class='section-header'>💹 Turnover Spikes ({len(spikes)} stocks)</div>",
                    unsafe_allow_html=True
                )
                show_cols = [c for c in ["symbol", "turnover", "pct_change", "volume", "price_impact"] if c in spikes.columns]
                st.dataframe(
                    spikes[show_cols]
                    .sort_values("turnover", ascending=False)
                    .reset_index(drop=True)
                    .style.format({c: "{:.4f}" for c in spikes[show_cols].select_dtypes("number").columns}),
                    use_container_width=True,
                )

    # ══════════════════════════════════════════════════════════════════════
    # TAB 6 — Data Source Status
    # ══════════════════════════════════════════════════════════════════════
    with tabs[5]:
        st.markdown("<div class='section-header'>Data Source Status & Pipeline</div>", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### 🌐 Source Registry")
            sources_detail = [
                ("NEPSE Official API",  "https://nepalstock.com.np/api",
                 "Primary",   source == "NEPSE Official API"),
                ("ShareSansar",         "https://www.sharesansar.com/today-share-price",
                 "Secondary", source == "ShareSansar"),
                ("NepseAlpha",          "https://nepsealpha.com/nepse-data",
                 "Tertiary",  source == "NepseAlpha"),
            ]
            for sname, url, priority, is_active in sources_detail:
                dot = "dot-green" if is_active else "dot-red"
                st.markdown(f"""
                <div style="background:#161B22;border:1px solid #21262D;border-radius:8px;
                            padding:0.8rem 1rem;margin-bottom:0.5rem">
                  <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.3rem">
                    <div class="dot {dot}"></div>
                    <span style="font-weight:700;color:#E6EDF3">{sname}</span>
                    <span style="margin-left:auto;color:#8B949E;font-size:0.7rem">{priority}</span>
                  </div>
                  <div style="color:#8B949E;font-size:0.72rem;font-family:monospace">{url}</div>
                  {"<div style='color:#00E676;font-size:0.72rem;margin-top:0.3rem'>✓ Active source</div>" if is_active else ""}
                </div>
                """, unsafe_allow_html=True)

        with col_b:
            st.markdown("#### 📋 Raw Data Sample")
            sample_cols = [c for c in df.columns if c in
                           ["symbol","ltp","pct_change","volume","turnover","smart_money_score"]]
            if sample_cols:
                st.dataframe(
                    df[sample_cols].head(20)
                    .style.format({c: "{:.3f}" for c in df[sample_cols].select_dtypes("number").columns}),
                    use_container_width=True,
                )

        st.markdown("#### 🗂️ Column Schema")
        schema_df = pd.DataFrame([
            {"Standard Column": col, "Present": "✅" if col in df.columns else "❌",
             "Sample Value": str(df[col].iloc[0]) if col in df.columns and not df.empty else "—"}
            for col in ["symbol", "ltp", "pct_change", "volume", "turnover",
                        "transactions", "open", "high", "low", "prev_close",
                        "buy_pressure", "sell_pressure", "volume_ratio",
                        "smart_money_score", "price_impact", "volatility_rank"]
        ])
        st.dataframe(schema_df, use_container_width=True)

        # Pipeline architecture diagram
        with st.expander("🔁 Pipeline Architecture"):
            st.code("""
get_market_data()
│
├── fetch_from_api()           # nepalstock.com.np
│   └── Success? → normalize_market_data() → enrich_dataframe()
│
├── [FALLBACK] fetch_from_sharesansar()
│   └── Success? → normalize_market_data() → enrich_dataframe()
│
├── [FALLBACK] fetch_from_nepsealpha()
│   └── Success? → normalize_market_data() → enrich_dataframe()
│
└── [FALLBACK] load_cache()    # last successful dataset
    └── Available? → enrich_dataframe()

enrich_dataframe()
├── order_flow_signal()     → buy_pressure, sell_pressure, persistence_score
├── detect_large_activity() → volume_ratio, large_activity_flag
├── liquidity_metrics()     → volatility_rank, price_impact, turnover_spike_flag
└── smart_money_score()     → smart_money_score (0–100), smart_money_label
            """, language="text")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
