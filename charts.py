"""
NEPSE Charts Module
All visualisations use Plotly exclusively.
"""

import logging
from typing import Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# ── Shared theme ───────────────────────────────────────────────────────────────
DARK_BG   = "#0D1117"
CARD_BG   = "#161B22"
ACCENT    = "#00D4FF"
GREEN     = "#00E676"
RED       = "#FF5252"
AMBER     = "#FFB300"
PURPLE    = "#B388FF"
TEXT_MAIN = "#E6EDF3"
TEXT_DIM  = "#8B949E"
GRID_COL  = "#21262D"

LAYOUT_DEFAULTS = dict(
    paper_bgcolor=DARK_BG,
    plot_bgcolor=CARD_BG,
    font=dict(family="'JetBrains Mono', 'Courier New', monospace", color=TEXT_MAIN, size=12),
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
)

AXIS_DEFAULTS = dict(
    showgrid=True,
    gridcolor=GRID_COL,
    zeroline=False,
    color=TEXT_DIM,
    tickfont=dict(size=10),
)


def _safe(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns and not df[col].isna().all()


# ── 1. Volume Bar Chart ────────────────────────────────────────────────────────

def volume_bar_chart(df: pd.DataFrame, top_n: int = 25) -> go.Figure:
    """Horizontal bar chart of top N stocks by volume."""
    fig = go.Figure()
    try:
        if df.empty or "volume" not in df.columns or "symbol" not in df.columns:
            return _empty_fig("Volume data unavailable")

        plot_df = (
            df[["symbol", "volume", "pct_change"] if "pct_change" in df.columns else ["symbol", "volume"]]
            .dropna(subset=["volume"])
            .nlargest(top_n, "volume")
            .sort_values("volume")
        )

        colors = (
            [GREEN if x >= 0 else RED for x in plot_df["pct_change"]]
            if "pct_change" in plot_df.columns
            else [ACCENT] * len(plot_df)
        )

        fig.add_trace(go.Bar(
            x=plot_df["volume"],
            y=plot_df["symbol"],
            orientation="h",
            marker=dict(color=colors, opacity=0.85),
            hovertemplate="<b>%{y}</b><br>Volume: %{x:,.0f}<extra></extra>",
        ))

        fig.update_layout(
            title=dict(text=f"📊 Top {top_n} Stocks by Volume", font=dict(size=15, color=ACCENT)),
            xaxis=dict(**AXIS_DEFAULTS, title="Volume"),
            yaxis=dict(**AXIS_DEFAULTS, title=""),
            height=max(350, top_n * 20),
            **LAYOUT_DEFAULTS,
        )
    except Exception as e:
        logger.error(f"volume_bar_chart error: {e}")
        return _empty_fig("Chart error")
    return fig


# ── 2. % Change Distribution ──────────────────────────────────────────────────

def pct_change_distribution(df: pd.DataFrame) -> go.Figure:
    """Histogram of pct_change with KDE overlay."""
    fig = go.Figure()
    try:
        if df.empty or "pct_change" not in df.columns:
            return _empty_fig("No % change data")

        pct = df["pct_change"].dropna()
        if pct.empty:
            return _empty_fig("No % change data")

        # Histogram
        fig.add_trace(go.Histogram(
            x=pct,
            nbinsx=40,
            marker=dict(
                color=[GREEN if x >= 0 else RED for x in pct],
                line=dict(width=0),
            ),
            opacity=0.7,
            name="Frequency",
            hovertemplate="Change: %{x:.2f}%<br>Count: %{y}<extra></extra>",
        ))

        # Zero line
        fig.add_vline(x=0, line=dict(color=AMBER, dash="dash", width=1.5))

        # Mean line
        mean_val = pct.mean()
        fig.add_vline(
            x=mean_val,
            line=dict(color=ACCENT, dash="dot", width=1.5),
            annotation_text=f"μ={mean_val:.2f}%",
            annotation_font_color=ACCENT,
        )

        fig.update_layout(
            title=dict(text="📉 % Change Distribution", font=dict(size=15, color=ACCENT)),
            xaxis=dict(**AXIS_DEFAULTS, title="% Change"),
            yaxis=dict(**AXIS_DEFAULTS, title="Count"),
            bargap=0.05,
            showlegend=False,
            height=350,
            **LAYOUT_DEFAULTS,
        )
    except Exception as e:
        logger.error(f"pct_change_distribution error: {e}")
        return _empty_fig("Chart error")
    return fig


# ── 3. Smart Money Heatmap ────────────────────────────────────────────────────

def smart_money_heatmap(df: pd.DataFrame, top_n: int = 40) -> go.Figure:
    """Scatter plot styled as heatmap: Volume vs % Change, coloured by smart_money_score."""
    fig = go.Figure()
    try:
        needed = ["symbol", "pct_change", "volume"]
        if df.empty or any(c not in df.columns for c in needed):
            return _empty_fig("Insufficient data for heatmap")

        plot_df = df.dropna(subset=needed).copy()
        if plot_df.empty:
            return _empty_fig("No valid data")

        score_col = "smart_money_score" if "smart_money_score" in plot_df.columns else None
        size_col  = "volume"

        # Normalise size
        vol = plot_df["volume"].fillna(0)
        vol_norm = ((vol - vol.min()) / (vol.max() - vol.min() + 1) * 30 + 6).clip(6, 40)

        hover = (
            "<b>%{customdata[0]}</b><br>"
            "% Change: %{x:.2f}%<br>"
            "Volume: %{y:,.0f}<br>"
            "Smart Money: %{marker.color:.1f}<extra></extra>"
        )

        fig.add_trace(go.Scatter(
            x=plot_df["pct_change"],
            y=plot_df["volume"],
            mode="markers",
            marker=dict(
                size=vol_norm,
                color=plot_df[score_col] if score_col else ACCENT,
                colorscale="Plasma",
                showscale=True if score_col else False,
                colorbar=dict(title="Score", tickfont=dict(color=TEXT_DIM)),
                line=dict(width=0.5, color=DARK_BG),
                opacity=0.85,
            ),
            customdata=plot_df[["symbol"]].values,
            hovertemplate=hover,
        ))

        # Quadrant lines
        fig.add_vline(x=0, line=dict(color=GRID_COL, width=1))
        fig.add_hline(y=plot_df["volume"].median(), line=dict(color=GRID_COL, dash="dot", width=1))

        fig.update_layout(
            title=dict(text="🔥 Smart Money Heatmap — Volume × % Change", font=dict(size=15, color=ACCENT)),
            xaxis=dict(**AXIS_DEFAULTS, title="% Change"),
            yaxis=dict(**AXIS_DEFAULTS, title="Volume", type="log"),
            height=450,
            **LAYOUT_DEFAULTS,
        )
    except Exception as e:
        logger.error(f"smart_money_heatmap error: {e}")
        return _empty_fig("Chart error")
    return fig


# ── 4. Price vs Volume Impact ─────────────────────────────────────────────────

def price_volume_impact_chart(df: pd.DataFrame, top_n: int = 30) -> go.Figure:
    """Bubble chart: LTP vs Volume, size = price_impact."""
    fig = go.Figure()
    try:
        needed = ["symbol", "ltp", "volume"]
        if df.empty or any(c not in df.columns for c in needed):
            return _empty_fig("Insufficient data")

        plot_df = df.dropna(subset=needed).copy()
        impact_col = "price_impact" if "price_impact" in plot_df.columns else None

        # Take top N by volume
        plot_df = plot_df.nlargest(top_n, "volume")

        size_vals = (
            np.log1p(plot_df[impact_col].fillna(0)) * 10 + 5
            if impact_col else
            pd.Series(10, index=plot_df.index)
        ).clip(5, 50)

        colors = (
            [GREEN if x >= 0 else RED for x in plot_df["pct_change"]]
            if "pct_change" in plot_df.columns
            else [ACCENT] * len(plot_df)
        )

        fig.add_trace(go.Scatter(
            x=plot_df["volume"],
            y=plot_df["ltp"],
            mode="markers+text",
            text=plot_df["symbol"],
            textposition="top center",
            textfont=dict(size=8, color=TEXT_DIM),
            marker=dict(
                size=size_vals,
                color=colors,
                opacity=0.8,
                line=dict(width=0.5, color=DARK_BG),
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "LTP: Rs %{y:,.2f}<br>"
                "Volume: %{x:,.0f}<extra></extra>"
            ),
        ))

        fig.update_layout(
            title=dict(text="💡 Price vs Volume Impact", font=dict(size=15, color=ACCENT)),
            xaxis=dict(**AXIS_DEFAULTS, title="Volume", type="log"),
            yaxis=dict(**AXIS_DEFAULTS, title="LTP (Rs)"),
            height=420,
            **LAYOUT_DEFAULTS,
        )
    except Exception as e:
        logger.error(f"price_volume_impact_chart error: {e}")
        return _empty_fig("Chart error")
    return fig


# ── 5. Market Breadth Gauge ───────────────────────────────────────────────────

def market_breadth_gauge(advances: int, declines: int, unchanged: int) -> go.Figure:
    """Donut chart showing advance/decline/unchanged split."""
    try:
        total = advances + declines + unchanged
        if total == 0:
            return _empty_fig("No breadth data")

        fig = go.Figure(go.Pie(
            labels=["Advances", "Declines", "Unchanged"],
            values=[advances, declines, unchanged],
            hole=0.60,
            marker=dict(colors=[GREEN, RED, AMBER]),
            textinfo="label+percent",
            hovertemplate="%{label}: %{value} stocks (%{percent})<extra></extra>",
        ))

        ratio = advances / (advances + declines) if (advances + declines) > 0 else 0.5
        sentiment = "Bullish" if ratio > 0.55 else ("Bearish" if ratio < 0.45 else "Neutral")
        sent_color = GREEN if sentiment == "Bullish" else (RED if sentiment == "Bearish" else AMBER)

        fig.update_layout(
            title=dict(text="🌡️ Market Breadth", font=dict(size=15, color=ACCENT)),
            annotations=[dict(
                text=f"<b>{sentiment}</b>",
                x=0.5, y=0.5,
                font=dict(size=18, color=sent_color),
                showarrow=False,
            )],
            height=320,
            **LAYOUT_DEFAULTS,
        )
        return fig
    except Exception as e:
        logger.error(f"market_breadth_gauge error: {e}")
        return _empty_fig("Chart error")


# ── 6. Top Movers Chart ───────────────────────────────────────────────────────

def top_movers_chart(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Horizontal bar chart for top gainers and losers side-by-side."""
    fig = go.Figure()
    try:
        if df.empty or "pct_change" not in df.columns or "symbol" not in df.columns:
            return _empty_fig("No data available")

        valid = df.dropna(subset=["pct_change", "symbol"])
        gainers = valid.nlargest(top_n, "pct_change")
        losers  = valid.nsmallest(top_n, "pct_change")

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("🟢 Top Gainers", "🔴 Top Losers"),
        )

        fig.add_trace(go.Bar(
            x=gainers["pct_change"],
            y=gainers["symbol"],
            orientation="h",
            marker=dict(color=GREEN, opacity=0.85),
            hovertemplate="<b>%{y}</b>: +%{x:.2f}%<extra></extra>",
            name="Gainers",
        ), row=1, col=1)

        fig.add_trace(go.Bar(
            x=losers["pct_change"].abs(),
            y=losers["symbol"],
            orientation="h",
            marker=dict(color=RED, opacity=0.85),
            hovertemplate="<b>%{y}</b>: -%{x:.2f}%<extra></extra>",
            name="Losers",
        ), row=1, col=2)

        fig.update_layout(
            title=dict(text="📈 Top Movers", font=dict(size=15, color=ACCENT)),
            showlegend=False,
            height=380,
            **LAYOUT_DEFAULTS,
        )
        for ax in ("xaxis", "xaxis2", "yaxis", "yaxis2"):
            fig.update_layout(**{ax: AXIS_DEFAULTS})

    except Exception as e:
        logger.error(f"top_movers_chart error: {e}")
        return _empty_fig("Chart error")
    return fig


# ── 7. Smart Money Score Bar ──────────────────────────────────────────────────

def smart_money_bar(df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    """Bar chart of top N stocks by smart money score."""
    fig = go.Figure()
    try:
        if df.empty or "smart_money_score" not in df.columns:
            return _empty_fig("Smart money scores unavailable")

        plot_df = (
            df[["symbol", "smart_money_score"]]
            .dropna()
            .nlargest(top_n, "smart_money_score")
            .sort_values("smart_money_score")
        )

        cmap_vals = plot_df["smart_money_score"] / 100
        colors = [
            f"rgba({int(255*(1-v))},{int(200*v)},{int(255*v)},0.85)"
            for v in cmap_vals
        ]

        fig.add_trace(go.Bar(
            x=plot_df["smart_money_score"],
            y=plot_df["symbol"],
            orientation="h",
            marker=dict(color=colors),
            hovertemplate="<b>%{y}</b><br>Score: %{x:.1f}/100<extra></extra>",
        ))

        fig.update_layout(
            title=dict(text="🧠 Smart Money Score — Top Candidates", font=dict(size=15, color=ACCENT)),
            xaxis=dict(**AXIS_DEFAULTS, title="Score (0-100)", range=[0, 105]),
            yaxis=dict(**AXIS_DEFAULTS),
            height=max(300, top_n * 22),
            **LAYOUT_DEFAULTS,
        )
    except Exception as e:
        logger.error(f"smart_money_bar error: {e}")
        return _empty_fig("Chart error")
    return fig


# ── Utility ───────────────────────────────────────────────────────────────────

def _empty_fig(message: str = "No data") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5, y=0.5, xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=16, color=TEXT_DIM),
    )
    fig.update_layout(height=300, **LAYOUT_DEFAULTS)
    return fig
