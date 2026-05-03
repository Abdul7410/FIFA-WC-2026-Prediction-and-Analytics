"""pages/sentiment.py — Live Sentiment Tracker Streamlit page."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import time, sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from modules.sentiment_tracker import load_recent_tweets, run_demo_mode, get_live_buffer_df


def sentiment_color(s: str) -> str:
    return {"positive": "🟢", "neutral": "🟡", "negative": "🔴"}.get(s, "⚪")


def show():
    st.title("🔴 Live World Cup Sentiment Tracker")
    st.markdown(
        "Streams tweets during matches, runs **HuggingFace RoBERTa** sentiment analysis, "
        "and shows how public opinion shifts in real time."
    )

    # ── Configuration panel ───────────────────────────────────────────────────
    with st.expander("⚙️ Stream configuration", expanded=False):
        st.markdown(
            """
            **To use live Twitter streaming:**
            1. Create a Twitter Developer account at [developer.twitter.com](https://developer.twitter.com)
            2. Generate a Bearer Token
            3. Copy `.env.example` → `.env` and paste your token
            4. Click **Start live stream** below

            **No API key? Use Demo Mode** — simulates a live match feed locally.
            """
        )
        query = st.text_input(
            "Search query",
            value="#WorldCup2026 OR #FIFAWorldCup lang:en -is:retweet",
        )

    col1, col2 = st.columns(2)
    demo_btn = col1.button("▶ Run Demo Mode (no API needed)", type="primary")
    refresh_btn = col2.button("🔄 Refresh dashboard")

    if demo_btn:
        with st.spinner("Simulating 20 live tweets…"):
            run_demo_mode(n_rounds=20, delay=0.1)
        st.success("Demo complete! Scroll down to see results.")

    # ── Load data ─────────────────────────────────────────────────────────────
    df = load_recent_tweets(n=500)

    if df.empty:
        st.info("No tweets yet. Click **Run Demo Mode** or start the live stream.")
        return

    # ── KPI row ───────────────────────────────────────────────────────────────
    st.markdown("---")
    total = len(df)
    pos_pct = (df["sentiment"] == "positive").sum() / total * 100
    neg_pct = (df["sentiment"] == "negative").sum() / total * 100
    neu_pct = (df["sentiment"] == "neutral").sum() / total * 100

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total tweets", f"{total:,}")
    k2.metric("🟢 Positive", f"{pos_pct:.1f}%")
    k3.metric("🟡 Neutral", f"{neu_pct:.1f}%")
    k4.metric("🔴 Negative", f"{neg_pct:.1f}%")

    # ── Sentiment over time ───────────────────────────────────────────────────
    st.subheader("Sentiment over time")
    df["minute"] = pd.to_datetime(df["created_at"]).dt.floor("1min")
    timeline = (
        df.groupby(["minute", "sentiment"]).size().reset_index(name="count")
    )

    fig_time = px.bar(
        timeline,
        x="minute",
        y="count",
        color="sentiment",
        color_discrete_map={"positive": "#1d9e75", "neutral": "#888780", "negative": "#d85a30"},
        title="Tweet sentiment per minute",
        labels={"minute": "Time", "count": "Tweets"},
    )
    fig_time.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_time, use_container_width=True)

    # ── Donut chart ───────────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        sentiment_counts = df["sentiment"].value_counts()
        fig_pie = go.Figure(
            go.Pie(
                labels=sentiment_counts.index,
                values=sentiment_counts.values,
                hole=0.55,
                marker_colors=["#1d9e75", "#888780", "#d85a30"],
            )
        )
        fig_pie.update_layout(
            title="Overall sentiment split",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        st.subheader("Confidence score distribution")
        fig_hist = px.histogram(
            df, x="score", color="sentiment",
            color_discrete_map={"positive": "#1d9e75", "neutral": "#888780", "negative": "#d85a30"},
            nbins=30, title="Model confidence scores",
        )
        fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── Recent tweet feed ─────────────────────────────────────────────────────
    st.subheader("📋 Recent tweets")
    recent = df.sort_values("created_at", ascending=False).head(20)
    for _, row in recent.iterrows():
        icon = sentiment_color(row["sentiment"])
        st.markdown(f"{icon} **[{row['sentiment']}]** {row['text']}")
        st.caption(str(row["created_at"])[:19])
        st.markdown("---")

    # ── Auto-refresh hint ─────────────────────────────────────────────────────
    st.caption("💡 Tip: Set `st.experimental_rerun()` on a timer for true live updates, or use Plotly Dash (see `dash_live.py`).")
