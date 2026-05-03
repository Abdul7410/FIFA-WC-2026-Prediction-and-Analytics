"""pages/prediction.py — Match Prediction Streamlit page."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from modules.prediction_model import predict_match, load_results, MODEL_PATH


def show():
    st.title("🎯 Match Outcome Predictor")
    st.markdown(
        "Built on **47,915 historical FIFA matches** (1872–2024). "
        "Select two teams and get win/draw/loss probabilities."
    )

    # ── Train model if not yet done ───────────────────────────────────────────
    if not MODEL_PATH.exists():
        with st.spinner("Training model for the first time… (≈2 min)"):
            from modules.prediction_model import train
            train(sample_frac=0.3)
        st.success("Model trained!")

    # ── Team selector ─────────────────────────────────────────────────────────
    df = load_results()
    teams = sorted(set(df["home_team"].unique()) | set(df["away_team"].unique()))
    tournaments = [
        "FIFA World Cup", "UEFA Euro", "Copa América",
        "African Cup of Nations", "UEFA Nations League", "Friendly",
    ]

    col1, col2, col3 = st.columns(3)
    home = col1.selectbox("Home team", teams, index=teams.index("Brazil") if "Brazil" in teams else 0)
    away = col2.selectbox("Away team", teams, index=teams.index("Germany") if "Germany" in teams else 1)
    tournament = col3.selectbox("Tournament", tournaments)

    if st.button("⚽ Predict", type="primary"):
        if home == away:
            st.warning("Please select different teams.")
            return

        with st.spinner("Predicting…"):
            try:
                probs = predict_match(home, away, tournament)
            except FileNotFoundError:
                st.error("Model not found. Click 'Train model' first.")
                return

        # ── Result display ────────────────────────────────────────────────────
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{home} wins", f"{probs.get('home_win', 0)*100:.1f}%")
        c2.metric("Draw", f"{probs.get('draw', 0)*100:.1f}%")
        c3.metric(f"{away} wins", f"{probs.get('away_win', 0)*100:.1f}%")

        # Gauge chart
        labels = [f"{home} win", "Draw", f"{away} win"]
        values = [
            probs.get("home_win", 0),
            probs.get("draw", 0),
            probs.get("away_win", 0),
        ]
        colors = ["#1d9e75", "#888780", "#d85a30"]

        fig = go.Figure(
            go.Bar(
                x=labels,
                y=[v * 100 for v in values],
                marker_color=colors,
                text=[f"{v*100:.1f}%" for v in values],
                textposition="outside",
            )
        )
        fig.update_layout(
            title=f"{home} vs {away} — {tournament}",
            yaxis_title="Probability (%)",
            yaxis_range=[0, 100],
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Historical H2H ────────────────────────────────────────────────────────
    with st.expander("📜 Head-to-head history"):
        mask = (
            ((df["home_team"] == home) & (df["away_team"] == away))
            | ((df["home_team"] == away) & (df["away_team"] == home))
        )
        h2h = df[mask].sort_values("date", ascending=False).head(10)
        if h2h.empty:
            st.info("No previous meetings found.")
        else:
            st.dataframe(
                h2h[["date", "home_team", "home_score", "away_score", "away_team", "tournament"]],
                use_container_width=True,
            )
