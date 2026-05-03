"""pages/player_dashboard.py — Player Performance Dashboard Streamlit page."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from modules.player_analysis import build_player_stats, cluster_players, get_market_value_trend


@st.cache_data(show_spinner="Loading player data…")
def get_clustered_players():
    df = build_player_stats(min_minutes=500)
    return cluster_players(df, n_clusters=5)


def show():
    st.title("📊 Player Performance Dashboard")
    st.markdown(
        "Explore **30,000+ player** stats from Transfermarkt. "
        "Players are clustered into 5 archetypes using K-means on per-90 metrics."
    )

    df = get_clustered_players()

    # ── Filters ───────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    positions = ["All"] + sorted(df["position"].dropna().unique().tolist())
    sel_position = col1.selectbox("Position", positions)
    sel_cluster = col2.selectbox("Archetype", ["All"] + list(df["cluster_label"].unique()))
    min_mv = col3.number_input("Min market value (€M)", min_value=0.0, value=0.0, step=0.5)

    filtered = df.copy()
    if sel_position != "All":
        filtered = filtered[filtered["position"] == sel_position]
    if sel_cluster != "All":
        filtered = filtered[filtered["cluster_label"] == sel_cluster]
    filtered = filtered[filtered["market_value_in_eur"].fillna(0) >= min_mv * 1_000_000]

    st.caption(f"Showing {len(filtered):,} players")

    # ── Tab layout ────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🗺 Style Map", "🏆 Top Performers", "💰 Market Value", "🔍 Player Detail"]
    )

    with tab1:
        st.subheader("Playing style clusters (PCA projection)")
        fig = px.scatter(
            filtered,
            x="pca_x",
            y="pca_y",
            color="cluster_label",
            hover_data=["name", "goals_per90", "assists_per90", "total_minutes"],
            title="Player archetypes in 2D style space",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig.update_traces(marker=dict(size=5, opacity=0.7))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Cluster centroids — what each archetype looks like:**")
        centroid = (
            filtered.groupby("cluster_label")[["goals_per90", "assists_per90", "minutes_per_game", "yellow_per90"]]
            .mean()
            .round(3)
        )
        st.dataframe(centroid, use_container_width=True)

    with tab2:
        st.subheader("Top 20 by goal contribution per 90 mins")
        top20 = filtered.nlargest(20, "goal_contribution_per90")[
            ["name", "position", "total_goals", "total_assists", "goal_contribution_per90", "cluster_label", "current_club_name"]
        ]
        st.dataframe(top20, use_container_width=True)

        fig2 = px.bar(
            top20,
            x="name",
            y="goal_contribution_per90",
            color="cluster_label",
            title="Goal contributions per 90 min (top 20)",
        )
        fig2.update_layout(xaxis_tickangle=-45, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.subheader("Market value distribution")
        mv_df = filtered.dropna(subset=["market_value_in_eur"])
        mv_df = mv_df[mv_df["market_value_in_eur"] > 0]

        fig3 = px.histogram(
            mv_df,
            x="market_value_in_eur",
            nbins=60,
            log_x=True,
            color="position",
            title="Market value distribution (log scale)",
            labels={"market_value_in_eur": "Market value (€)"},
        )
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Goals vs market value")
        fig4 = px.scatter(
            mv_df.sample(min(3000, len(mv_df))),
            x="goals_per90",
            y="market_value_in_eur",
            color="position",
            hover_data=["name"],
            log_y=True,
            title="Goals per 90 vs market value",
            labels={"market_value_in_eur": "Market value (€, log)"},
        )
        fig4.update_traces(marker=dict(size=5, opacity=0.6))
        fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig4, use_container_width=True)

    with tab4:
        st.subheader("Player detail & valuation history")
        player_names = sorted(filtered["name"].dropna().tolist())
        sel_player = st.selectbox("Select a player", player_names)
        player_row = filtered[filtered["name"] == sel_player].iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Goals/90", player_row["goals_per90"])
        c2.metric("Assists/90", player_row["assists_per90"])
        c3.metric("Minutes/game", player_row["minutes_per_game"])
        c4.metric("Archetype", player_row["cluster_label"])

        pid = int(player_row["player_id"])
        val_df = get_market_value_trend(pid)
        if not val_df.empty:
            fig5 = px.line(
                val_df, x="date", y="market_value_in_eur",
                title=f"{sel_player} — market value history",
                labels={"market_value_in_eur": "Value (€)"},
            )
            fig5.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig5, use_container_width=True)
        else:
            st.info("No valuation history available for this player.")
