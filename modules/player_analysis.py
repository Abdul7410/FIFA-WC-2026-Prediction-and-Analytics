"""
modules/player_analysis.py
===========================
Loads Transfermarkt player/appearance data, engineers per-player stats,
and clusters players by playing style using K-means.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent.parent / "data"

CLUSTER_LABELS = {
    0: "Goal Machine",
    1: "Creative Playmaker",
    2: "Defensive Anchor",
    3: "Box-to-Box Runner",
    4: "Utility Performer",
}

FEATURE_COLS = [
    "goals_per90",
    "assists_per90",
    "minutes_per_game",
    "yellow_per90",
    "red_per90",
    "goal_contribution_per90",
]


def load_players() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "players.csv")


def load_appearances() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "appearances.csv", parse_dates=["date"])


def load_valuations() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "player_valuations.csv", parse_dates=["date"])


def build_player_stats(min_minutes: int = 500) -> pd.DataFrame:
    """
    Aggregate appearance-level data into per-player career stats.
    Filters to players with at least min_minutes total playing time.
    """
    apps = load_appearances()
    players = load_players()

    # Aggregate per player
    agg = (
        apps.groupby("player_id")
        .agg(
            total_goals=("goals", "sum"),
            total_assists=("assists", "sum"),
            total_minutes=("minutes_played", "sum"),
            total_yellow=("yellow_cards", "sum"),
            total_red=("red_cards", "sum"),
            appearances=("appearance_id", "count"),
        )
        .reset_index()
    )

    # Per-90 stats
    agg["per90"] = agg["total_minutes"] / 90
    agg["goals_per90"] = (agg["total_goals"] / agg["per90"]).round(3)
    agg["assists_per90"] = (agg["total_assists"] / agg["per90"]).round(3)
    agg["yellow_per90"] = (agg["total_yellow"] / agg["per90"]).round(3)
    agg["red_per90"] = (agg["total_red"] / agg["per90"]).round(3)
    agg["goal_contribution_per90"] = agg["goals_per90"] + agg["assists_per90"]
    agg["minutes_per_game"] = (agg["total_minutes"] / agg["appearances"]).round(1)

    # Merge player meta
    merged = agg.merge(
        players[
            [
                "player_id",
                "name",
                "position",
                "sub_position",
                "country_of_citizenship",
                "market_value_in_eur",
                "highest_market_value_in_eur",
                "current_club_name",
                "date_of_birth",
            ]
        ],
        on="player_id",
        how="left",
    )

    # Filter
    merged = merged[merged["total_minutes"] >= min_minutes].copy()
    merged = merged.dropna(subset=FEATURE_COLS)
    return merged


def cluster_players(df: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
    """Add cluster labels and PCA coordinates to the player dataframe."""
    X = df[FEATURE_COLS].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df = df.copy()
    df["cluster"] = kmeans.fit_predict(X_scaled)
    df["cluster_label"] = df["cluster"].map(CLUSTER_LABELS)

    # PCA for scatter viz
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    df["pca_x"] = coords[:, 0].round(3)
    df["pca_y"] = coords[:, 1].round(3)

    return df


def get_top_players_by_position(df: pd.DataFrame, position: str, top_n: int = 20) -> pd.DataFrame:
    """Filter and rank players for a given position."""
    filtered = df[df["position"].str.contains(position, case=False, na=False)]
    return filtered.nlargest(top_n, "goal_contribution_per90")


def get_market_value_trend(player_id: int) -> pd.DataFrame:
    """Return valuation history for a single player."""
    vals = load_valuations()
    return vals[vals["player_id"] == player_id].sort_values("date")
