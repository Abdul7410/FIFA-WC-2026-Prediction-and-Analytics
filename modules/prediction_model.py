"""
modules/prediction_model.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
import joblib
import warnings

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_PATH = Path(__file__).parent.parent / "models" / "match_predictor.pkl"

TOURNAMENT_WEIGHT = {
    "FIFA World Cup": 5,
    "UEFA Euro": 4,
    "Copa América": 4,
    "African Cup of Nations": 3,
    "UEFA Nations League": 3,
    "Confederations Cup": 3,
    "Friendly": 1,
}


def load_results():
    df = pd.read_csv(DATA_DIR / "results.csv", parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def get_outcome(row):
    if row["home_score"] > row["away_score"]:
        return "home_win"
    elif row["home_score"] < row["away_score"]:
        return "away_win"
    return "draw"


def rolling_win_rate(df, team, before_date, n=20):
    mask = (
        ((df["home_team"] == team) | (df["away_team"] == team))
        & (df["date"] < before_date)
    )
    recent = df[mask].tail(n)
    if recent.empty:
        return 0.5

    wins = sum(
        (r["home_team"] == team and r["home_score"] > r["away_score"])
        or (r["away_team"] == team and r["away_score"] > r["home_score"])
        for _, r in recent.iterrows()
    )
    return wins / len(recent)


def rolling_avg_goals(df, team, before_date, n=20):
    mask = (
        ((df["home_team"] == team) | (df["away_team"] == team))
        & (df["date"] < before_date)
    )
    recent = df[mask].tail(n)
    if recent.empty:
        return 1.2, 1.2

    scored, conceded = [], []
    for _, r in recent.iterrows():
        if r["home_team"] == team:
            scored.append(r["home_score"])
            conceded.append(r["away_score"])
        else:
            scored.append(r["away_score"])
            conceded.append(r["home_score"])

    return np.mean(scored), np.mean(conceded)


def h2h_win_rate(df, home, away, before_date, n=10):
    mask = (
        (
            ((df["home_team"] == home) & (df["away_team"] == away))
            | ((df["home_team"] == away) & (df["away_team"] == home))
        )
        & (df["date"] < before_date)
    )
    recent = df[mask].tail(n)
    if recent.empty:
        return 0.5

    wins = sum(
        (r["home_team"] == home and r["home_score"] > r["away_score"])
        or (r["away_team"] == home and r["away_score"] > r["home_score"])
        for _, r in recent.iterrows()
    )
    return wins / len(recent)


def build_features(df, sample_frac=1.0):
    df = df[df["date"] >= "1990-01-01"].copy()
    df["outcome"] = df.apply(get_outcome, axis=1)
    df["tournament_weight"] = df["tournament"].map(lambda t: TOURNAMENT_WEIGHT.get(t, 2))
    df["neutral"] = df["neutral"].map({"TRUE": 1, "FALSE": 0, True: 1, False: 0})

    full_df = df.copy()
    df = df.sample(frac=sample_frac, random_state=42).sort_values("date")

    rows = []
    for _, row in df.iterrows():
        h, a, d = row["home_team"], row["away_team"], row["date"]

        h_wr = rolling_win_rate(full_df, h, d)
        a_wr = rolling_win_rate(full_df, a, d)

        h_gs, h_gc = rolling_avg_goals(full_df, h, d)
        a_gs, a_gc = rolling_avg_goals(full_df, a, d)

        h2h = h2h_win_rate(full_df, h, a, d)

        rows.append(
            {
                "home_win_rate": h_wr,
                "away_win_rate": a_wr,
                "win_rate_diff": h_wr - a_wr,
                "strength_balance": abs(h_wr - a_wr),

                "home_avg_scored": h_gs,
                "home_avg_conceded": h_gc,
                "away_avg_scored": a_gs,
                "away_avg_conceded": a_gc,

                "goals_diff": h_gs - a_gs,
                "defensive_strength_diff": (h_gc - a_gc),
                "attacking_strength_diff": (h_gs - a_gs),
                "defensive_balance": abs(h_gc - a_gc),

                "h2h_home_win_rate": h2h,
                "tournament_weight": row["tournament_weight"],
                "neutral": row["neutral"],
                "outcome": row["outcome"],
            }
        )

    return pd.DataFrame(rows)


def train(sample_frac=1.0):
    print("Loading data …")
    df = load_results()

    print("Building features …")
    features_df = build_features(df, sample_frac=sample_frac)

    le = LabelEncoder()
    X = features_df.drop(columns=["outcome"])
    y = le.fit_transform(features_df["outcome"])

    X = X.fillna(0)

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 🔥 Handle imbalance
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train, sample_weight=sample_weights)

    preds = model.predict(X_test)

    print(f"\nAccuracy: {accuracy_score(y_test, preds):.3f}")
    print(classification_report(y_test, preds, target_names=le.classes_))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(
        {"model": model, "label_encoder": le, "feature_cols": list(X.columns)},
        MODEL_PATH,
    )

    print(f"Model saved to {MODEL_PATH}")


def predict_match(home_team, away_team, tournament="FIFA World Cup"):
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Run training first")

    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    le = artifact["label_encoder"]

    df = load_results()
    now = df["date"].max()

    h_wr = rolling_win_rate(df, home_team, now)
    a_wr = rolling_win_rate(df, away_team, now)

    h_gs, h_gc = rolling_avg_goals(df, home_team, now)
    a_gs, a_gc = rolling_avg_goals(df, away_team, now)

    h2h = h2h_win_rate(df, home_team, away_team, now)

    X = pd.DataFrame(
        [
            {
                "home_win_rate": h_wr,
                "away_win_rate": a_wr,
                "win_rate_diff": h_wr - a_wr,
                "strength_balance": abs(h_wr - a_wr),

                "home_avg_scored": h_gs,
                "home_avg_conceded": h_gc,
                "away_avg_scored": a_gs,
                "away_avg_conceded": a_gc,

                "goals_diff": h_gs - a_gs,
                "defensive_strength_diff": (h_gc - a_gc),
                "attacking_strength_diff": (h_gs - a_gs),
                "defensive_balance": abs(h_gc - a_gc),

                "h2h_home_win_rate": h2h,
                "tournament_weight": TOURNAMENT_WEIGHT.get(tournament, 3),
                "neutral": 1,
            }
        ]
    )

    X = X.fillna(0)

    proba = model.predict_proba(X)[0]
    return {cls: round(float(p), 3) for cls, p in zip(le.classes_, proba)}


if __name__ == "__main__":
    train(sample_frac=1.0)