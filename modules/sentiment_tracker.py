"""
modules/sentiment_tracker.py
=============================
Streams tweets during live World Cup matches, runs HuggingFace
sentiment inference, and stores results for live dashboard updates.

Setup:
  1. Create a Twitter Developer account → developer.twitter.com
  2. Generate Bearer Token, API Key/Secret, Access Token/Secret
  3. Add them to .env (see .env.example)

HuggingFace model used: cardiffnlp/twitter-roberta-base-sentiment-latest
  - Label 0 = Negative
  - Label 1 = Neutral
  - Label 2 = Positive
"""

import os
import json
import time
import sqlite3
from datetime import datetime
from pathlib import Path
from collections import deque
from typing import Optional

import tweepy
import pandas as pd
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()

DB_PATH = Path(__file__).parent.parent / "data" / "sentiment.db"
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
LABEL_MAP = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}

# In-memory ring buffer for the live Dash dashboard (last 500 tweets)
_live_buffer: deque = deque(maxlen=500)


# ── Database helpers ──────────────────────────────────────────────────────────

def init_db():
    """Create the SQLite table if it doesn't exist."""
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tweets (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at  TEXT,
            text        TEXT,
            sentiment   TEXT,
            score       REAL,
            query       TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def insert_tweet(created_at: str, text: str, sentiment: str, score: float, query: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO tweets (created_at, text, sentiment, score, query) VALUES (?,?,?,?,?)",
        (created_at, text, sentiment, score, query),
    )
    conn.commit()
    conn.close()


def load_recent_tweets(n: int = 200) -> pd.DataFrame:
    """Load the most recent n rows from the DB."""
    if not DB_PATH.exists():
        return pd.DataFrame(columns=["created_at", "text", "sentiment", "score", "query"])
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        f"SELECT * FROM tweets ORDER BY id DESC LIMIT {n}", conn, parse_dates=["created_at"]
    )
    conn.close()
    return df.sort_values("created_at")


# ── Sentiment pipeline ────────────────────────────────────────────────────────

_sentiment_pipe = None


def get_sentiment_pipeline():
    global _sentiment_pipe
    if _sentiment_pipe is None:
        _sentiment_pipe = pipeline(
            "sentiment-analysis",
            model=SENTIMENT_MODEL,
            tokenizer=SENTIMENT_MODEL,
            return_all_scores=False,
            truncation=True,
            max_length=128,
        )
    return _sentiment_pipe


def analyze(text: str) -> tuple[str, float]:
    """Return (label, confidence_score) for a tweet text."""
    pipe = get_sentiment_pipeline()
    result = pipe(text)[0]
    label = LABEL_MAP.get(result["label"], result["label"])
    return label, round(result["score"], 4)


# ── Twitter streaming ─────────────────────────────────────────────────────────

class WorldCupStreamListener(tweepy.StreamingClient):
    """
    Extends Tweepy's v2 streaming client.
    Analyzes each tweet and writes to SQLite + live buffer.
    """

    def __init__(self, bearer_token: str, query: str):
        super().__init__(bearer_token)
        self.query = query
        init_db()

    def on_tweet(self, tweet):
        text = tweet.text
        if text.startswith("RT "):          # skip retweets
            return
        try:
            sentiment, score = analyze(text)
        except Exception:
            return

        ts = datetime.utcnow().isoformat()
        record = {
            "created_at": ts,
            "text": text[:280],
            "sentiment": sentiment,
            "score": score,
            "query": self.query,
        }
        _live_buffer.append(record)
        insert_tweet(ts, text[:280], sentiment, score, self.query)

    def on_errors(self, errors):
        print("Stream error:", errors)


def start_stream(query: str, max_results: int = 0):
    """
    Start a filtered stream. Blocks until interrupted.
    query: e.g. '#WorldCup2026 OR #FIFAWorldCup lang:en -is:retweet'
    max_results: 0 = stream forever.
    """
    bearer = os.getenv("TWITTER_BEARER_TOKEN")
    if not bearer:
        raise EnvironmentError(
            "TWITTER_BEARER_TOKEN not set. Copy .env.example → .env and fill in your keys."
        )

    client = WorldCupStreamListener(bearer_token=bearer, query=query)

    # Clear old rules
    existing = client.get_rules()
    if existing.data:
        client.delete_rules([r.id for r in existing.data])

    client.add_rules(tweepy.StreamRule(query))
    client.filter(tweet_fields=["created_at", "text"])


def get_live_buffer_df() -> pd.DataFrame:
    """Return the in-memory ring buffer as a DataFrame (for Dash callbacks)."""
    if not _live_buffer:
        return pd.DataFrame(columns=["created_at", "text", "sentiment", "score"])
    return pd.DataFrame(list(_live_buffer))


# ── Offline demo mode (no Twitter API needed) ─────────────────────────────────

DEMO_TWEETS = [
    ("GOOOAL! Mbappe scores! France are unbelievable tonight!", "positive"),
    ("VAR is ruining this tournament. Ridiculous decision.", "negative"),
    ("Watching the World Cup with the family. Great atmosphere.", "positive"),
    ("0-0 after 70 mins. This game needs something special.", "neutral"),
    ("That save was absolutely world class. Incredible keeper.", "positive"),
    ("Penalty? No way was that a penalty. Terrible referee.", "negative"),
    ("Half time. Both teams evenly matched so far.", "neutral"),
    ("The energy in this stadium is absolutely electric!", "positive"),
    ("Injured player down. Hope it's not serious.", "neutral"),
    ("What a finish! Top corner. No chance for the keeper.", "positive"),
]


def run_demo_mode(n_rounds: int = 10, delay: float = 0.3):
    """
    Simulate live tweet stream without API access.
    Useful for local development and demos.
    """
    init_db()
    import random
    pipe = get_sentiment_pipeline()

    for i in range(n_rounds):
        text, _ = random.choice(DEMO_TWEETS)
        sentiment, score = analyze(text)
        ts = datetime.utcnow().isoformat()
        record = {
            "created_at": ts,
            "text": text,
            "sentiment": sentiment,
            "score": score,
            "query": "DEMO",
        }
        _live_buffer.append(record)
        insert_tweet(ts, text, sentiment, score, "DEMO")
        print(f"[{sentiment.upper():8}] {text[:60]}")
        time.sleep(delay)

    print(f"\nDemo complete. {n_rounds} simulated tweets stored.")


if __name__ == "__main__":
    run_demo_mode()
