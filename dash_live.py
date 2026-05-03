"""
dash_live.py
=============
Standalone Plotly Dash app for true real-time sentiment updates.
Run alongside app.py: python dash_live.py  (serves on port 8050)
Then embed it in the Streamlit page via st.components.v1.iframe.

Dash auto-refreshes every 5 seconds via dcc.Interval.
"""

from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from modules.sentiment_tracker import load_recent_tweets

app = Dash(__name__)

app.layout = html.Div(
    style={"fontFamily": "sans-serif", "padding": "20px"},
    children=[
        html.H2("⚽ World Cup 2026 — Live Sentiment", style={"marginBottom": "8px"}),
        html.P("Auto-refreshes every 5 seconds", style={"color": "#888"}),
        dcc.Interval(id="interval", interval=5_000, n_intervals=0),

        html.Div(id="kpis", style={"display": "flex", "gap": "20px", "marginBottom": "20px"}),

        dcc.Graph(id="timeline"),
        dcc.Graph(id="pie"),
    ],
)


@app.callback(
    Output("kpis", "children"),
    Output("timeline", "figure"),
    Output("pie", "figure"),
    Input("interval", "n_intervals"),
)
def update(_):
    df = load_recent_tweets(n=300)
    if df.empty:
        empty_fig = go.Figure()
        return [], empty_fig, empty_fig

    total = len(df)
    pos_pct = (df["sentiment"] == "positive").sum() / total * 100
    neg_pct = (df["sentiment"] == "negative").sum() / total * 100

    kpis = [
        html.Div([html.H3(f"{total:,}"), html.P("Tweets")], style={"textAlign": "center"}),
        html.Div([html.H3(f"{pos_pct:.1f}%"), html.P("🟢 Positive")], style={"textAlign": "center"}),
        html.Div([html.H3(f"{neg_pct:.1f}%"), html.P("🔴 Negative")], style={"textAlign": "center"}),
    ]

    df["minute"] = pd.to_datetime(df["created_at"]).dt.floor("1min")
    timeline_data = df.groupby(["minute", "sentiment"]).size().reset_index(name="count")
    fig_time = px.bar(
        timeline_data, x="minute", y="count", color="sentiment",
        color_discrete_map={"positive": "#1d9e75", "neutral": "#888780", "negative": "#d85a30"},
        title="Tweets per minute",
    )
    fig_time.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

    counts = df["sentiment"].value_counts()
    fig_pie = go.Figure(
        go.Pie(
            labels=counts.index, values=counts.values,
            hole=0.5, marker_colors=["#1d9e75", "#888780", "#d85a30"],
        )
    )
    fig_pie.update_layout(title="Sentiment split", paper_bgcolor="rgba(0,0,0,0)")

    return kpis, fig_time, fig_pie


if __name__ == "__main__":
    app.run(debug=True, port=8050)
