"""
2026 World Cup Data Science Hub
================================
Combined entry point: match prediction, player dashboard, sentiment tracker.
Run with: streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="World Cup 2026 DS Hub",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.title("⚽ World Cup 2026")
st.sidebar.caption("Data Science Hub")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Home",
        "🎯 Match Prediction",
        "📊 Player Dashboard",
        "🔴 Live Sentiment",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Data sources**\n"
    "- FIFA results (1872–2024)\n"
    "- Transfermarkt players\n"
    "- Twitter / X live stream"
)

# ── Page routing ──────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.title("⚽ 2026 FIFA World Cup — Data Science Hub")
    st.markdown(
        """
        This project combines three data science modules into one unified app:

        | Module | Description | Tech |
        |--------|-------------|------|
        | 🎯 Match Prediction | Predict win / draw / loss using historical FIFA data | Pandas · Scikit-learn |
        | 📊 Player Dashboard | Explore Transfermarkt stats, cluster players by style | Plotly · Seaborn · K-means |
        | 🔴 Live Sentiment | Real-time tweet sentiment as goals happen | Tweepy · HuggingFace · Dash |

        **Pick a module from the sidebar to get started.**
        """
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("Historical matches", "47,915", "FIFA dataset")
    col2.metric("Players tracked", "30,000+", "Transfermarkt")
    col3.metric("Sentiment labels", "3 classes", "pos · neu · neg")

elif page == "🎯 Match Prediction":
    from pages.prediction import show
    show()

elif page == "📊 Player Dashboard":
    from pages.player_dashboard import show
    show()

elif page == "🔴 Live Sentiment":
    from pages.sentiment import show
    show()
