# ⚽ 2026 FIFA World Cup — Data Science Hub

A combined, production-ready data science project uniting three modules into a single Streamlit app.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## 🗂 Project structure

```
worldcup2026/
├── app.py                  ← Streamlit entry point (navigation)
├── dash_live.py            ← Optional: standalone Dash live-refresh dashboard
├── setup_data.py           ← One-time script to rename & place CSV files
├── requirements.txt
├── .env.example            ← Twitter API key template
├── .gitignore
├── .streamlit/
│   └── config.toml         ← Dark theme config
├── data/                   ← CSVs live here (gitignored)
│   ├── results.csv
│   ├── players.csv
│   ├── appearances.csv
│   ├── player_valuations.csv
│   ├── goalscorers.csv
│   └── sentiment.db        ← SQLite created at runtime
├── modules/
│   ├── prediction_model.py ← Module 1: RandomForest match predictor
│   ├── player_analysis.py  ← Module 2: K-means player clustering
│   └── sentiment_tracker.py← Module 3: Twitter + HuggingFace sentiment
├── pages/
│   ├── prediction.py       ← Streamlit UI for Module 1
│   ├── player_dashboard.py ← Streamlit UI for Module 2
│   └── sentiment.py        ← Streamlit UI for Module 3
└── models/                 ← Trained model saved here (gitignored)
    └── match_predictor.pkl
```

---

## 🚀 Phase-by-phase setup

### Phase 1 — Environment

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/worldcup2026-ds.git
cd worldcup2026-ds

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Phase 2 — Data

```bash
# Place all 16 CSV files in an 'uploads/' folder, then:
python setup_data.py

# Verify:
ls data/
# Should show: results.csv  players.csv  appearances.csv  …
```

**Datasets used:**

| File | Module | Description |
|------|--------|-------------|
| `results.csv` | Prediction | 47,915 international match results (1872–2024) |
| `goalscorers.csv` | Prediction | Per-match goalscorer details |
| `players.csv` | Dashboard | 30k+ player profiles from Transfermarkt |
| `appearances.csv` | Dashboard | Per-game stats per player |
| `player_valuations.csv` | Dashboard | Market value history |
| `clubs.csv` | Dashboard | Club metadata |
| `national_teams.csv` | Prediction | National team info |

### Phase 3 — Train the prediction model

```bash
python -m modules.prediction_model
# Trains on 30% of data by default (~2 min on a laptop)
# For full dataset: edit sample_frac=1.0 in the script
```

### Phase 4 — Twitter API (optional — Module 3)

1. Go to [developer.twitter.com](https://developer.twitter.com) → create a project → get Bearer Token
2. `cp .env.example .env`
3. Paste your Bearer Token in `.env`

**No Twitter API?** → Use **Demo Mode** inside the app (simulates live tweets).

### Phase 5 — Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501)

For the real-time Dash dashboard (Module 3 live mode), in a second terminal:

```bash
python dash_live.py
```

Open [http://localhost:8050](http://localhost:8050)

---

## 📦 Module breakdown

### 🎯 Module 1 — Match Prediction

**File:** `modules/prediction_model.py`

Features engineered per match:
- Rolling win rate (last 20 matches) for home and away team
- Head-to-head win rate (last 10 meetings)
- Average goals scored / conceded per team
- Tournament weight (World Cup = 5, Friendly = 1)
- Neutral venue flag

Model: `RandomForestClassifier` with 200 trees.
Output: `{home_win: p, draw: p, away_win: p}`

### 📊 Module 2 — Player Dashboard

**File:** `modules/player_analysis.py`

Stats computed per player (per-90-minutes):
- Goals, assists, goal contributions
- Yellow / red cards
- Minutes per game

Clustering: K-means (k=5) on standardised per-90 stats → 5 archetypes:
`Goal Machine`, `Creative Playmaker`, `Defensive Anchor`, `Box-to-Box Runner`, `Utility Performer`

Visualisations: PCA scatter, top-performer bar, market value histogram, valuation history line.

### 🔴 Module 3 — Sentiment Tracker

**File:** `modules/sentiment_tracker.py`

- **Streaming:** Tweepy v4 filtered stream on custom query
- **Model:** [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
- **Storage:** SQLite for persistence, in-memory deque for live updates
- **Labels:** `positive`, `neutral`, `negative`
- **Demo mode:** Simulates 20 tweets locally without any API key

---

## ☁️ Deploy to Streamlit Community Cloud

```bash
# 1. Push to GitHub (see GitHub section below)

# 2. Go to share.streamlit.io → New app
# 3. Connect your GitHub repo
# 4. Set:
#      Main file path: app.py
# 5. Add secrets (Settings → Secrets):
#      TWITTER_BEARER_TOKEN = "..."
# 6. Click Deploy
```

> **Data files are gitignored.** For Streamlit Cloud, either:
> - Add a `@st.cache_data` function that downloads from Kaggle via `kaggle datasets download`
> - Or upload a trimmed sample to the repo (< 25 MB per file)

---

## 📤 Upload to GitHub — step by step

```bash
# Inside the project folder:

# 1. Initialise git
git init

# 2. Stage everything (gitignore will exclude .env, data/, models/)
git add .

# 3. First commit
git commit -m "feat: initial commit — World Cup 2026 DS Hub"

# 4. Create repo on GitHub (github.com → New repository)
#    Name: worldcup2026-ds   Visibility: Public   No README (we have one)

# 5. Link and push
git remote add origin https://github.com/YOUR_USERNAME/worldcup2026-ds.git
git branch -M main
git push -u origin main
```

**For large data files** (results.csv is 3.6 MB — fine; appearances.csv is 140 MB — too large):

```bash
# Option A: Git Large File Storage
git lfs install
git lfs track "data/*.csv"
git add .gitattributes
git add data/results.csv data/players.csv   # only the small ones

# Option B: Download in-app using Kaggle API
# Add kaggle.json to .streamlit/secrets.toml and fetch on first run
```

---

## 📚 Resources

| Topic | Link |
|-------|------|
| HuggingFace sentiment guide | https://huggingface.co/blog/sentiment-analysis-python |
| Twitter / X API docs | https://developer.twitter.com/en/docs |
| Kaggle data visualisation | https://kaggle.com/learn/data-visualization |
| Kaggle project walkthroughs | https://kaggle.com/learn |

---

## 🤝 Contributing

PRs welcome. Open an issue for bugs or feature ideas.

## 📄 Licence

MIT
