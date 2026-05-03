"""
setup_data.py
=============
Copies and renames the uploaded CSV files into the data/ folder
with clean names that the modules expect.

Run once: python setup_data.py
"""

import shutil
from pathlib import Path

# ── Edit this path to wherever your uploaded CSVs live ───────────────────────
UPLOAD_DIR = Path("uploads")   # change to your actual path

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Map: source filename fragment → target name in data/
MAPPINGS = [
    ("results",            "results.csv"),
    ("goalscorers",        "goalscorers.csv"),
    ("shootouts",          "shootouts.csv"),
    ("former_names",       "former_names.csv"),
    ("players",            "players.csv"),
    ("appearances",        "appearances.csv"),
    ("player_valuations",  "player_valuations.csv"),
    ("games",              "games.csv"),
    ("game_lineups",       "game_lineups.csv"),
    ("game_events",        "game_events.csv"),
    ("clubs",              "clubs.csv"),
    ("club_games",         "club_games.csv"),
    ("competitions",       "competitions.csv"),
    ("national_teams",     "national_teams.csv"),
    ("transfers",          "transfers.csv"),
    ("countries",          "countries.csv"),
]

for fragment, target_name in MAPPINGS:
    matches = list(UPLOAD_DIR.glob(f"*{fragment}*"))
    if not matches:
        print(f"  ⚠  Not found: {fragment}")
        continue
    src = matches[0]
    dst = DATA_DIR / target_name
    shutil.copy2(src, dst)
    print(f"  ✓  {src.name} → data/{target_name}")

print("\nDone. Check data/ folder.")
