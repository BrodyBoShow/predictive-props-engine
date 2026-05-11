"""
build_training_data.py
──────────────────────
Pull 3 seasons of player game logs from nba_api and build a clean,
leakage-free training dataset for the XGBoost projection models.

Each output row = one player-game appearance with ONLY pre-game features.

Usage:
    pip install nba_api pandas numpy pyarrow
    python build_training_data.py

Output:
    training_data.parquet   (master dataset — ~200k+ rows, 3 seasons)
    data_cache/             (raw API responses cached to avoid re-fetching)
"""

import os
import sys
import time
import json
import pandas as pd
import numpy as np

try:
    from nba_api.stats.endpoints import leaguegamelog
except ImportError:
    print("Missing dependency — run: pip install nba_api pandas numpy pyarrow")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────
SEASONS      = ["2022-23", "2023-24", "2024-25"]
SEASON_TYPES = ["Regular Season", "Playoffs"]
SLEEP_SEC    = 1.0          # rate-limit pause between API calls
MIN_GP_PRIOR = 5            # drop rows where player has < 5 prior games (L5 invalid)
CACHE_DIR    = "data_cache"
OUTPUT_FILE  = "training_data.parquet"


def _fetch(season: str, season_type: str, retries: int = 3) -> pd.DataFrame:
    """Fetch LeagueGameLog with local parquet caching to avoid re-hitting the API."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    tag   = season_type.replace(" ", "_")
    fpath = os.path.join(CACHE_DIR, f"{season}_{tag}.parquet")
    if os.path.exists(fpath):
        print(f"  cache hit  → {fpath}")
        return pd.read_parquet(fpath)
    for attempt in range(retries):
        try:
            time.sleep(SLEEP_SEC)
            df = leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star=season_type,
                player_or_team_abbreviation="P",
                timeout=90,
            ).get_data_frames()[0]
            df.to_parquet(fpath, index=False)
            print(f"  fetched    → {len(df):,} rows")
            return df
        except Exception as exc:
            wait = 3 * (attempt + 1)
            print(f"  attempt {attempt+1} failed ({exc}), retrying in {wait}s…")
            time.sleep(wait)
    print(f"  FAILED after {retries} attempts — skipping")
    return pd.DataFrame()


def _parse_min(val) -> float:
    """Handle both 'MM:SS' strings and numeric minutes."""
    try:
        if isinstance(val, str) and ":" in val:
            m, s = val.split(":")
            return float(m) + float(s) / 60
        return float(val)
    except Exception:
        return 0.0


def _rolling_prior(series: pd.Series, n: int, min_periods: int = 1) -> pd.Series:
    """Rolling mean of the n games BEFORE the current one (no leakage)."""
    return series.shift(1).rolling(n, min_periods=min_periods).mean()


def _std_prior(series: pd.Series, n: int, min_periods: int = 2) -> pd.Series:
    """Rolling std of the n games BEFORE current."""
    return series.shift(1).rolling(n, min_periods=min_periods).std()


def _expanding_prior(series: pd.Series, min_periods: int = 1) -> pd.Series:
    """Season-to-date mean using only prior games."""
    return series.shift(1).expanding(min_periods=min_periods).mean()


def _per_player(grp: pd.DataFrame) -> pd.DataFrame:
    """Compute all rolling pre-game features for one player's history."""
    grp = grp.sort_values("GAME_DATE").copy()

    # Rest days (capped at 14 — bye weeks, start of season get 7)
    grp["rest_days"] = grp["GAME_DATE"].diff().dt.days.fillna(7).clip(upper=14)

    # L5 rolling averages (prior games only)
    for col, feat in [
        ("PTS",      "l5_pts"),
        ("REB",      "l5_reb"),
        ("AST",      "l5_ast"),
        ("MIN",      "l5_min"),
        ("usg_prox", "l5_usg"),
        ("ts_pct",   "l5_ts"),
    ]:
        grp[feat] = _rolling_prior(grp[col], 5)

    # L10 volatility features
    grp["l10_pts_std"] = _std_prior(grp["PTS"], 10)
    grp["l10_min_std"] = _std_prior(grp["MIN"], 10)

    # Season-to-date expanding means (reset each season via groupby upstream)
    for col, feat in [
        ("PTS", "std_pts"),
        ("REB", "std_reb"),
        ("AST", "std_ast"),
        ("MIN", "std_min"),
    ]:
        grp[feat] = _expanding_prior(grp[col])

    # Games played BEFORE this game in the current season
    grp["gp_prior"] = range(len(grp))  # 0,1,2,… after sort within player-season group

    return grp


def main():
    # ── 1. Fetch all seasons ──────────────────────────────────────────────────
    frames = []
    for season in SEASONS:
        for stype in SEASON_TYPES:
            print(f"Fetching {season} / {stype}…")
            df = _fetch(season, stype)
            if not df.empty:
                df["season"]      = season
                df["season_type"] = stype
                frames.append(df)

    if not frames:
        print("No data fetched — exiting.")
        return

    raw = pd.concat(frames, ignore_index=True)
    print(f"\nRaw rows: {len(raw):,}")

    # ── 2. Parse & clean ──────────────────────────────────────────────────────
    raw["GAME_DATE"] = pd.to_datetime(raw["GAME_DATE"], errors="coerce")
    raw = raw.dropna(subset=["GAME_DATE", "PLAYER_ID"]).copy()

    num_cols = ["PTS", "REB", "AST", "MIN", "FGA", "FTA", "TOV",
                "OREB", "DREB", "FGM", "FG3M", "FG3A", "FTM",
                "STL", "BLK", "PLUS_MINUS"]
    for c in num_cols:
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors="coerce").fillna(0)

    raw["MIN"] = raw["MIN"].apply(_parse_min)

    # Venue
    raw["is_home"] = raw["MATCHUP"].str.contains(r"vs\.", na=False).astype(int)

    # Per-game TS% and usage proxy
    denom = 2 * (raw["FGA"] + 0.44 * raw["FTA"])
    raw["ts_pct"]   = np.where(denom > 0, raw["PTS"] / denom, np.nan)
    raw["usg_prox"] = raw["FGA"] + 0.44 * raw["FTA"] + raw["TOV"]

    # ── 3. Build opponent defensive efficiency proxy ───────────────────────────
    # Aggregate player logs to team totals per game
    team_game = (
        raw.groupby(["GAME_ID", "TEAM_ABBREVIATION", "GAME_DATE"])
        .agg(team_pts=("PTS", "sum"), team_fga=("FGA", "sum"))
        .reset_index()
    )

    # Self-join: each team row gets the opponent's PTS (= pts they allowed)
    tg = team_game.merge(
        team_game[["GAME_ID", "TEAM_ABBREVIATION", "team_pts", "team_fga"]],
        on="GAME_ID", suffixes=("", "_opp")
    )
    tg = tg[tg["TEAM_ABBREVIATION"] != tg["TEAM_ABBREVIATION_opp"]].copy()
    tg = tg.rename(columns={"team_pts_opp": "pts_allowed", "team_fga_opp": "opp_fga"})
    tg = tg.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"])

    # Rolling 10-game defensive and pace metrics (prior games only)
    tg["opp_def_roll10"]  = tg.groupby("TEAM_ABBREVIATION")["pts_allowed"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=3).mean()
    )
    tg["opp_pace_roll10"] = tg.groupby("TEAM_ABBREVIATION")["team_fga"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=3).mean()
    )

    # Lookup table: game_id + player team → rolling opponent defensive stats
    opp_lookup = tg[["GAME_ID", "TEAM_ABBREVIATION", "opp_def_roll10", "opp_pace_roll10"]].copy()

    # ── 4. Per-player rolling features ────────────────────────────────────────
    # Group by player+season so gp_prior resets each season
    print("Computing per-player rolling features…")
    raw = (
        raw.groupby(["PLAYER_ID", "season"], group_keys=False)
        .apply(_per_player)
        .reset_index(drop=True)
    )

    # ── 5. Join opponent context ───────────────────────────────────────────────
    raw = raw.merge(opp_lookup, on=["GAME_ID", "TEAM_ABBREVIATION"], how="left")

    # ── 6. Define targets ─────────────────────────────────────────────────────
    raw["target_pts"] = raw["PTS"]
    raw["target_reb"] = raw["REB"]
    raw["target_ast"] = raw["AST"]

    # ── 7. Select output columns & filter ─────────────────────────────────────
    keep = [
        # Identifiers
        "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE",
        "season", "season_type",
        # Pre-game features
        "is_home", "rest_days",
        "l5_pts", "l5_reb", "l5_ast", "l5_min", "l5_usg", "l5_ts",
        "l10_pts_std", "l10_min_std",
        "std_pts", "std_reb", "std_ast", "std_min",
        "gp_prior",
        "opp_def_roll10", "opp_pace_roll10",
        # Targets
        "target_pts", "target_reb", "target_ast",
        # Keep actual MIN for analysis — NOT a training feature (future leakage)
        "MIN",
    ]
    out = raw[[c for c in keep if c in raw.columns]].copy()

    # Drop rows with insufficient prior history
    out = out[out["gp_prior"] >= MIN_GP_PRIOR].reset_index(drop=True)

    print(f"\nFinal rows: {len(out):,}")
    print(f"Seasons:    {sorted(out['season'].unique())}")
    print(f"Players:    {out['PLAYER_ID'].nunique():,}")
    print(f"\nNaN counts in key features:")
    feat_cols = ["l5_pts", "l5_reb", "l5_ast", "l5_min", "opp_def_roll10", "opp_pace_roll10"]
    print(out[feat_cols].isna().sum().to_string())

    out.to_parquet(OUTPUT_FILE, index=False)
    print(f"\nSaved → {OUTPUT_FILE}")

    # Save feature column list and median imputation values for inference
    feature_cols = [
        "l5_pts", "l5_reb", "l5_ast", "l5_min", "l5_usg", "l5_ts",
        "l10_pts_std", "l10_min_std",
        "std_pts", "std_reb", "std_ast", "std_min",
        "gp_prior", "is_home", "rest_days",
        "opp_def_roll10", "opp_pace_roll10",
    ]
    medians = {c: float(out[c].median()) for c in feature_cols if c in out.columns}
    meta = {"feature_cols": feature_cols, "medians": medians}
    with open("model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved → model_meta.json")


if __name__ == "__main__":
    main()
