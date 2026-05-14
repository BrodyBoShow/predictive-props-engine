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

# Force UTF-8 output so Windows cp1252 console doesn't choke on Unicode in print/errors
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    from nba_api.stats.endpoints import leaguegamelog, leaguedashptstats, leaguedashplayerstats
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

# Maps each training season to the prior season whose tracking stats we fetch once.
# Zero leakage: a 2023-24 game row uses only 2022-23 full-season tracking.
SEASON_PRIOR = {
    "2022-23": "2021-22",
    "2023-24": "2022-23",
    "2024-25": "2023-24",
}
_TRACKING_MEASURE_TYPES = ["Drives", "PullUpShot", "CatchShoot", "Passing"]

_LEAGUE_AVG_TS_TRAIN        = 0.559   # 2024-25 NBA avg TS% — xPPS fallback baseline
_LEAGUE_AVG_DRIVE_FG_PCT    = 0.477   # FGA-weighted drive FG%      (matches server.py)
_LEAGUE_AVG_PULLUP_EFG_PCT  = 0.451   # FGA-weighted pull-up EFG%   (matches server.py)
_LEAGUE_AVG_CS_EFG_PCT      = 0.531   # FGA-weighted catch-&-shoot EFG% (matches server.py)


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


def _fetch_tracking_season(season: str, measure_type: str, retries: int = 3) -> pd.DataFrame:
    """
    Fetch full-season LeagueDashPtStats for one measure type, with caching.
    Only Regular Season — full-season archetypes are more stable than small playoff samples.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    fpath = os.path.join(CACHE_DIR, f"tracking_{season}_{measure_type}.parquet")
    if os.path.exists(fpath):
        print(f"  cache hit  → {fpath}")
        return pd.read_parquet(fpath)
    for attempt in range(retries):
        try:
            time.sleep(SLEEP_SEC)
            df = leaguedashptstats.LeagueDashPtStats(
                season=season,
                season_type_all_star="Regular Season",
                per_mode_simple="PerGame",
                pt_measure_type=measure_type,
                player_or_team="Player",
                timeout=90,
            ).get_data_frames()[0]
            df.to_parquet(fpath, index=False)
            print(f"  fetched tracking {season} / {measure_type}: {len(df)} players")
            return df
        except Exception as exc:
            wait = 3 * (attempt + 1)
            print(f"  attempt {attempt+1} failed ({exc}), retrying in {wait}s…")
            time.sleep(wait)
    print(f"  FAILED tracking {season} / {measure_type}")
    return pd.DataFrame()


def _fetch_scoring_season(season: str, retries: int = 3) -> pd.DataFrame:
    """Fetch full-season LeagueDashPlayerStats Scoring breakdown per player, with caching."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    fpath = os.path.join(CACHE_DIR, f"scoring_{season}.parquet")
    if os.path.exists(fpath):
        print(f"  cache hit  → {fpath}")
        return pd.read_parquet(fpath)
    for attempt in range(retries):
        try:
            time.sleep(SLEEP_SEC)
            df = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season,
                season_type_all_star="Regular Season",
                per_mode_simple="PerGame",
                measure_type_detailed_defense="Scoring",
                timeout=90,
            ).get_data_frames()[0]
            df.to_parquet(fpath, index=False)
            print(f"  fetched scoring {season}: {len(df)} players")
            return df
        except Exception as exc:
            wait = 3 * (attempt + 1)
            print(f"  attempt {attempt+1} failed ({exc}), retrying in {wait}s…")
            time.sleep(wait)
    print(f"  FAILED scoring {season}")
    return pd.DataFrame()


def _build_tracking_prior_lookup() -> dict:
    """
    Fetch prior-season full-season tracking for every training season and build
    a two-level lookup:  tracking_prior[prior_season][player_id] = {xPPS_base, potentialAst, drivePg}

    Uses Regular Season final splits only — prior-season archetypes are highly stable YoY
    (shot diet / passing role) and this avoids daily scraping / rate-limit issues.
    """
    prior_seasons = sorted(set(SEASON_PRIOR.values()))  # ["2021-22", "2022-23", "2023-24"]
    tracking_prior = {}

    for prior_season in prior_seasons:
        print(f"\nFetching prior-season tracking for {prior_season}…")
        drives_df  = _fetch_tracking_season(prior_season, "Drives")
        pullup_df  = _fetch_tracking_season(prior_season, "PullUpShot")
        cs_df      = _fetch_tracking_season(prior_season, "CatchShoot")
        passing_df = _fetch_tracking_season(prior_season, "Passing")
        scoring_df = _fetch_scoring_season(prior_season)

        # Index each measure by PLAYER_ID
        def _idx(df):
            if df.empty or "PLAYER_ID" not in df.columns:
                return {}
            return df.set_index("PLAYER_ID").to_dict("index")

        d_idx  = _idx(drives_df)
        pu_idx = _idx(pullup_df)
        cs_idx = _idx(cs_df)
        pa_idx = _idx(passing_df)
        sc_idx = _idx(scoring_df)

        all_pids = set(d_idx) | set(pu_idx) | set(cs_idx) | set(pa_idx) | set(sc_idx)
        lookup = {}

        for pid in all_pids:
            d  = d_idx.get(pid,  {})
            pu = pu_idx.get(pid, {})
            c  = cs_idx.get(pid, {})
            pa = pa_idx.get(pid, {})
            sc = sc_idx.get(pid, {})

            d_fga  = float(d.get("DRIVE_FGA",          0) or 0)
            pu_fga = float(pu.get("PULL_UP_FGA",        0) or 0)
            cs_fga = float(c.get("CATCH_SHOOT_FGA",     0) or 0)
            t_fga  = d_fga + pu_fga + cs_fga

            if t_fga > 0:
                # Fall back to league-average zone efficiency when API returns 0
                # (sparse sub-category or data quality gap) to prevent xPPS collapsing
                # toward 0 and producing a spurious negative efficiency_delta.
                d_eff  = float(d.get("DRIVE_FG_PCT",       0) or 0) or _LEAGUE_AVG_DRIVE_FG_PCT
                pu_eff = float(pu.get("PULL_UP_EFG_PCT",   0) or 0) or _LEAGUE_AVG_PULLUP_EFG_PCT
                cs_eff = float(c.get("CATCH_SHOOT_EFG_PCT",0) or 0) or _LEAGUE_AVG_CS_EFG_PCT
                xPPS = 2.0 * (
                    (d_fga  / t_fga) * d_eff +
                    (pu_fga / t_fga) * pu_eff +
                    (cs_fga / t_fga) * cs_eff
                )
            else:
                xPPS = None

            potential_ast = float(pa.get("POTENTIAL_AST", 0) or 0) if pa else 0.0
            drive_pg      = d_fga  # drives per game ≈ drive FGA per game (best proxy available)

            # Scoring breakdown — pct of points from paint / 3s (0–1 decimal from API)
            pct_pts_paint = float(sc.get("PCT_PTS_PAINT", 0) or 0)
            pct_pts_3pt   = float(sc.get("PCT_PTS_3PT",   0) or 0)

            lookup[int(pid)] = {
                "xPPS_base":    xPPS,
                "potentialAst": potential_ast,
                "drivePg":      drive_pg,
                "pctPtsPaint":  pct_pts_paint,
                "pctPts3pt":    pct_pts_3pt,
            }

        tracking_prior[prior_season] = lookup
        print(f"  Built tracking lookup for {prior_season}: {len(lookup)} players")

    return tracking_prior


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


def _ewma_prior(series: pd.Series, halflife: float = 3.0, min_periods: int = 3) -> pd.Series:
    """
    Exponentially weighted mean of prior games.
    halflife=3 means a game 3 games ago gets 50% the weight of the most recent game.
    This captures hot/cold streaks better than flat L5/L10 windows.
    """
    return series.shift(1).ewm(halflife=halflife, min_periods=min_periods).mean()


def _per_player(grp: pd.DataFrame) -> pd.DataFrame:
    """Compute all rolling pre-game features for one player's history."""
    grp = grp.sort_values("GAME_DATE").copy()

    # Rest days (capped at 14 — bye weeks, start of season get 7)
    grp["rest_days"]      = grp["GAME_DATE"].diff().dt.days.fillna(7).clip(upper=14)
    grp["is_b2b"]         = (grp["rest_days"] == 0).astype(int)
    grp["is_well_rested"] = (grp["rest_days"] >= 3).astype(int)

    # L5 rolling averages (prior games only — flat window)
    for col, feat in [
        ("PTS",    "l5_pts"),
        ("REB",    "l5_reb"),
        ("AST",    "l5_ast"),
        ("MIN",    "l5_min"),
        ("ts_pct", "l5_ts"),
    ]:
        grp[feat] = _rolling_prior(grp[col], 5)

    # EWMA recency features (halflife=3 games — recent form weighted 2× more than 3-game-old)
    # Captures hot/cold streaks missed by flat L5 averages.
    for col, feat in [
        ("PTS", "ewma_pts"),
        ("REB", "ewma_reb"),
        ("AST", "ewma_ast"),
        ("MIN", "ewma_min"),
    ]:
        grp[feat] = _ewma_prior(grp[col], halflife=3.0, min_periods=3)

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

    # USG proxy (numerator of USG% formula) — used for inactive_usg_pool computation
    if "FGA" in grp.columns and "FTA" in grp.columns and "TOV" in grp.columns:
        grp["usg_proxy_raw"] = grp["FGA"] + 0.44 * grp["FTA"] + grp["TOV"]
        grp["l5_usg_proxy"]  = _rolling_prior(grp["usg_proxy_raw"], 5, min_periods=3)
    else:
        grp["l5_usg_proxy"] = np.nan

    # Games played BEFORE this game in the current season (resets each season)
    grp["gp_prior"] = grp.groupby("season").cumcount()

    return grp


def _compute_inactive_pool(
    df: pd.DataFrame,
    value_lookup: dict,   # (player_id, game_date) → scalar  OR  (player_id, season) → scalar
    col_name: str,
    key_mode: str = "date",   # "date" or "season"
) -> pd.Series:
    """
    Generic inactive-teammate pool computation.

    For each player-game, sums `value_lookup[key]` for every teammate who was
    expected to play but did NOT appear (0 MIN or absent from that game's log).

    key_mode="date":   key = (player_id, game_date)   — rolling box-score values
    key_mode="season": key = (player_id, season)       — prior-season tracking values
    """
    label = col_name
    print(f"  Computing {label}…")

    played_df = df[df["MIN"] > 0][["GAME_ID", "GAME_DATE", "TEAM_ABBREVIATION", "PLAYER_ID"]].copy()
    game_team_active = (
        played_df.groupby(["GAME_ID", "TEAM_ABBREVIATION"])["PLAYER_ID"]
        .apply(set)
    )

    team_date_active = (
        played_df.groupby(["TEAM_ABBREVIATION", "GAME_DATE"])["PLAYER_ID"]
        .apply(set)
        .reset_index()
        .rename(columns={"PLAYER_ID": "active_set"})
    )

    roster_lookup = {}
    for team, grp in team_date_active.groupby("TEAM_ABBREVIATION"):
        grp = grp.sort_values("GAME_DATE").reset_index(drop=True)
        for i in range(len(grp)):
            prior_sets = grp.iloc[max(0, i - 10):i]["active_set"]
            roster = frozenset().union(*prior_sets) if len(prior_sets) > 0 else frozenset()
            roster_lookup[(team, grp.at[i, "GAME_DATE"])] = roster

    teams      = df["TEAM_ABBREVIATION"].values
    game_ids   = df["GAME_ID"].values
    player_ids = df["PLAYER_ID"].values
    dates      = df["GAME_DATE"].values
    seasons    = df["season"].values

    pool_values = np.zeros(len(df), dtype=np.float32)
    for i in range(len(df)):
        key_team = (teams[i], dates[i])
        expected = roster_lookup.get(key_team, frozenset())
        if not expected:
            continue
        active   = game_team_active.get((game_ids[i], teams[i]), set())
        inactive = expected - active - {player_ids[i]}
        for pid in inactive:
            if key_mode == "date":
                val = value_lookup.get((int(pid), dates[i]), 0.0) or 0.0
            else:
                val = value_lookup.get((int(pid), seasons[i]), 0.0) or 0.0
            pool_values[i] += float(val)

    return pd.Series(pool_values, index=df.index, name=col_name)


def main():
    # ── 0. Fetch prior-season tracking archetype lookups ──────────────────────
    # One full-season fetch per prior season — no daily scraping, no rate limits.
    tracking_prior = _build_tracking_prior_lookup()

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

    # Per-game TS%
    denom = 2 * (raw["FGA"] + 0.44 * raw["FTA"])
    raw["ts_pct"] = np.where(denom > 0, raw["PTS"] / denom, np.nan)

    # ── 3. Build opponent defensive efficiency proxy ───────────────────────────
    agg_cols = {"team_pts": ("PTS", "sum"), "team_fga": ("FGA", "sum")}
    if "FG3A" in raw.columns:
        agg_cols["team_fg3a"] = ("FG3A", "sum")
    if "FGM" in raw.columns:
        agg_cols["team_fgm"] = ("FGM", "sum")
    team_game = raw.groupby(["GAME_ID", "TEAM_ABBREVIATION", "GAME_DATE"]).agg(**agg_cols).reset_index()

    join_cols = ["GAME_ID", "TEAM_ABBREVIATION", "team_pts", "team_fga"]
    if "team_fg3a" in team_game.columns:
        join_cols.append("team_fg3a")
    if "team_fgm" in team_game.columns:
        join_cols.append("team_fgm")
    tg = team_game.merge(team_game[join_cols], on="GAME_ID", suffixes=("", "_opp"))
    tg = tg[tg["TEAM_ABBREVIATION"] != tg["TEAM_ABBREVIATION_opp"]].copy()
    tg = tg.rename(columns={"team_pts_opp": "pts_allowed", "team_fga_opp": "opp_fga"})
    tg = tg.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"])

    tg["opp_def_roll10"]  = tg.groupby("TEAM_ABBREVIATION")["pts_allowed"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=3).mean()
    )
    tg["opp_pace_roll10"] = tg.groupby("TEAM_ABBREVIATION")["team_fga"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=3).mean()
    )

    _LEAGUE_AVG_FG3A_RATE = 0.37
    _LEAGUE_AVG_FG_PCT    = 0.470
    if "team_fg3a_opp" in tg.columns:
        tg["opp_fg3a_rate"] = tg["team_fg3a_opp"] / tg["opp_fga"].replace(0, np.nan)
        tg["fg3_vs_avg"] = tg.groupby("TEAM_ABBREVIATION")["opp_fg3a_rate"].transform(
            lambda x: x.shift(1).rolling(10, min_periods=3).mean()
        ) - _LEAGUE_AVG_FG3A_RATE
    else:
        tg["fg3_vs_avg"] = np.nan

    if "team_fgm_opp" in tg.columns:
        tg["opp_fgpct"] = tg["team_fgm_opp"] / tg["opp_fga"].replace(0, np.nan)
        tg["rim_vs_avg"] = tg.groupby("TEAM_ABBREVIATION")["opp_fgpct"].transform(
            lambda x: x.shift(1).rolling(10, min_periods=3).mean()
        ) - _LEAGUE_AVG_FG_PCT
    else:
        tg["rim_vs_avg"] = np.nan

    opp_lookup_cols = ["GAME_ID", "TEAM_ABBREVIATION", "opp_def_roll10", "opp_pace_roll10",
                       "fg3_vs_avg", "rim_vs_avg"]
    opp_lookup = tg[[c for c in opp_lookup_cols if c in tg.columns]].copy()

    # ── 4. Per-player rolling features ────────────────────────────────────────
    print("Computing per-player rolling features…")
    raw = raw.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)
    pieces = [_per_player(grp) for _, grp in raw.groupby("PLAYER_ID")]
    raw = pd.concat(pieces, ignore_index=True)

    # ── 5. Join opponent context ───────────────────────────────────────────────
    raw = raw.merge(opp_lookup, on=["GAME_ID", "TEAM_ABBREVIATION"], how="left")

    # ── 5b. Map prior-season tracking archetypes ──────────────────────────────
    # For each game row, look up the player's prior-season tracking stats.
    # Zero leakage: 2023-24 game → 2022-23 final splits.
    print("  Mapping prior-season tracking archetypes…")
    raw["PLAYER_ID_int"] = raw["PLAYER_ID"].astype(int)

    def _get_prior_tracking(row, field):
        prior_season = SEASON_PRIOR.get(row["season"])
        if not prior_season:
            return np.nan
        lookup = tracking_prior.get(prior_season, {})
        entry  = lookup.get(row["PLAYER_ID_int"])
        if entry is None:
            return np.nan
        return entry.get(field)

    raw["xPPS_base"]     = raw.apply(lambda r: _get_prior_tracking(r, "xPPS_base"),    axis=1)
    raw["prior_pot_ast"] = raw.apply(lambda r: _get_prior_tracking(r, "potentialAst"), axis=1)
    raw["prior_drives"]  = raw.apply(lambda r: _get_prior_tracking(r, "drivePg"),      axis=1)
    raw["pct_pts_paint"] = raw.apply(lambda r: _get_prior_tracking(r, "pctPtsPaint"),  axis=1).fillna(0.0)
    raw["pct_pts_3pt"]   = raw.apply(lambda r: _get_prior_tracking(r, "pctPts3pt"),    axis=1).fillna(0.0)

    # efficiency_delta: player's recent TS% vs their prior-season shot-quality baseline.
    # At inference: server.py computes l5_ts - xPPS from live tracking (same definition).
    # Fallback to league-avg baseline when prior tracking is unavailable (rookies, etc.).
    raw["efficiency_delta"] = np.where(
        raw["xPPS_base"].notna(),
        (raw["l5_ts"] - raw["xPPS_base"]).fillna(0.0),
        (raw["l5_ts"] - _LEAGUE_AVG_TS_TRAIN).fillna(0.0),
    )

    # l5_potential_ast: real prior-season potentialAst/g replaces the l5_ast × 3.33 proxy.
    # At inference: server.py uses live potentialAst/g from passing tracking.
    raw["l5_potential_ast"] = np.where(
        raw["prior_pot_ast"].notna() & (raw["prior_pot_ast"] > 0),
        raw["prior_pot_ast"],
        (raw["l5_ast"] * 3.33).fillna(0.0),   # proxy fallback for rookies / no prior data
    )

    # ── 5c. Compute inactive pools ────────────────────────────────────────────
    # inactive_usg_pool: sum of absent teammates' rolling USG proxies
    usg_ser     = raw.set_index(["PLAYER_ID", "GAME_DATE"])["l5_usg_proxy"].dropna()
    usg_lookup  = {(int(pid), date): v for (pid, date), v in usg_ser.items()}
    raw["inactive_usg_pool"] = _compute_inactive_pool(raw, usg_lookup, "inactive_usg_pool", key_mode="date")
    print(f"  inactive_usg_pool: mean={raw['inactive_usg_pool'].mean():.2f}  "
          f"max={raw['inactive_usg_pool'].max():.2f}  "
          f"non-zero={(raw['inactive_usg_pool'] > 0).mean():.1%}")

    # inactive_potential_ast_pool: sum of absent teammates' prior-season potentialAst/g
    # Captures the void in creation volume when an elite passer is scratched.
    pa_lookup = {}
    for _, row in raw[["PLAYER_ID_int", "season", "prior_pot_ast"]].drop_duplicates().iterrows():
        if pd.notna(row["prior_pot_ast"]):
            pa_lookup[(int(row["PLAYER_ID_int"]), row["season"])] = float(row["prior_pot_ast"])
    raw["inactive_potential_ast_pool"] = _compute_inactive_pool(
        raw, pa_lookup, "inactive_potential_ast_pool", key_mode="season"
    )
    print(f"  inactive_potential_ast_pool: mean={raw['inactive_potential_ast_pool'].mean():.2f}  "
          f"non-zero={(raw['inactive_potential_ast_pool'] > 0).mean():.1%}")

    # inactive_drives_pool: sum of absent teammates' prior-season drives/g
    # Captures the void in paint pressure when a drive-heavy creator is scratched.
    drives_lookup = {}
    for _, row in raw[["PLAYER_ID_int", "season", "prior_drives"]].drop_duplicates().iterrows():
        if pd.notna(row["prior_drives"]):
            drives_lookup[(int(row["PLAYER_ID_int"]), row["season"])] = float(row["prior_drives"])
    raw["inactive_drives_pool"] = _compute_inactive_pool(
        raw, drives_lookup, "inactive_drives_pool", key_mode="season"
    )
    print(f"  inactive_drives_pool: mean={raw['inactive_drives_pool'].mean():.2f}  "
          f"non-zero={(raw['inactive_drives_pool'] > 0).mean():.1%}")

    # ── 5d. Derived interaction features ─────────────────────────────────────
    # leverage_index: regular season=0, playoffs=1 (G7 detection too sparse at 2 seasons)
    raw["leverage_index"] = (raw["season_type"] == "Playoffs").astype(float)

    # Stylistic matchup overlays: player shot-diet fraction × opponent zone concession
    # A perimeter shooter facing a soft arc defense gets a large positive scalar;
    # a post-up center in the same game gets near-zero — tree sees the interaction directly.
    raw["paint_overlay"]     = raw["pct_pts_paint"] * raw["rim_vs_avg"].fillna(0.0)
    raw["perimeter_overlay"] = raw["pct_pts_3pt"]   * raw["fg3_vs_avg"].fillna(0.0)

    # Role-vacuum absorption overlays: player archetype rate × inactive teammate pool size
    # Ensures a high-ast-rate guard absorbs a creation void far more than a spot-up shooter.
    raw["creation_absorption"] = raw["prior_pot_ast"].fillna(0.0) * raw["inactive_potential_ast_pool"]
    raw["slashing_absorption"] = raw["prior_drives"].fillna(0.0)  * raw["inactive_drives_pool"]

    # ── 6. Define targets ─────────────────────────────────────────────────────
    raw["target_pts"] = raw["PTS"]
    raw["target_reb"] = raw["REB"]
    raw["target_ast"] = raw["AST"]

    # ── 7. Select output columns & filter ─────────────────────────────────────
    keep = [
        # Identifiers
        "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE",
        "season", "season_type",
        # Pre-game features — flat windows
        "is_home", "rest_days", "is_b2b", "is_well_rested",
        "l5_pts", "l5_reb", "l5_ast", "l5_min", "l5_ts",
        "l10_pts_std", "l10_min_std",
        "std_pts", "std_reb", "std_ast", "std_min",
        "gp_prior",
        "opp_def_roll10", "opp_pace_roll10",
        # EWMA recency features
        "ewma_pts", "ewma_reb", "ewma_ast", "ewma_min",
        # Usage redistribution pools
        "inactive_usg_pool",
        "inactive_potential_ast_pool",
        "inactive_drives_pool",
        # Tracking-derived features
        "fg3_vs_avg", "rim_vs_avg",       # opponent scheme concessions
        "xPPS_base",                       # prior-season shot-quality baseline
        "efficiency_delta",               # l5_ts - xPPS_base (regression signal)
        "l5_potential_ast",               # prior-season creation volume
        # Context and interaction features
        "leverage_index",                  # 0=RS, 1=playoffs
        "paint_overlay",                   # pct_pts_paint × rim_vs_avg
        "perimeter_overlay",               # pct_pts_3pt × fg3_vs_avg
        "creation_absorption",             # prior_pot_ast × inactive_potential_ast_pool
        "slashing_absorption",             # prior_drives × inactive_drives_pool
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
    feat_cols_diag = ["l5_pts", "l5_reb", "l5_ast", "l5_min",
                      "ewma_pts", "ewma_reb", "ewma_ast",
                      "inactive_usg_pool", "opp_def_roll10", "opp_pace_roll10",
                      "xPPS_base", "efficiency_delta", "l5_potential_ast",
                      "inactive_potential_ast_pool", "inactive_drives_pool",
                      "paint_overlay", "perimeter_overlay",
                      "creation_absorption", "slashing_absorption"]
    print(out[[c for c in feat_cols_diag if c in out.columns]].isna().sum().to_string())

    out.to_parquet(OUTPUT_FILE, index=False)
    print(f"\nSaved → {OUTPUT_FILE}")

    # Save feature column list and median imputation values for inference
    feature_cols = [
        "l5_pts", "l5_reb", "l5_ast", "l5_min", "l5_ts",
        "l10_pts_std", "l10_min_std",
        "std_pts", "std_reb", "std_ast", "std_min",
        "gp_prior", "is_home", "rest_days",
        "is_b2b", "is_well_rested",
        "opp_def_roll10", "opp_pace_roll10",
        "ewma_pts", "ewma_reb", "ewma_ast", "ewma_min",
        "inactive_usg_pool",
        # Opponent scheme concessions
        "fg3_vs_avg", "rim_vs_avg",
        # Tracking-derived features (prior-season archetypes at training; live values at inference)
        "xPPS_base",
        "efficiency_delta",
        "l5_potential_ast",
        # Multi-dimensional inactive teammate pools
        "inactive_potential_ast_pool",
        "inactive_drives_pool",
        # Context and interaction features
        "leverage_index",
        "paint_overlay",
        "perimeter_overlay",
        "creation_absorption",
        "slashing_absorption",
    ]
    medians = {c: float(out[c].median()) for c in feature_cols if c in out.columns}
    meta = {"feature_cols": feature_cols, "medians": medians}
    with open("model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved → model_meta.json")


if __name__ == "__main__":
    main()
