import math
import random
import os
import statistics
from flask import Flask, jsonify, request
from flask_cors import CORS
from nba_api.stats.endpoints import (
    leaguedashplayerstats,
    leaguedashteamstats,
    leaguedashptteamdefend,
    leaguedashplayerclutch,
    leaguehustlestatsplayer,
    leaguedashptstats,
    playergamelog,
)
from nba_api.live.nba.endpoints import scoreboard as live_scoreboard, boxscore as live_boxscore
from nba_api.stats.static import teams as nba_teams_static
from datetime import datetime, timezone, timedelta
import pandas as pd
import time
import logging
import threading
import json
import urllib.request
import urllib.error

# XGBoost — optional; falls back to multiplier model if not installed / models absent
try:
    import xgboost as _xgb_lib
    _XGB_AVAILABLE = True
except ImportError:
    _xgb_lib      = None
    _XGB_AVAILABLE = False

# Live odds — optional; gracefully unavailable if ODDS_API_KEY not set
try:
    import requests as _requests_lib
    from cachetools import TTLCache, cached as _ct_cached
    _ODDS_CACHE   = TTLCache(maxsize=20, ttl=600)
    _ODDS_AVAILABLE = True
except ImportError:
    _requests_lib = None
    _ODDS_CACHE   = None
    _ODDS_AVAILABLE = False

SERVER_VERSION = "v6.12.0"  # quantile models + KNN Monte Carlo + adj gate live

# Static TEAM_ID → abbreviation lookup (no API call needed)
_TEAM_ID_TO_ABBR = {t["id"]: t["abbreviation"] for t in nba_teams_static.get_teams()}
_TEAM_NAME_TO_ABBR = {t["full_name"]: t["abbreviation"] for t in nba_teams_static.get_teams()}
_TEAM_ABBR_TO_ID  = {v: k for k, v in _TEAM_ID_TO_ABBR.items()}

# ESPN uses shorter abbreviations that differ from NBA stats API
# Normalize ESPN → NBA so scoreboard enrichment key-matching works
_ESPN_TO_NBA_ABBR = {
    "SA": "SAS", "GS": "GSW", "NY": "NYK", "NO": "NOP",
    "UTAH": "UTA", "MEM": "MEM", "PHX": "PHX",
}

def _norm_abbr(abbr: str) -> str:
    """Normalize ESPN/short abbreviation to NBA stats API abbreviation."""
    return _ESPN_TO_NBA_ABBR.get((abbr or "").upper(), (abbr or "").upper())

# ── Injury architecture (updated May 4 2026) ─────────────────────────────────
# Priority order: _CLEARED_PLAYERS > ESPN live > _INJURY_OVERRIDES (gap-fill only)
#
# _CLEARED_PLAYERS  — confirmed ACTIVE, removed from report even if ESPN still lags.
#                     Add a player here the moment they are confirmed playing.
# _INJURY_OVERRIDES — ONLY used when ESPN has NO entry for a player (fills ESPN gaps).
#                     Do NOT put returning/day-to-day players here.
#                     Use for confirmed season-ending or multi-week absences only.
# ESPN live data    — primary source of truth, fetched fresh every request.

_CLEARED_PLAYERS = {
    # Add confirmed-active returning players here so ESPN lag can't mark them OUT
    "anthony edwards",   # returned from knee injury — confirmed active May 4 2026
}

# Gap-fill only: applied ONLY when ESPN returns no entry for this player.
# Keep to truly confirmed, long-term injuries — never day-to-day status.
# Updated: May 4 2026
_INJURY_OVERRIDES = {
    "franz wagner":     {"status": "Out", "detail": "Calf strain — confirmed OUT (ESPN May 3 2026)", "team": "ORL"},
    "kevin durant":     {"status": "Out", "detail": "Left ankle bone bruise — out (NBA official)", "team": "HOU"},
    "fred vanvleet":    {"status": "Out", "detail": "Right knee ACL repair — out for season", "team": "CLE"},
    "donte divincenzo": {"status": "Out", "detail": "Right Achilles repair — out for season", "team": "NYK"},
    "luka doncic":      {"status": "Out", "detail": "Left hamstring strain — no timetable", "team": "LAL"},
    "steven adams":     {"status": "Out", "detail": "Left ankle surgery — out for season", "team": "HOU"},
    "ayo dosunmu":      {"status": "Out", "detail": "Injury — confirmed OUT (May 2026)", "team": "MIN"},
    "joel embiid":      {"status": "Out", "detail": "Confirmed OUT — May 6 2026", "team": "PHI"},
}

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
CORS(app)

SEASON = "2025-26"
_CACHE_TTL = 3600  # 1 hour (default; see _dynamic_ttl for time-aware values)

# ── XGBoost model registry ────────────────────────────────────────────────────
# Populated by _load_xgb_models() during warmup.
# _XGB_MODELS["pts"]      → Poisson point-estimate model
# _XGB_QUANTILE["pts"]["q25/q50/q75"] → quantile models (may be absent on old deploys)
_XGB_MODELS:   dict = {}
_XGB_QUANTILE: dict = {}   # { "pts": {"q25": model, "q50": model, "q75": model}, ... }
_XGB_META:     dict = {}
_XGB_LOAD_ERRORS: dict = {}

_XGB_PROP_MAP = {
    "points":     "pts",
    "rebounds":   "reb",
    "assists":    "ast",
    "pra":        None,
    "pa":         None,
    "pr":         None,
}

def _load_xgb_models():
    """Load Poisson + quantile XGBoost model files and metadata from disk."""
    global _XGB_MODELS, _XGB_QUANTILE, _XGB_META
    if not _XGB_AVAILABLE:
        logging.info("XGBoost not installed — running multiplier-only mode.")
        return

    base = os.path.dirname(__file__)
    meta_path = os.path.join(base, "model_meta.json")
    if not os.path.exists(meta_path):
        logging.info("model_meta.json not found — XGBoost inference disabled.")
        return

    with open(meta_path) as f:
        _XGB_META = json.load(f)

    loaded = []
    for prop in ("pts", "reb", "ast"):
        # Primary Poisson model
        mp = os.path.join(base, f"xgb_{prop}_model.json")
        if not os.path.exists(mp):
            logging.warning("XGB model file missing: %s", mp)
            continue
        try:
            m = _xgb_lib.XGBRegressor()
            m.load_model(mp)
            _XGB_MODELS[prop] = m
            loaded.append(prop)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            _XGB_LOAD_ERRORS[prop] = err
            logging.error("XGB %s load failed: %s", prop, err)

        # Quantile models (q25 / q50 / q75) — optional, present after retrain
        _XGB_QUANTILE[prop] = {}
        for tag in ("q25", "q50", "q75"):
            qp = os.path.join(base, f"xgb_{prop}_{tag}.json")
            if not os.path.exists(qp):
                continue
            try:
                qm = _xgb_lib.XGBRegressor()
                qm.load_model(qp)
                _XGB_QUANTILE[prop][tag] = qm
            except Exception as e:
                logging.warning("XGB %s %s load failed: %s", prop, tag, e)

        if _XGB_QUANTILE.get(prop):
            logging.info("XGB %s quantile models loaded: %s", prop, list(_XGB_QUANTILE[prop]))

    logging.info("XGBoost models loaded: %s (xgb v%s)",
                 loaded or "none", getattr(_xgb_lib, "__version__", "?"))


def _dynamic_ttl(base_seconds=3600):
    """
    Return TTL adjusted for time-of-day. Stats don't change overnight, so we
    can hold cache much longer when nothing's happening.

      02:00–14:00 ET (dead window)  → 4× base (4 hours default)
      14:00–18:00 ET (pregame)      → 1× base (1 hour default)
      18:00–02:00 ET (game time)    → 0.5× base (30 min default)

    Slashes external API hits during the long quiet stretches without ever
    going stale near tipoff. Any caller can override base_seconds.
    """
    try:
        et_hour = datetime.now(timezone(timedelta(hours=-4))).hour
    except Exception:
        return base_seconds
    if 2 <= et_hour < 14:
        return int(base_seconds * 4)
    if 14 <= et_hour < 18:
        return int(base_seconds)
    return max(60, int(base_seconds * 0.5))

# ── In-memory cache ───────────────────────────────────────────────────────────
_cache: dict = {}
_cache_lock = threading.Lock()
# Event set to True once the warmup thread finishes.
# _cached_endpoint waits on this (up to 240s) instead of blocking forever
# or fast-failing with 503 — whichever the warmup finishes first.
_warmup_done = threading.Event()

def _keep_alive():
    import os
    port = os.environ.get("PORT", "10000")
    time.sleep(300)
    while True:
        try:
            urllib.request.urlopen(f"http://localhost:{port}/api/health", timeout=10)
            logging.info("Keep-alive ping sent")
        except Exception as e:
            logging.warning("Keep-alive ping failed: %s", e)
        time.sleep(600)

threading.Thread(target=_keep_alive, daemon=True).start()

def _cache_get(key, ttl=None):
    """
    Cache lookup with time-aware TTL.
    Pass `ttl` for per-call override; otherwise uses _dynamic_ttl(_CACHE_TTL)
    which extends to 4 hours during dead overnight hours and tightens to
    30 min near tipoff.
    """
    effective_ttl = ttl if ttl is not None else _dynamic_ttl(_CACHE_TTL)
    with _cache_lock:
        entry = _cache.get(key)
        if entry and (time.time() - entry["ts"]) < effective_ttl:
            logging.info("Cache HIT: %s (age %.0fs, ttl %ds)",
                         key, time.time() - entry["ts"], effective_ttl)
            return entry["data"]
    return None


def _cache_set(key, data):
    with _cache_lock:
        _cache[key] = {"data": data, "ts": time.time()}
    logging.info("Cache SET: %s", key)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _sleep():
    time.sleep(0.8)


def _pct(val):
    try:
        v = float(val)
        return round(v * 100, 1) if v else 0.0
    except (TypeError, ValueError):
        return 0.0


def _f(val, d=1):
    try:
        return round(float(val), d)
    except (TypeError, ValueError):
        return 0.0


def _i(val):
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


def _stat_row(r):
    if not r:
        return None
    return {
        "ppg":  _f(r.get("PTS", 0)),
        "rpg":  _f(r.get("REB", 0)),
        "apg":  _f(r.get("AST", 0)),
        "spg":  _f(r.get("STL", 0)),
        "bpg":  _f(r.get("BLK", 0)),
        "topg": _f(r.get("TOV", 0)),
        "fg":   _pct(r.get("FG_PCT")),
        "fg3":  _pct(r.get("FG3_PCT")),
        "ft":   _pct(r.get("FT_PCT")),
        "min":  _f(r.get("MIN", 0)),
        "gp":   _i(r.get("GP", 0)),
    }


def _fetch_player_stats(season_type, measure="Base", location=None):
    kwargs = dict(
        season=SEASON,
        season_type_all_star=season_type,
        per_mode_detailed="PerGame",
        measure_type_detailed_defense=measure,
    )
    if location:
        kwargs["location_nullable"] = location
    return leaguedashplayerstats.LeagueDashPlayerStats(**kwargs).get_data_frames()[0]


# ── Core data builders ────────────────────────────────────────────────────────
def _build_players():
    logging.info("Fetching PO base stats...")
    po_base = _fetch_player_stats("Playoffs")
    _sleep()
    logging.info("Fetching PO advanced stats...")
    po_adv = _fetch_player_stats("Playoffs", "Advanced")
    _sleep()
    logging.info("Fetching RS base stats...")
    rs_base = _fetch_player_stats("Regular Season")
    _sleep()
    logging.info("Fetching RS advanced stats...")
    rs_adv = _fetch_player_stats("Regular Season", "Advanced")

    po_adv_idx = po_adv.set_index("PLAYER_ID").to_dict("index")
    rs_adv_idx = rs_adv.set_index("PLAYER_ID").to_dict("index")
    rs_base_idx = rs_base.set_index("PLAYER_ID").to_dict("index")

    players = {}
    for _, row in po_base.iterrows():
        pid = int(row["PLAYER_ID"])
        name = row["PLAYER_NAME"].lower()
        team = row["TEAM_ABBREVIATION"]
        pa = po_adv_idx.get(pid, {})
        ra = rs_adv_idx.get(pid, {})
        rb = rs_base_idx.get(pid, {})

        po = {
            "ppg": _f(row["PTS"]), "rpg": _f(row["REB"]), "apg": _f(row["AST"]),
            "spg": _f(row["STL"]), "bpg": _f(row["BLK"]), "topg": _f(row["TOV"]),
            "fg": _pct(row.get("FG_PCT")), "fg3": _pct(row.get("FG3_PCT")),
            "ft": _pct(row.get("FT_PCT")), "min": _f(row["MIN"]), "gp": _i(row["GP"]),
            "usg": _pct(pa.get("USG_PCT")) if pa else None,
            "ts":  _pct(pa.get("TS_PCT"))  if pa else None,
        }
        if rb:
            rs = {
                "ppg": _f(rb.get("PTS", 0)), "rpg": _f(rb.get("REB", 0)),
                "apg": _f(rb.get("AST", 0)), "spg": _f(rb.get("STL", 0)),
                "bpg": _f(rb.get("BLK", 0)), "topg": _f(rb.get("TOV", 0)),
                "fg": _pct(rb.get("FG_PCT")), "fg3": _pct(rb.get("FG3_PCT")),
                "ft": _pct(rb.get("FT_PCT")), "min": _f(rb.get("MIN", 0)),
                "gp": _i(rb.get("GP", 0)),
                "usg": _pct(ra.get("USG_PCT")) if ra else None,
                "ts":  _pct(ra.get("TS_PCT"))  if ra else None,
            }
        else:
            rs = po
        players[name] = {"team": team, "pid": pid, "rs": rs, "po": po}

    logging.info("Built %d playoff players", len(players))
    return players


def _build_teams():
    """
    Build team stats with RS as the FOUNDATION (82-game sample) and PO as the
    overlay (small-sample but recent). Pace comes from RS (stable trait).
    Final dEFF/oEFF is a sample-weighted blend so first-round teams whose PO
    stats are entirely from one matchup don't pollute the data.

    Critical fix: previously used PO Advanced exclusively, which made BOS and
    PHI show identical pace 93.6 and mirror dEFF/oEFF (because they only
    played each other). Now uses real RS values with PO weighted by GP.
    """
    teams_data = {}

    def _try_team_fetch(season_type, measure):
        df = leaguedashteamstats.LeagueDashTeamStats(
            season=SEASON, season_type_all_star=season_type,
            per_mode_detailed="PerGame",
            measure_type_detailed_defense=measure,
        ).get_data_frames()[0]
        logging.info("Team stats [%s/%s] rows:%d cols:%s",
                     season_type, measure, len(df), df.columns.tolist()[:8])
        return df

    def _resolve_abbr(row):
        return (row.get("TEAM_ABBREVIATION")
                or _TEAM_ID_TO_ABBR.get(int(row.get("TEAM_ID", 0)))
                or _TEAM_NAME_TO_ABBR.get(row.get("TEAM_NAME", ""))
                or str(row.get("TEAM_NAME", "UNK"))[:3].upper())

    def _fetch_first_working(season_type):
        """Try Advanced → Opponent → Base; return (df, measure_used) or (None, None)."""
        for measure in ("Advanced", "Opponent", "Base"):
            try:
                df = _try_team_fetch(season_type, measure)
                _sleep()
                if (not df.empty and
                    ("TEAM_ID" in df.columns or "TEAM_ABBREVIATION" in df.columns)):
                    return df, measure
            except Exception as e:
                logging.warning("Team stats [%s/%s] failed: %s", season_type, measure, e)
        return None, None

    # ── Step 1: RS = foundation (full-season pace + season-long efficiency) ──
    rs_df, rs_measure = _fetch_first_working("Regular Season")
    if rs_df is not None:
        for _, row in rs_df.iterrows():
            abbr = _resolve_abbr(row)
            if abbr == "UNK":
                continue
            o = row.get("OFF_RATING") or row.get("PTS")
            d = row.get("DEF_RATING") or row.get("OPP_PTS")
            teams_data[abbr] = {
                "fullName":  row.get("TEAM_NAME", abbr),
                "rsPace":    _f(row.get("PACE") or 100.0),
                "rsGP":      int(row.get("GP") or 0),
                "rsOEFF":    _f(o) if o else None,
                "rsDEFF":    _f(d) if d else None,
                "poGP":      0,
                "poPace":    None,
                "poOEFF":    None,
                "poDEFF":    None,
            }
        logging.info("RS team baseline: %d teams via %s", len(teams_data), rs_measure)

    # ── Step 2: PO overlay (small sample but most recent) ─────────────────────
    po_df, po_measure = _fetch_first_working("Playoffs")
    if po_df is not None:
        for _, row in po_df.iterrows():
            abbr = _resolve_abbr(row)
            if abbr == "UNK":
                continue
            if abbr not in teams_data:
                # Team has PO data but no RS data — uncommon, but handle it
                teams_data[abbr] = {
                    "fullName": row.get("TEAM_NAME", abbr),
                    "rsPace": None, "rsGP": 0, "rsOEFF": None, "rsDEFF": None,
                }
            o = row.get("OFF_RATING") or row.get("PTS")
            d = row.get("DEF_RATING") or row.get("OPP_PTS")
            teams_data[abbr]["poGP"]   = int(row.get("GP") or 0)
            teams_data[abbr]["poPace"] = _f(row.get("PACE")) if row.get("PACE") else None
            teams_data[abbr]["poOEFF"] = _f(o) if o else None
            teams_data[abbr]["poDEFF"] = _f(d) if d else None
        logging.info("PO team overlay: %d teams via %s", len(po_df), po_measure)

    # ── Step 3: Compute final blended dEFF/oEFF ──────────────────────────────
    # Weight PO based on sample size: 0 PO games = 0% weight, 7+ games = 50% weight.
    # This prevents 1-2 PO games from drowning out 82 RS games.
    for abbr, t in teams_data.items():
        po_gp = int(t.get("poGP") or 0)
        po_w  = min(0.5, po_gp / 14.0)   # 7 PO games = 50% weight
        rs_w  = 1.0 - po_w

        rs_d, po_d = t.get("rsDEFF"), t.get("poDEFF")
        rs_o, po_o = t.get("rsOEFF"), t.get("poOEFF")

        if rs_d and po_d:
            t["dEFF"] = round(rs_d * rs_w + po_d * po_w, 1)
        else:
            t["dEFF"] = rs_d or po_d

        if rs_o and po_o:
            t["oEFF"] = round(rs_o * rs_w + po_o * po_w, 1)
        else:
            t["oEFF"] = rs_o or po_o

        t["eDIFF"] = round((t["oEFF"] or 0) - (t["dEFF"] or 0), 1) if (t["oEFF"] and t["dEFF"]) else None

    if not teams_data:
        raise RuntimeError("All team stat fetches failed")

    logging.info("Built %d teams total (RS+PO blended)", len(teams_data))
    return teams_data


def _build_splits():
    logging.info("Fetching PO home splits...")
    home_df = _fetch_player_stats("Playoffs", location="Home")
    _sleep()
    logging.info("Fetching PO road splits...")
    road_df = _fetch_player_stats("Playoffs", location="Road")

    home_idx = home_df.set_index("PLAYER_ID").to_dict("index")
    road_idx = road_df.set_index("PLAYER_ID").to_dict("index")

    all_players: dict = {}
    for df in (home_df, road_df):
        for _, row in df.iterrows():
            pid = int(row["PLAYER_ID"])
            name = row["PLAYER_NAME"].lower()
            if name not in all_players:
                all_players[name] = pid

    splits = {}
    for name, pid in all_players.items():
        splits[name] = {
            "pid": pid,
            "home": _stat_row(home_idx.get(pid)),
            "road": _stat_row(road_idx.get(pid)),
        }

    logging.info("Built splits for %d players", len(splits))
    return splits


def _build_team_defense():
    logging.info("Fetching PO 3pt team defense...")
    fg3_df = leaguedashptteamdefend.LeagueDashPtTeamDefend(
        season=SEASON, season_type_all_star="Playoffs",
        defense_category="3 Pointers", per_mode_simple="PerGame",
    ).get_data_frames()[0]
    _sleep()

    logging.info("Fetching PO rim (<6ft) team defense...")
    rim_df = leaguedashptteamdefend.LeagueDashPtTeamDefend(
        season=SEASON, season_type_all_star="Playoffs",
        defense_category="Less Than 6Ft", per_mode_simple="PerGame",
    ).get_data_frames()[0]

    logging.info("fg3 team defend cols: %s", fg3_df.columns.tolist())
    logging.info("rim team defend cols: %s", rim_df.columns.tolist())

    def _abbr_from_row(row):
        return (row.get("TEAM_ABBREVIATION")
                or _TEAM_ID_TO_ABBR.get(int(row.get("TEAM_ID", 0)))
                or _TEAM_NAME_TO_ABBR.get(row.get("TEAM_NAME", ""))
                or "UNK")

    fg3_idx = {_abbr_from_row(r): r.to_dict() for _, r in fg3_df.iterrows()}
    rim_idx  = {_abbr_from_row(r): r.to_dict() for _, r in rim_df.iterrows()}

    # Column names confirmed from debug-team-defense endpoint (2025-26 nba_api):
    #   3pt:  PLUSMINUS = FG3_PCT - NS_FG3_PCT,  FG3_PCT = opp 3pt pct
    #   rim:  PLUSMINUS = LT_06_PCT - NS_LT_06_PCT,  LT_06_PCT = opp rim pct
    sample_fg3 = next(iter(fg3_idx.values()), {})
    sample_rim = next(iter(rim_idx.values()), {})
    vsavg_col   = next((c for c in ["PLUSMINUS", "PCT_PLUSMINUS", "PCT_PLUS_MINUS"]
                        if c in sample_fg3), "PLUSMINUS")
    fg3pct_col  = next((c for c in ["FG3_PCT", "D_FG_PCT", "FG_PCT"]
                        if c in sample_fg3), "FG3_PCT")
    rim_pct_col = next((c for c in ["LT_06_PCT", "D_FG_PCT", "FG_PCT"]
                        if c in sample_rim), "LT_06_PCT")
    logging.info("Team def cols: vsavg=%s fg3pct=%s rimpct=%s", vsavg_col, fg3pct_col, rim_pct_col)

    all_abbrs = set(fg3_idx.keys()) | set(rim_idx.keys())
    team_def = {}
    for abbr in all_abbrs:
        fg3 = fg3_idx.get(abbr, {})
        rim = rim_idx.get(abbr, {})
        team_def[abbr] = {
            "fg3VsAvg":  _f(fg3.get(vsavg_col,   0), 4),
            "fg3OppPct": _f(fg3.get(fg3pct_col,  0), 4),
            "rimVsAvg":  _f(rim.get(vsavg_col,   0), 4),
            "rimOppPct": _f(rim.get(rim_pct_col, 0), 4),
        }
    logging.info("Built team defense for %d teams", len(team_def))
    return team_def


def _build_scoring():
    """
    Shot profile breakdown per player (Playoffs).
    Key cols: PCT_PTS_3PT, PCT_PTS_PAINT, PCT_FGA_3PT, PCT_PTS_FT, PCT_PTS_2PT_MR
    Used to weight zone-specific defense by how a player actually scores.
    """
    logging.info("Fetching PO scoring breakdown...")
    df = _fetch_player_stats("Playoffs", "Scoring")

    scoring = {}
    for _, row in df.iterrows():
        name = row["PLAYER_NAME"].lower()
        scoring[name] = {
            "pid":        int(row["PLAYER_ID"]),
            "pctPts3pt":  _f(row.get("PCT_PTS_3PT", 0) or 0, 1) * 100,   # % of PTS from 3s
            "pctPtsPaint":_f(row.get("PCT_PTS_PAINT", 0) or 0, 1) * 100,  # % from paint
            "pctPtsFt":   _f(row.get("PCT_PTS_FT", 0) or 0, 1) * 100,    # % from FTs
            "pctPtsMr":   _f(row.get("PCT_PTS_2PT_MR", 0) or 0, 1) * 100,# % midrange
            "pctFga3pt":  _f(row.get("PCT_FGA_3PT", 0) or 0, 1) * 100,   # % of shots = 3s
            "efgPct":     _pct(row.get("EFG_PCT")),
        }

    logging.info("Built scoring breakdown for %d players", len(scoring))
    return scoring


def _build_clutch():
    """
    Clutch performance per player (Playoffs, last 5 min within 5 pts).
    Used to adjust scoring projections for players who elevate/disappear in close games.
    """
    logging.info("Fetching PO clutch stats...")
    df = leaguedashplayerclutch.LeagueDashPlayerClutch(
        season=SEASON,
        season_type_all_star="Playoffs",
        per_mode_detailed="PerGame",
    ).get_data_frames()[0]

    clutch = {}
    for _, row in df.iterrows():
        name = row["PLAYER_NAME"].lower()
        clutch[name] = {
            "pid": int(row["PLAYER_ID"]),
            "gp":  _i(row.get("GP", 0)),
            "min": _f(row.get("MIN", 0)),
            "ppg": _f(row.get("PTS", 0)),
            "rpg": _f(row.get("REB", 0)),
            "apg": _f(row.get("AST", 0)),
            "fg":  _pct(row.get("FG_PCT")),
            "fg3": _pct(row.get("FG3_PCT")),
            "ft":  _pct(row.get("FT_PCT")),
            "pm":  _f(row.get("PLUS_MINUS", 0)),
        }

    logging.info("Built clutch stats for %d players", len(clutch))
    return clutch


def _build_hustle():
    """
    Hustle stats per player (Playoffs): contested shots, deflections, box-outs.
    Used for rebounds/steals prop context and adjustments.
    """
    logging.info("Fetching PO hustle stats...")
    df = leaguehustlestatsplayer.LeagueHustleStatsPlayer(
        season=SEASON,
        season_type_all_star="Playoffs",
        per_mode_time="PerGame",
    ).get_data_frames()[0]

    hustle = {}
    for _, row in df.iterrows():
        name = row["PLAYER_NAME"].lower()
        hustle[name] = {
            "pid":            int(row["PLAYER_ID"]),
            "gp":             _i(row.get("G", 0)),
            "min":            _f(row.get("MIN", 0)),
            "contestedShots": _f(row.get("CONTESTED_SHOTS", 0)),
            "contested2pt":   _f(row.get("CONTESTED_SHOTS_2PT", 0)),
            "contested3pt":   _f(row.get("CONTESTED_SHOTS_3PT", 0)),
            "deflections":    _f(row.get("DEFLECTIONS", 0)),
            "chargesDrawn":   _f(row.get("CHARGES_DRAWN", 0)),
            "screenAssists":  _f(row.get("SCREEN_ASSISTS", 0)),
            "defBoxouts":     _f(row.get("DEF_BOXOUTS", 0)),
            "offBoxouts":     _f(row.get("OFF_BOXOUTS", 0)),
            "boxoutRebounds": _f(row.get("BOX_OUT_PLAYER_REBS", 0)),
        }

    logging.info("Built hustle stats for %d players", len(hustle))
    return hustle


def _build_tracking():
    """
    Player tracking stats — passing + rebounding merged per player.
    Tries Playoffs first; falls back to Regular Season if PO data is empty
    (tracking data is sparse early in the playoffs).
    Passing  → POTENTIAL_AST, AST, PASSES_MADE (for AST conversion rate)
    Rebounding → OREB_CHANCE, DREB_CHANCE, REB_CHANCE_PCT (optional enrichment)
    """
    def _fetch_passing(season_type):
        df = leaguedashptstats.LeagueDashPtStats(
            season=SEASON,
            season_type_all_star=season_type,
            per_mode_simple="PerGame",
            pt_measure_type="Passing",
        ).get_data_frames()[0]
        logging.info("Passing tracking [%s] cols: %s  rows: %d",
                     season_type, df.columns.tolist()[:8], len(df))
        return df

    def _fetch_rebounding(season_type):
        return leaguedashptstats.LeagueDashPtStats(
            season=SEASON,
            season_type_all_star=season_type,
            per_mode_simple="PerGame",
            pt_measure_type="Rebounding",
        ).get_data_frames()[0]

    # Try PO first, fall back to RS if empty
    pass_df = None
    for stype in ("Playoffs", "Regular Season"):
        try:
            df = _fetch_passing(stype)
            _sleep()
            if not df.empty and "PLAYER_ID" in df.columns:
                pass_df = df
                logging.info("Using %s passing tracking (%d players)", stype, len(df))
                break
        except Exception as e:
            logging.warning("Passing tracking [%s] failed: %s", stype, e)

    if pass_df is None or pass_df.empty:
        logging.warning("No passing tracking data available — tracking cache will be empty")
        return {}

    # ── Rebounding (optional enrichment) ─────────────────────────────────────
    reb_idx = {}
    for stype in ("Playoffs", "Regular Season"):
        try:
            reb_df = _fetch_rebounding(stype)
            _sleep()
            pid_col = next(
                (c for c in ["PLAYER_ID", "PlayerID", "PERSONID"] if c in reb_df.columns),
                None,
            )
            if pid_col and not reb_df.empty:
                reb_idx = reb_df.set_index(pid_col).to_dict("index")
                logging.info("Rebounding tracking [%s] %d players", stype, len(reb_idx))
                break
        except Exception as e:
            logging.warning("Rebounding tracking [%s] failed: %s", stype, e)

    # ── Detect player ID / name columns in passing DF ──────────────────────────
    pid_col  = next((c for c in ["PLAYER_ID", "PlayerID", "PERSONID"] if c in pass_df.columns), None)
    name_col = next((c for c in ["PLAYER_NAME", "PlayerName", "PLAYER"] if c in pass_df.columns), None)

    tracking = {}
    for _, row in pass_df.iterrows():
        if pid_col is None or name_col is None:
            break  # can't build without player identity columns
        pid  = int(row[pid_col])
        name = str(row[name_col]).lower()
        reb  = reb_idx.get(pid, {})

        potential_ast = _f(row.get("POTENTIAL_AST", 0) or 0)
        actual_ast    = _f(row.get("AST",           0) or 0)
        conv_rate     = round(actual_ast / max(potential_ast, 0.1), 3) if potential_ast > 0.1 else None

        tracking[name] = {
            "pid":            pid,
            "gp":             _i(row.get("GP", 0)),
            "potentialAst":   potential_ast,
            "ast":            actual_ast,
            "astConvRate":    conv_rate,
            "passes":         _f(row.get("PASSES_MADE",     0) or 0),
            "passesReceived": _f(row.get("PASSES_RECEIVED", 0) or 0),
            "secondaryAst":   _f(row.get("SECONDARY_AST",   0) or 0),
            "orebChance":     _f(reb.get("OREB_CHANCE",     0) or 0) if reb else 0,
            "drebChance":     _f(reb.get("DREB_CHANCE",     0) or 0) if reb else 0,
            "orebChancePct":  _pct(reb.get("OREB_CHANCE_PCT")) if reb else 0,
            "drebChancePct":  _pct(reb.get("DREB_CHANCE_PCT")) if reb else 0,
            "rebChancePct":   _pct(reb.get("REB_CHANCE_PCT"))  if reb else 0,
        }

    logging.info("Built tracking stats for %d players", len(tracking))
    return tracking


def _build_matchup_delta():
    """
    Matchup quality score per playoff team, derived from the teams cache.
    Uses whichever efficiency metric is available (DEF_RATING from Advanced,
    OPP_PTS from Opponent, or PTS proxy from Base) and compares vs league avg.

    dEFF_delta > 0 → team allows MORE than average → project opponent stats UP
    dEFF_delta < 0 → elite defense → project DOWN
    """
    # Teams cache is built before this runs — wait briefly for it
    teams_cache = None
    for _ in range(4):
        teams_cache = _cache_get("teams")
        if teams_cache and teams_cache.get("teams"):
            break
        time.sleep(8)

    if teams_cache and teams_cache.get("teams"):
        t = teams_cache["teams"]
        # Use whichever defensive metric is available: dEFF > oEFF fallback > None
        teams_raw = {
            abbr: {
                "dEFF": v.get("dEFF") or v.get("oEFF") or None,
                "oEFF": v.get("oEFF") or None,
                "pace": v.get("rsPace") or 100.0,
            }
            for abbr, v in t.items()
        }
        source = "teams_cache"
    else:
        # Direct fetch fallback: try Opponent measure (gives OPP_PTS)
        teams_raw = {}
        source = "none"
        for season_type in ("Playoffs", "Regular Season"):
            for measure in ("Opponent", "Base"):
                try:
                    df = leaguedashteamstats.LeagueDashTeamStats(
                        season=SEASON, season_type_all_star=season_type,
                        per_mode_detailed="PerGame",
                        measure_type_detailed_defense=measure,
                    ).get_data_frames()[0]
                    logging.info("matchup_delta direct [%s/%s] cols:%s rows:%d",
                                 season_type, measure, df.columns.tolist()[:8], len(df))
                    has_id = not df.empty and ("TEAM_ID" in df.columns or "TEAM_ABBREVIATION" in df.columns)
                    if not has_id:
                        continue
                    for _, row in df.iterrows():
                        abbr = (row.get("TEAM_ABBREVIATION")
                                or _TEAM_ID_TO_ABBR.get(int(row.get("TEAM_ID", 0)))
                                or _TEAM_NAME_TO_ABBR.get(row.get("TEAM_NAME", ""))
                                or "UNK")
                        # Advanced → DEF_RATING; Opponent → OPP_PTS; Base → PTS proxy
                        d = row.get("DEF_RATING") or row.get("OPP_PTS") or row.get("PTS") or None
                        o = row.get("OFF_RATING") or row.get("PTS") or None
                        teams_raw[abbr] = {
                            "dEFF": _f(d) if d else None,
                            "oEFF": _f(o) if o else None,
                            "pace": _f(row.get("PACE") or 100.0),
                        }
                    if teams_raw:
                        source = f"{season_type}/{measure}"
                        break
                except Exception as e:
                    logging.warning("matchup_delta direct [%s/%s] failed: %s",
                                    season_type, measure, e)
            if teams_raw:
                break

    if not teams_raw:
        logging.warning("matchup_delta: no team data at all — returning empty")
        return {}

    logging.info("matchup_delta: %d teams from %s", len(teams_raw), source)

    # ── Compute league-average dEFF proxy ────────────────────────────────────
    deffs = [v["dEFF"] for v in teams_raw.values() if v.get("dEFF")]
    oeffs = [v["oEFF"] for v in teams_raw.values() if v.get("oEFF")]
    league_avg_deff = round(sum(deffs) / len(deffs), 2) if deffs else 113.5
    league_avg_oeff = round(sum(oeffs) / len(oeffs), 2) if oeffs else 113.5

    # ── Build delta dict ──────────────────────────────────────────────────────
    delta = {}
    for abbr, stats in teams_raw.items():
        deff = stats.get("dEFF") or league_avg_deff
        oeff = stats.get("oEFF") or league_avg_oeff
        delta[abbr] = {
            "l5_dEFF":     deff,
            "season_dEFF": league_avg_deff,
            "dEFF_delta":  round(deff - league_avg_deff, 2),
            "l5_oEFF":     oeff,
            "season_oEFF": league_avg_oeff,
            "oEFF_delta":  round(oeff - league_avg_oeff, 2),
            "l5_pace":     stats["pace"],
            "season_pace": stats["pace"],
            "gp":          0,
        }

    logging.info("Built matchup delta for %d teams (league avg dEFF=%.1f)",
                 len(delta), league_avg_deff)
    return delta


# ── Background warm-up ────────────────────────────────────────────────────────
def _warmup():
    logging.info("Background warm-up starting...")
    steps = [
        ("players",       lambda: _build_players(),        "players",       lambda d: {"success": True, "players": d, "count": len(d)}),
        ("teams",         lambda: _build_teams(),          "teams",         lambda d: {"success": True, "teams": d}),
        ("splits",        lambda: _build_splits(),         "splits",        lambda d: {"success": True, "splits": d}),
        ("team_defense",  lambda: _build_team_defense(),   "teamDefense",   lambda d: {"success": True, "teamDefense": d}),
        ("scoring",       lambda: _build_scoring(),        "scoring",       lambda d: {"success": True, "scoring": d}),
        ("clutch",        lambda: _build_clutch(),         "clutch",        lambda d: {"success": True, "clutch": d}),
        ("hustle",        lambda: _build_hustle(),         "hustle",        lambda d: {"success": True, "hustle": d}),
        ("tracking",      lambda: _build_tracking(),       "tracking",      lambda d: {"success": True, "tracking": d}),
        ("matchup_delta", lambda: _build_matchup_delta(),  "matchupDelta",  lambda d: {"success": True, "matchupDelta": d}),
    ]
    for cache_key, builder, _, wrapper in steps:
        try:
            data = builder()
            _cache_set(cache_key, wrapper(data))
            _sleep()
        except Exception as e:
            logging.error("Warm-up failed for %s: %s", cache_key, e)
    _warmup_done.set()
    logging.info("Warm-up complete — all endpoints cached and ready.")


threading.Thread(target=_warmup, daemon=True).start()

# Load XGBoost models synchronously at startup — pure local file reads, < 1s.
# Done at module level so global state is visible to all threads/workers.
try:
    _load_xgb_models()
except Exception as _e:
    logging.error("XGBoost model load failed at startup: %s", _e)


# ── Helper: generic cached endpoint builder ───────────────────────────────────
def _cached_endpoint(cache_key, builder, response_key):
    # Hot path: cache hit — return immediately
    cached = _cache_get(cache_key)
    if cached:
        return cached, 200
    # Cache miss — wait for warmup thread to finish (max 240s).
    # This blocks the request thread until the warmup populates the cache,
    # then returns cached data without doing duplicate NBA API work.
    _warmup_done.wait(timeout=240)
    cached = _cache_get(cache_key)
    if cached:
        return cached, 200
    # Warmup finished but this specific key failed — try building on demand
    try:
        data = builder()
        result = {"success": True, response_key: data}
        _cache_set(cache_key, result)
        return result, 200
    except Exception as e:
        logging.error("Error building %s: %s", cache_key, e)
        return {"success": False, "error": str(e)}, 500


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/api/players")
def get_players():
    r, status = _cached_endpoint("players", _build_players, "players")
    return r, status


@app.route("/api/teams")
def get_teams():
    r, status = _cached_endpoint("teams", _build_teams, "teams")
    return r, status


@app.route("/api/splits")
def get_splits():
    r, status = _cached_endpoint("splits", _build_splits, "splits")
    return r, status


@app.route("/api/team-defense")
def get_team_defense():
    r, status = _cached_endpoint("team_defense", _build_team_defense, "teamDefense")
    return r, status


@app.route("/api/scoring")
def get_scoring():
    """PO shot profile: % pts from 3s, paint, FTs, midrange. Weights zone defense by player type."""
    r, status = _cached_endpoint("scoring", _build_scoring, "scoring")
    return r, status


@app.route("/api/clutch")
def get_clutch():
    """PO clutch stats (last 5 min, within 5 pts): PPG/RPG/APG/FG% in pressure situations."""
    r, status = _cached_endpoint("clutch", _build_clutch, "clutch")
    return r, status


@app.route("/api/hustle")
def get_hustle():
    """PO hustle stats: deflections, contested shots, box-outs per game."""
    r, status = _cached_endpoint("hustle", _build_hustle, "hustle")
    return r, status


@app.route("/api/tracking")
def get_tracking():
    """
    PO player tracking — Passing (potential assists, passes, conversion rate)
    + Rebounding (rebound chance %, oreb/dreb chances per game).
    Used for assist conversion regression and rebound chance context.
    """
    r, status = _cached_endpoint("tracking", _build_tracking, "tracking")
    return r, status


@app.route("/api/matchup-delta")
def get_matchup_delta():
    """
    Per-team last-5-game dEFF vs full-PO-season dEFF.
    dEFF_delta > 0 = opponent defense has softened recently → good for scorer.
    dEFF_delta < 0 = opponent defense has tightened recently → bad for scorer.
    """
    r, status = _cached_endpoint("matchup_delta", _build_matchup_delta, "matchupDelta")
    return r, status


# ── Constants for correlation math ───────────────────────────────────────────
_LEAGUE_AVG_AST_CONV   = 0.30   # baseline AST/POTENTIAL_AST
_LEAGUE_AVG_REB_CONV   = 55.0   # % of rebound chances that convert (league avg)
_THREE_PT_RELY_THRESH  = 35.0   # % pts from 3s = "3pt-reliant" shooter
_FG3_ELITE_DEF_THRESH  = -0.015 # fg3VsAvg ≤ this → elite 3pt defense
_MATCHUP_SCALE         = 0.015  # +1.5% projection per +1.0 dEFF point increase
_MATCHUP_CAP           = 0.09   # ±9% max matchup delta swing
_LEAGUE_AVG_PACE       = 96.5   # league-avg PO pace (possessions per 48 min)
_PACE_CAP              = 0.05   # ±5% max pace adjustment
_FORM_BLEND_WEIGHT     = 0.25   # how much L5 divergence counts vs base
_FORM_CAP              = 0.08   # ±8% max recent form swing
_REST_B2B              = -0.03  # 0 days rest = -3% counting stats
_REST_LONG             = 0.015  # 2+ days rest = +1.5%
_SPLIT_BLEND_WEIGHT    = 0.40   # how much home/road delta counts (40%)
_SPLIT_CAP             = 0.06   # ±6% max venue swing
_USG_ELITE_THRESH      = 28.0   # USG% above this = high-usage role
_TS_GOOD_THRESH        = 55.0   # TS% above this = efficient
_TS_BAD_THRESH         = 50.0   # TS% below this = inefficient
_CLUTCH_LIFT_THRESH    = 0.20   # ±20% clutch divergence → meaningful signal
_COUNTING_PROPS = ("points","assists","rebounds","steals","blocks",
                   "pra","pa","pr","ra","three_pointers")
_SCORING_PROPS  = ("points","pra","pa","pr","three_pointers")


# ── Math utility functions ────────────────────────────────────────────────────

def _soft_cap(x: float, cap: float, k: float = 1.5) -> float:
    """
    Smooth asymptotic cap using tanh instead of a hard min/max clamp.
    Replaces all hard-cap patterns: min(cap, max(-cap, x)).

    Formula: f(x) = L * tanh(k * x / L)   where L = cap * 1.2
      • f(0)    = 0           (zero input → zero output)
      • f(cap)  ≈ cap         (at the old hard-cap value, output matches it)
      • f(2cap) ≈ cap * 1.18  (extreme outliers get ~20% headroom, never clamp hard)
      • Sign-preserving, monotonic

    k=1.5 calibrated so f(cap) / L = tanh(1.5/1.2) ≈ 0.848 → output ≈ 1.018*cap.
    """
    if cap <= 0:
        return 0.0
    L = cap * 1.2
    return L * math.tanh(k * x / L)


def _dynamic_weights(po_gp: int, rs_gp: int, has_l5: bool) -> tuple:
    """
    Replace static PO/RS/L5 weights with logarithmic sample-size scaling.

    Design goals:
      • PO weight grows as log(po_gp): more playoff evidence → trust it more
      • RS weight diluted by PO growth but FLOORED at 5% to prevent instability
      • L5 stays fixed at 35% when available (highest predictive value for recent form)

    Returns (po_weight, rs_weight, l5_weight)  — always sum to 1.0.

    Sample walk-through (with L5):
      po_gp=3  → (0.25, 0.40, 0.35)  — early, RS anchors
      po_gp=6  → (0.35, 0.30, 0.35)  — PO earning its weight
      po_gp=12 → (0.46, 0.19, 0.35)  — PO dominant
      po_gp=21 → (0.50, 0.15, 0.35)  — max PO, RS floored
    """
    if po_gp < 3 or rs_gp < 5:
        # Insufficient PO sample — PO-only or RS-only
        if po_gp >= 1:
            return (1.0, 0.0, 0.0)
        return (0.0, 1.0, 0.0)

    if has_l5:
        L5_W = 0.35
        # PO: log-scale anchored at 0.25 when gp=3, caps at 0.50 around gp=21
        po_raw = min(0.50, 0.25 * math.log1p(po_gp) / math.log1p(3))
        rs_w = max(0.05, 1.0 - L5_W - po_raw)
        po_w = 1.0 - L5_W - rs_w
        return (round(po_w, 4), round(rs_w, 4), L5_W)
    else:
        # No L5: PO/RS only. PO log-scales from 0.45 at gp=3 to 0.75 at gp=21+
        po_raw = min(0.75, 0.45 * math.log1p(po_gp) / math.log1p(3))
        rs_w = max(0.05, 1.0 - po_raw)
        po_w = 1.0 - rs_w
        return (round(po_w, 4), round(rs_w, 4), 0.0)


def _resolve_player(name: str, players_cache: dict):
    """Exact match first, then substring, then None."""
    if name in players_cache:
        return name, players_cache[name]
    for k, v in players_cache.items():
        if name in k or k in name:
            return k, v
    return None, None


def _confidence_band(stat_values, projection):
    """
    Variance-aware confidence band from a list of L5 game-by-game stat values.

    Formula:
      mean   = avg(values)
      std    = sqrt( sum((x - mean)^2) / n )
      cv     = std / mean                 (coefficient of variation)
      floor  = max(0, projection - 1σ)    (~68% probability lower bound)
      ceiling= projection + 1σ            (~68% probability upper bound)
      trust  = 100 * (1 - 2*cv)           (clamped 0..100)

    Trust score interpretation:
      ≥ 70 → tight band, projection is reliable
      40-70 → moderate variance, exercise caution
      < 40 → high volatility (Duncan Robinson, role players); projection is noisy

    Returns dict or None when insufficient sample.
    """
    if not stat_values or len(stat_values) < 3:
        return None
    n = len(stat_values)
    mean = sum(stat_values) / n
    if mean <= 0:
        return None
    var = sum((x - mean) ** 2 for x in stat_values) / n
    std = math.sqrt(var)
    cv = std / mean
    floor   = round(max(0.0, projection - std), 2)
    ceiling = round(projection + std, 2)
    trust   = max(0.0, min(100.0, 100.0 * (1.0 - cv * 2.0)))
    return {
        "mean":        round(mean, 2),
        "std":         round(std, 2),
        "cv":          round(cv, 3),
        "floor":       floor,
        "ceiling":     ceiling,
        "trust_score": int(round(trust)),
        "n":           n,
    }


def _knn_select(game_log_ctx: list, stat_key: str, target_min: float | None,
                target_home: int | None, k: int = 15) -> list:
    """
    Select K contextually-similar historical games using Euclidean distance
    on [minutes played, home/away]. Returns list of stat values.

    game_log_ctx: list of game-log dicts (from _game_log_array) containing
                  min, matchup, and stat fields (pts/reb/ast/etc.)
    """
    records = []
    for g in game_log_ctx:
        val = g.get(stat_key)
        if val is None:
            continue
        g_min     = float(g.get("min") or 0)
        g_home    = 1 if "vs." in str(g.get("matchup", "")) else 0
        records.append({"val": float(val), "min": g_min, "home": g_home})

    if len(records) < 3:
        return [r["val"] for r in records]

    if target_min is None or len(records) < k:
        return [r["val"] for r in records]

    t_home = int(bool(target_home)) if target_home is not None else 0.5
    def dist(r):
        d_min  = (r["min"]  - target_min) ** 2
        d_home = ((r["home"] - t_home) * 6) ** 2  # 6 pt weight on venue mismatch
        return math.sqrt(d_min + d_home)

    records.sort(key=dist)
    return [r["val"] for r in records[:k]]


def _monte_carlo(stat_values, projection, book_line, n_sims=10000, seed=None,
                 game_log_ctx=None, stat_key=None, target_min=None, target_home=None):
    """
    Bootstrap Monte Carlo simulation around the projected value.

    When game_log_ctx is provided (full game log with min/matchup context),
    uses KNN feature-space matching to select the K=15 most contextually
    similar historical games as the bootstrap pool — producing a smoother
    eCDF than the sparse L5 sample alone.

    Falls back to plain bootstrap on stat_values when context is unavailable.

    Pipeline:
      1. pool = KNN-selected games (or stat_values fallback)
      2. shift = projection - mean(pool)   → recenter on ML projection
      3. n_sims bootstrap draws; floor at 0
      4. prob_over/under, P10/P25/P50/P75/P90 from sorted sims

    Returns dict or None when sample is too small (< 3 games).
    """
    # Build bootstrap pool — KNN when context available, L5 fallback otherwise
    if game_log_ctx and stat_key and len(game_log_ctx) >= 8:
        arr = _knn_select(game_log_ctx, stat_key, target_min, target_home, k=15)
        knn_n = len(arr)
    else:
        arr = []
        knn_n = 0

    if len(arr) < 3:
        # Fall back to the plain stat_values array
        if not stat_values or len(stat_values) < 3:
            return None
        try:
            arr = [float(v) for v in stat_values if v is not None]
        except (TypeError, ValueError):
            return None
        knn_n = 0

    if len(arr) < 3:
        return None

    sample_mean = sum(arr) / len(arr)
    shift = projection - sample_mean

    rng = random.Random(seed) if seed is not None else random
    sims = [rng.choice(arr) + shift for _ in range(n_sims)]
    # Floor at 0 — counting stats can't be negative
    sims = [max(0.0, s) for s in sims]
    sims.sort()

    def pctl(p):
        idx = max(0, min(n_sims - 1, int(n_sims * p)))
        return round(sims[idx], 2)

    p10, p25, p50, p75, p90 = pctl(.10), pctl(.25), pctl(.50), pctl(.75), pctl(.90)

    prob_over = prob_under = prob_push = None
    if book_line is not None and book_line > 0:
        # For integer lines (e.g. 5), allow a tiny epsilon for "push"
        # Most NBA lines are .5 lines so push is impossible
        is_integer = abs(book_line - round(book_line)) < 1e-6
        if is_integer:
            eps = 0.5
            prob_over  = round(sum(1 for s in sims if s > book_line + eps) / n_sims, 4)
            prob_under = round(sum(1 for s in sims if s < book_line - eps) / n_sims, 4)
            prob_push  = round(max(0.0, 1.0 - prob_over - prob_under), 4)
        else:
            prob_over  = round(sum(1 for s in sims if s > book_line) / n_sims, 4)
            prob_under = round(sum(1 for s in sims if s < book_line) / n_sims, 4)
            prob_push  = 0.0

    # Decimal-odds equivalent: implied fair payout = 1 / prob_over (American: ±)
    fair_odds_over = round(1.0 / prob_over, 3) if prob_over and prob_over > 0 else None
    fair_odds_under = round(1.0 / prob_under, 3) if prob_under and prob_under > 0 else None

    # Edge-from-50: how far is prob_over from a coin-flip? (Vegas vig is usually ~4.5%)
    # Anything > 0.55 with low CV is a strong play.
    edge_from_50 = round((prob_over - 0.5) * 100, 2) if prob_over is not None else None

    return {
        "n_sims":             n_sims,
        "p10":                p10,
        "p25":                p25,
        "p50":                p50,
        "p75":                p75,
        "p90":                p90,
        "implied_fair_line":  p50,
        "prob_over":          prob_over,
        "prob_under":         prob_under,
        "prob_push":          prob_push,
        "fair_odds_over":     fair_odds_over,
        "fair_odds_under":    fair_odds_under,
        "edge_from_50":       edge_from_50,
        "method":             "knn_bootstrap" if knn_n >= 8 else "bootstrap_recentered",
        "knn_n":              knn_n,
    }


def _base_stat(po, rs, prop_type, scoring_row=None, l5_avg=None,
               l5_min=None, own_team_data=None, opp_team_data=None):
    """
    Rate × Minutes baseline — decouples per-minute production from playing time.

    Pipeline:
      1. Resolve stat value for the prop type from PO/RS dicts
      2. Compute per-minute rate (stat / minutes) for PO and RS
      3. Project blended minutes: PO_min × po_w + RS_min × rs_w (pace-adjusted)
      4. Apply _dynamic_weights() for log-scaled sample-size blending
      5. base = blended_per_min_rate × projected_minutes

    Fallback (when minutes data is insufficient < 5 min/game):
      Falls back to per-game average blend using _dynamic_weights().
      This keeps backwards compatibility when l5_min is not provided.

    Returns (base: float, use_rate_base: bool)
      use_rate_base=True  → pace already baked into minutes; Adj 5 should skip.
      use_rate_base=False → per-game fallback; Adj 5 should still fire.
    """
    def _s(d, k): return float(d.get(k) or 0)

    po_gp = int(po.get("gp") or 0)
    rs_gp = int(rs.get("gp") or 0)
    po_min = _s(po, "min")
    rs_min = _s(rs, "min")

    # ── Resolve stat values for this prop type ───────────────────────────────
    if prop_type == "pra":
        po_v = _s(po,"ppg") + _s(po,"rpg") + _s(po,"apg")
        rs_v = _s(rs,"ppg") + _s(rs,"rpg") + _s(rs,"apg")
    elif prop_type == "pa":
        po_v = _s(po,"ppg") + _s(po,"apg")
        rs_v = _s(rs,"ppg") + _s(rs,"apg")
    elif prop_type == "pr":
        po_v = _s(po,"ppg") + _s(po,"rpg")
        rs_v = _s(rs,"ppg") + _s(rs,"rpg")
    elif prop_type == "ra":
        po_v = _s(po,"rpg") + _s(po,"apg")
        rs_v = _s(rs,"rpg") + _s(rs,"apg")
    elif prop_type == "three_pointers":
        pct = float((scoring_row or {}).get("pctPts3pt") or 0)
        po_v = _s(po,"ppg") * (pct / 100) / 3 if pct > 0 else 0
        rs_v = _s(rs,"ppg") * (pct / 100) / 3 if pct > 0 else 0
    else:
        key_map = {"points":"ppg","assists":"apg","rebounds":"rpg",
                   "steals":"spg","blocks":"bpg"}
        k = key_map.get(prop_type, "ppg")
        po_v, rs_v = _s(po, k), _s(rs, k)

    has_l5 = (l5_avg is not None)
    try:
        l5_v = float(l5_avg) if has_l5 else 0.0
        has_l5 = has_l5 and l5_v > 0
    except (TypeError, ValueError):
        has_l5, l5_v = False, 0.0

    # ── Dynamic sample-size weights ──────────────────────────────────────────
    po_w, rs_w, l5_w = _dynamic_weights(po_gp, rs_gp, has_l5)

    # ── Rate × Minutes approach (requires ≥5 min/game in both PO and RS) ─────
    if po_min >= 5.0 and rs_min >= 5.0 and po_v > 0 and rs_v > 0:
        po_pm = po_v / po_min   # stat per minute in playoffs
        rs_pm = rs_v / rs_min   # stat per minute in regular season

        # L5 per-minute rate — requires client to send l5_min alongside l5_avg
        has_l5_min = False
        l5_pm = None
        if has_l5 and l5_min is not None:
            try:
                l5_min_f = float(l5_min)
                if l5_min_f >= 5.0:
                    l5_pm = l5_v / l5_min_f
                    has_l5_min = True
            except (TypeError, ValueError):
                pass

        # Blended per-minute rate
        if has_l5_min and l5_pm is not None and l5_w > 0:
            blended_pm = po_pm * po_w + rs_pm * rs_w + l5_pm * l5_w
        else:
            # Renormalize PO+RS weights without L5 component
            total_w = po_w + rs_w
            blended_pm = (po_pm * po_w + rs_pm * rs_w) / total_w if total_w > 0 else po_pm

        # Project minutes: blend of PO / RS / (optional) L5 minutes
        if has_l5_min and l5_w > 0:
            proj_min = po_min * po_w + rs_min * rs_w + float(l5_min) * l5_w
        else:
            total_w = po_w + rs_w
            proj_min = (po_min * po_w + rs_min * rs_w) / total_w if total_w > 0 else po_min

        # Pace-adjust projected minutes directly (replaces Adj 5 when use_rate_base=True)
        # Faster game pace → marginally more clock time for key players.
        # Effect on minutes is weaker than on counting stats (~30% of full pace signal).
        if own_team_data and opp_team_data:
            own_pace = float(own_team_data.get("rsPace") or _LEAGUE_AVG_PACE)
            opp_pace = float(opp_team_data.get("rsPace") or _LEAGUE_AVG_PACE)
            game_pace = (own_pace + opp_pace) / 2.0
            pace_delta = (game_pace - _LEAGUE_AVG_PACE) / _LEAGUE_AVG_PACE
            min_pace_adj = _soft_cap(pace_delta * 0.30, 0.04)  # sigmoid, max ±4% on minutes
            proj_min = proj_min * (1.0 + min_pace_adj)

        base = round(blended_pm * proj_min, 2)
        return base, True   # use_rate_base=True → caller skips Adj 5

    # ── Fallback: per-game averages with dynamic weights ─────────────────────
    if has_l5 and l5_w > 0:
        base = round(po_v * po_w + rs_v * rs_w + l5_v * l5_w, 2)
    elif po_gp >= 3 and rs_gp >= 5:
        total_w = po_w + rs_w
        base = round((po_v * po_w + rs_v * rs_w) / total_w, 2) if total_w > 0 else round(po_v, 2)
    elif po_gp >= 1:
        base = round(po_v, 2)
    else:
        base = round(rs_v, 2)
    return base, False   # use_rate_base=False → caller lets Adj 5 fire normally


def _xgb_predict(prop_key: str, feature_vals: dict) -> dict | None:
    """
    Run XGBoost inference for one player-game.

    Returns dict with keys:
        point  — Poisson point estimate (primary corr base)
        q25    — 25th-percentile quantile model output (floor)
        q50    — 50th-percentile fair line (median, bypasses parametric adjustment)
        q75    — 75th-percentile (ceiling)
    Or None if model unavailable.
    """
    model = _XGB_MODELS.get(prop_key)
    if model is None or not _XGB_META:
        return None

    feature_cols = _XGB_META.get("feature_cols", [])
    medians      = _XGB_META.get("medians", {})

    row = []
    for col in feature_cols:
        val = feature_vals.get(col)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            val = medians.get(col, 0.0)
        row.append(float(val))

    if not row:
        return None

    try:
        df_row = pd.DataFrame([row], columns=feature_cols)
        point  = max(0.0, round(float(model.predict(df_row)[0]), 2))

        # Quantile predictions (available after retrain; degrade gracefully if absent)
        q_models = _XGB_QUANTILE.get(prop_key, {})
        q25 = max(0.0, round(float(q_models["q25"].predict(df_row)[0]), 2)) if "q25" in q_models else None
        q50 = max(0.0, round(float(q_models["q50"].predict(df_row)[0]), 2)) if "q50" in q_models else None
        q75 = max(0.0, round(float(q_models["q75"].predict(df_row)[0]), 2)) if "q75" in q_models else None

        return {"point": point, "q25": q25, "q50": q50, "q75": q75}
    except Exception as e:
        logging.warning("XGBoost predict failed (%s): %s", prop_key, e)
        return None


def _build_xgb_features(
    player: dict,
    po: dict,
    rs: dict,
    scoring_row: dict | None,
    tracking_row: dict | None,
    opp_def: dict | None,
    own_team_data: dict | None,
    opp_team_data: dict | None,
    l5_avg: float | None,
    l5_min: float | None,
    l5_stat_values: list,
    rest_days: int | None,
    is_home: bool | None,
) -> dict:
    """
    Assemble the feature vector matching the training pipeline columns.
    All values are best-effort from warm caches — None means imputed at inference.
    """
    # L5 averages — prefer client-sent L5 (already computed from /api/recent)
    l5_pts = float(l5_avg)  if l5_avg is not None else float(po.get("ppg") or po.get("pts") or 0) or None
    l5_reb = float(po.get("rpg") or po.get("reb") or 0) or None
    l5_ast = float(po.get("apg") or po.get("ast") or 0) or None
    l5_min_v = float(l5_min) if l5_min is not None else float(po.get("min") or 0) or None

    # L5 TS% from scoring cache
    l5_ts = None
    if scoring_row:
        ts = scoring_row.get("tsPct") or scoring_row.get("ts_pct")
        if ts is not None:
            l5_ts = float(ts)

    # l5_usg removed from feature set — FGA not reliably available at inference.

    # L10 volatility — std of the L5 values passed by client
    l10_pts_std = None
    l10_min_std = None
    if l5_stat_values and len(l5_stat_values) >= 3:
        try:
            vals        = [float(v) for v in l5_stat_values if v is not None]
            l10_pts_std = float(np.std(vals)) if vals else None
        except Exception:
            pass

    # Season-to-date from RS (larger sample than PO early in series)
    # RS cache uses ppg/rpg/apg; PO cache also uses ppg/rpg/apg
    std_pts = float(rs.get("ppg") or rs.get("pts") or po.get("ppg") or po.get("pts") or 0) or None
    std_reb = float(rs.get("rpg") or rs.get("reb") or po.get("rpg") or po.get("reb") or 0) or None
    std_ast = float(rs.get("apg") or rs.get("ast") or po.get("apg") or po.get("ast") or 0) or None
    std_min = float(rs.get("min") or po.get("min") or 0) or None

    gp_prior = int(po.get("gp") or rs.get("gp") or 0)

    # Venue / rest
    is_home_v  = int(bool(is_home)) if is_home is not None else None
    rest_days_v = int(rest_days) if rest_days is not None else None

    # Opponent defensive proxy — pts allowed rolling avg
    opp_def_roll10  = None
    opp_pace_roll10 = None
    if opp_def:
        # team_defense cache stores oppPtsAllowed or dRtg-style metrics
        pts_allowed = opp_def.get("oppPtsAllowed") or opp_def.get("pts_allowed")
        if pts_allowed is not None:
            opp_def_roll10 = float(pts_allowed)
    if opp_team_data:
        pace = opp_team_data.get("pace") or opp_team_data.get("PACE")
        if pace is not None:
            opp_pace_roll10 = float(pace)

    return {
        "l5_pts":        l5_pts,
        "l5_reb":        l5_reb,
        "l5_ast":        l5_ast,
        "l5_min":        l5_min_v,
        "l5_ts":         l5_ts,
        "l10_pts_std":   l10_pts_std,
        "l10_min_std":   l10_min_std,
        "std_pts":       std_pts,
        "std_reb":       std_reb,
        "std_ast":       std_ast,
        "std_min":       std_min,
        "gp_prior":      gp_prior,
        "is_home":       is_home_v,
        "rest_days":     rest_days_v,
        "opp_def_roll10": opp_def_roll10,
        "opp_pace_roll10": opp_pace_roll10,
    }


# ── Live Odds (The Odds API) ──────────────────────────────────────────────────
# Maps frontend prop_type keys → Odds API market strings
_ODDS_MARKET_MAP = {
    "points":         "player_points",
    "rebounds":       "player_rebounds",
    "assists":        "player_assists",
    "three_pointers": "player_threes",
    "steals":         "player_steals",
    "blocks":         "player_blocks",
    "pra":            "player_points_rebounds_assists",
    "pa":             "player_points_assists",
    "pr":             "player_points_rebounds",
    "ra":             "player_rebounds_assists",
}

_PRIMARY_BOOKS = {"draftkings", "fanduel", "betmgm", "caesars"}


def _fetch_odds_slate(market: str):
    """
    Pull the full NBA player-prop slate for one market string from The Odds API.
    Wrapped in a TTLCache (10 min) so 100 player lookups cost ≤ (N_games + 1) API calls.
    Returns list-of-game-dicts or None on any failure.
    """
    api_key = os.environ.get("ODDS_API_KEY", "")
    if not api_key or not _ODDS_AVAILABLE:
        return None
    try:
        # Step 1 — today's NBA events (event IDs required for player-prop endpoint)
        ev_resp = _requests_lib.get(
            "https://api.the-odds-api.com/v4/sports/basketball_nba/events",
            params={"apiKey": api_key},
            timeout=10,
        )
        if not ev_resp.ok:
            logging.warning("Odds API /events %s: %s", ev_resp.status_code, ev_resp.text[:120])
            return None
        events = ev_resp.json()
        if not events:
            return []

        # Step 2 — per-event player-prop odds (one call per game, cached together)
        games = []
        for ev in events[:12]:
            try:
                o = _requests_lib.get(
                    f"https://api.the-odds-api.com/v4/sports/basketball_nba/events/{ev['id']}/odds",
                    params={
                        "apiKey":      api_key,
                        "regions":     "us",
                        "markets":     market,
                        "oddsFormat":  "american",
                    },
                    timeout=10,
                )
                if o.ok:
                    games.append(o.json())
                else:
                    logging.warning("Odds API game %s %s: %s", ev['id'], market, o.status_code)
            except Exception as _ge:
                logging.warning("Odds API game fetch: %s", _ge)
        return games or None
    except Exception as exc:
        logging.warning("Odds API slate error: %s", exc)
        return None


# Apply TTL cache only when cachetools is available
if _ODDS_AVAILABLE:
    _fetch_odds_slate = _ct_cached(_ODDS_CACHE)(_fetch_odds_slate)


@app.route("/api/live-line/<player_name>/<prop_key>", methods=["GET"])
def get_live_line(player_name, prop_key):
    """
    Return the consensus sportsbook line for a player-prop by querying
    the cached Odds API slate. Never hits the API on a per-player basis.
    """
    market = _ODDS_MARKET_MAP.get(prop_key.lower())
    if not market:
        return jsonify({"error": f"unsupported prop '{prop_key}'"}), 400

    slate = _fetch_odds_slate(market)
    if slate is None:
        return jsonify({"error": "odds service unavailable — check ODDS_API_KEY"}), 503

    name_norm = player_name.lower().strip()
    lines = []
    for game in slate:
        for book in (game.get("bookmakers") or []):
            if book.get("key") not in _PRIMARY_BOOKS:
                continue
            for mkt in (book.get("markets") or []):
                for outcome in (mkt.get("outcomes") or []):
                    desc = (outcome.get("description") or outcome.get("name") or "").lower().strip()
                    if desc == name_norm and outcome.get("name", "").lower() in ("over", "under"):
                        pt = outcome.get("point")
                        if pt is not None:
                            lines.append(float(pt))

    if not lines:
        return jsonify({"error": f"no line posted for '{player_name}'"}), 404

    consensus = statistics.median(lines)
    return jsonify({
        "player":         player_name,
        "prop":           prop_key,
        "consensus_line": consensus,
        "books_tracked":  len(lines),
    })


@app.route("/api/project", methods=["POST"])
def post_project():
    """
    ── Correlation Logic Layer ──────────────────────────────────────────────────
    Server-side multi-variate projection engine. Pulls from all warm caches,
    applies four correlation adjustments, and returns a fully-explained output.

    POST body:
        { player_name, prop_type, book_line, opponent_abbr }

    prop_type values: points | assists | rebounds | three_pointers |
                      steals | blocks | pra | pa | pr

    Response:
        base_projection        — PO/RS blended historical average
        correlated_projection  — after all correlation adjustments
        ev_edge                — (correlated / book_line) - 1
        drivers                — ordered list of strings explaining each shift
        breakdown              — raw adjustment values for debugging
        data_quality           — which caches were available
    """
    body         = request.get_json(silent=True) or {}
    player_name  = (body.get("player_name") or "").strip().lower()
    prop_type    = (body.get("prop_type")   or "points").strip().lower()
    opp_abbr     = (body.get("opponent_abbr") or "").strip().upper()
    book_line    = body.get("book_line")
    # Optional context from client (enriches server projection — never fabricated)
    l5_avg       = body.get("l5_avg")        # client-computed L5 PO avg for this prop
    l5_min       = body.get("l5_min")        # client-computed L5 PO minutes/game (from /api/recent)
    l5_stat_values   = body.get("l5_stat_values") or []   # L5 stat values for this prop (variance band)
    game_log_context = body.get("game_log_context") or []  # full PO game log objects for KNN bootstrap
    rest_days    = body.get("rest_days")     # player's team rest days (int)
    team_abbr    = (body.get("team_abbr") or "").strip().upper()  # player's own team
    is_home      = body.get("is_home")       # bool: player playing at home?
    high_leverage = bool(body.get("high_leverage"))  # Game 7, elimination, etc.
    # Residual calibration — client sends stored projection/actual pairs from localStorage
    prior_residuals = body.get("prior_residuals") or []   # [{projected, actual, date, ctx?}, ...]
    current_ctx     = body.get("current_ctx") or {}       # {home, po, b2b, leverage, out: [names]}

    if not player_name:
        return jsonify({"success": False, "error": "player_name is required"}), 400
    try:
        book_line = float(book_line) if book_line is not None else None
    except (TypeError, ValueError):
        book_line = None

    # Wait for warmup (max 180 s — shorter than full 240 s warmup so we fail fast)
    _warmup_done.wait(timeout=180)

    # ── Pull caches ───────────────────────────────────────────────────────────
    players_cache  = (_cache_get("players")       or {}).get("players",     {})
    tracking_cache = (_cache_get("tracking")      or {}).get("tracking",    {})
    matchup_cache  = (_cache_get("matchup_delta") or {}).get("matchupDelta",{})
    scoring_cache  = (_cache_get("scoring")       or {}).get("scoring",     {})
    team_def_cache = (_cache_get("team_defense")  or {}).get("teamDefense", {})
    teams_cache    = (_cache_get("teams")         or {}).get("teams",       {})
    splits_cache   = (_cache_get("splits")        or {}).get("splits",      {})
    clutch_cache   = (_cache_get("clutch")        or {}).get("clutch",      {})

    # ── Resolve player ────────────────────────────────────────────────────────
    resolved_name, player = _resolve_player(player_name, players_cache)
    if not player:
        return jsonify({"success": False,
                        "error": f"Player '{player_name}' not found in cache"}), 404

    po  = player.get("po", {})
    rs  = player.get("rs", po)
    scoring_row  = scoring_cache.get(resolved_name)
    tracking_row = tracking_cache.get(resolved_name)
    splits_row   = splits_cache.get(resolved_name)
    clutch_row   = clutch_cache.get(resolved_name)
    opp_delta    = matchup_cache.get(opp_abbr, {})  if opp_abbr else {}
    opp_def      = team_def_cache.get(opp_abbr, {}) if opp_abbr else {}
    own_team     = team_abbr or player.get("team", "").upper()
    own_team_data = teams_cache.get(own_team, {}) if own_team else {}
    opp_team_data = teams_cache.get(opp_abbr, {}) if opp_abbr else {}

    # ── Base projection — Rate×Minutes with dynamic sample-size weights ───────
    base, use_rate_base = _base_stat(
        po, rs, prop_type, scoring_row,
        l5_avg=l5_avg, l5_min=l5_min,
        own_team_data=own_team_data, opp_team_data=opp_team_data,
    )
    corr = base
    drivers   = []
    breakdown = {}
    breakdown["use_rate_base"] = use_rate_base

    # ── XGBoost base override (replaces Adj 1-13 when models are loaded) ─────
    # When trained model files exist, use ML prediction as the starting point
    # instead of the sequential multiplier cascade. Residual calibration (Adj 14)
    # still fires on top — it corrects any systematic XGBoost bias.
    _xgb_prop_key = _XGB_PROP_MAP.get(prop_type)   # None for composites
    _xgb_used     = False
    if _xgb_prop_key and _XGB_MODELS.get(_xgb_prop_key):
        try:
            feats = _build_xgb_features(
                player, po, rs, scoring_row, tracking_row, opp_def,
                own_team_data, opp_team_data,
                l5_avg, l5_min, l5_stat_values, rest_days, is_home,
            )
            xgb_out = _xgb_predict(_xgb_prop_key, feats)
            if xgb_out is not None:
                xgb_pred = xgb_out["point"]   # Poisson point estimate drives corr
                xgb_q50  = xgb_out.get("q50") # true median fair line (may be None pre-retrain)
                xgb_q25  = xgb_out.get("q25")
                xgb_q75  = xgb_out.get("q75")
            else:
                xgb_pred = xgb_q50 = xgb_q25 = xgb_q75 = None

            if xgb_pred is not None and xgb_pred > 0:
                ev_vs_base = round((xgb_pred / base - 1) * 100, 1) if base else 0
                if base and abs(ev_vs_base) > 35:
                    logging.warning(
                        "XGBoost prediction %.2f rejected — %.1f%% from base %.2f",
                        xgb_pred, ev_vs_base, base,
                    )
                    breakdown["xgb_rejected"] = True
                    breakdown["xgb_rejection_reason"] = f"deviation {ev_vs_base:+.1f}% exceeds ±35% guardrail"
                    breakdown["xgb_raw_pred"] = round(xgb_pred, 3)
                    breakdown["xgb_base_used"] = round(base, 3) if base else None
                    breakdown["xgb_features_debug"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in feats.items()}
                else:
                    corr = xgb_pred
                    _xgb_used = True
                    l5_display = feats.get("l5_pts") or feats.get("l5_reb") or feats.get("l5_ast")
                    l5_str = f"{l5_display:.1f}" if l5_display else "?"
                    opp_str = feats.get("opp_def_roll10") or "?"
                    q_str = f" | q25={xgb_q25} q50={xgb_q50} q75={xgb_q75}" if xgb_q50 else ""
                    drivers.append(
                        f"XGBoost ML Base — gradient-boosted prediction ({xgb_pred:.1f}{q_str}) "
                        f"replaces heuristic cascade. Delta vs historical base: {ev_vs_base:+.1f}% "
                        f"(l5={l5_str}, opp_def={opp_str})."
                    )
                    breakdown["xgb_base"]             = round(xgb_pred, 2)
                    breakdown["xgb_vs_heuristic_pct"] = ev_vs_base
                    breakdown["xgb_q25"]              = xgb_q25
                    breakdown["xgb_q50"]              = xgb_q50   # true fair line
                    breakdown["xgb_q75"]              = xgb_q75
        except Exception as _xe:
            logging.warning("XGBoost inference error: %s", _xe)
    breakdown["xgb_active"] = _xgb_used

    # NOTE (v1 hybrid): When XGBoost is active, Adj 1-12 still run but their
    # weight is additive on top of the ML base rather than the heuristic base.
    # Adj 13 (injury cascade) MUST still run — it encodes real-time injury info
    # that XGBoost's training data cannot know.
    # Adj 14 (residual calibration) MUST still run — personalizes for this matchup.
    # v2: Adj 1-12 gated — XGBoost already encodes these features natively.
    # Only Adj 13 (injury), Adj 15 (blowout), Adj 14 (residual) always fire.
    if not _xgb_used:

        # ─────────────────────────────────────────────────────────────────────────
        # ADJUSTMENT 1 — AST CONVERSION WEIGHT
        # Pull astConvRate from tracking. If < 0.25 (cold) → +0.8 AST (player is
        # not converting chances; mean-reversion says actual assists should rise).
        # If > 0.35 (hot) → -0.5 AST (over-performing the league model; regress).
        # Only fires for assist-bearing props (assists, pa, pra).
        # ─────────────────────────────────────────────────────────────────────────
        ast_conv_delta = 0.0
        if prop_type in ("assists", "pa", "pra", "ra") and tracking_row:
            gp        = int(tracking_row.get("gp") or 0)
            conv_rate = tracking_row.get("astConvRate")
            if conv_rate is not None and gp >= 2:
                if conv_rate < 0.25:
                    ast_conv_delta = +0.8
                    drivers.append(
                        f"AST Conversion COLD — {resolved_name.title()} converts "
                        f"{conv_rate:.0%} of potential assists (< 25% cold threshold). "
                        f"Mean-reversion adds +0.8 AST to projection."
                    )
                elif conv_rate > 0.35:
                    ast_conv_delta = -0.5
                    drivers.append(
                        f"AST Conversion HOT — {resolved_name.title()} converts "
                        f"{conv_rate:.0%} of potential assists (> 35% hot threshold). "
                        f"Regression to mean removes −0.5 AST from projection."
                    )
                else:
                    drivers.append(
                        f"AST Conversion NEUTRAL — {conv_rate:.0%} conversion rate "
                        f"within normal [25–35%] band; no adjustment."
                    )
        breakdown["astConvAdj"] = ast_conv_delta
        corr = round(corr + ast_conv_delta, 2)

        # ─────────────────────────────────────────────────────────────────────────
        # ADJUSTMENT 2 — OPPONENT DEFENSE COMPOSITE (EWMA — replaces old Adj 2 + Adj 12)
        #
        # COLLINEARITY FIX: The old Adj 2 (matchup delta) and Adj 12 (defensive tier)
        # were both derived from the same underlying dEFF signal, applied twice against
        # different baselines (league avg vs 112). This double-counted defensive variance.
        #
        # New approach: single EWMA that blends two TRUE data layers:
        #   • PO dEFF  = opponent's playoff defensive rating (small sample, most recent)
        #   • RS dEFF  = opponent's full regular-season defensive rating (large sample, stable)
        #
        # EWMA alpha scales with opponent's PO game count:
        #   alpha = min(0.60, po_gp / 14 * 0.60)
        #   → 0 PO games:  purely RS (season anchor dominates)
        #   → 7 PO games:  ~30% PO, 70% RS
        #   → 14 PO games: 60% PO, 40% RS (cap — PO never fully crowds out RS)
        #
        # Single shift applied at 1%/pt from 112 baseline, sigmoid-capped at ±8%.
        # Scoring props: full effect. Rebounds/assists: 40% scale.
        # ─────────────────────────────────────────────────────────────────────────
        _DEF_COMPOSITE_PROPS = {
            **{p: 1.0 for p in _SCORING_PROPS},
            "rebounds": 0.40,
            "assists":  0.40,
        }
        def_composite_pct = 0.0
        if prop_type in _DEF_COMPOSITE_PROPS and opp_team_data:
            po_deff_raw  = opp_team_data.get("poDEFF")
            rs_deff_raw  = opp_team_data.get("rsDEFF")
            opp_po_gp    = int(opp_team_data.get("poGP") or 0)

            # Need at least RS dEFF to compute anything meaningful
            if rs_deff_raw:
                rs_deff = float(rs_deff_raw)
                # EWMA alpha: 0→0% PO weight, 14→60% PO weight (capped)
                alpha = min(0.60, opp_po_gp / 14.0 * 0.60) if opp_po_gp > 0 else 0.0
                if po_deff_raw and alpha > 0:
                    po_deff = float(po_deff_raw)
                    composite_deff = alpha * po_deff + (1.0 - alpha) * rs_deff
                else:
                    composite_deff = rs_deff   # pre-playoff or no PO data

                _DEF_COMPOSITE_BASELINE = 112.0   # PO league-average dEFF
                delta_from_baseline = composite_deff - _DEF_COMPOSITE_BASELINE
                scale = _DEF_COMPOSITE_PROPS[prop_type]
                raw_shift = delta_from_baseline * (0.01 / 1.0) * scale  # 1% per dEFF point
                # Sigmoid-capped: ±8% hard headroom (L = 0.096), smooth approach
                def_composite_pct = _soft_cap(raw_shift, 0.08)

                if abs(def_composite_pct) >= 0.003:
                    grade = ("ELITE"   if composite_deff <= 108 else
                             "STRONG"  if composite_deff <= 110 else
                             "AVERAGE" if composite_deff <= 114 else
                             "WEAK")
                    def_abs    = round(corr * def_composite_pct, 2)
                    prop_label = "scoring" if prop_type in _SCORING_PROPS else prop_type
                    drivers.append(
                        f"Defense Composite (EWMA) — {opp_abbr} grades {grade} "
                        f"({composite_deff:.1f} composite dEFF: "
                        f"{alpha*100:.0f}% PO [{po_deff_raw or 'n/a'}] + "
                        f"{(1-alpha)*100:.0f}% RS [{rs_deff:.1f}], {prop_label} effect). "
                        f"Impact: {def_composite_pct*100:+.1f}% ({def_abs:+.2f})."
                    )
        breakdown["defCompositeAdj"] = round(def_composite_pct * 100, 2)
        corr = round(corr * (1.0 + def_composite_pct), 2)

        # ─────────────────────────────────────────────────────────────────────────
        # ADJUSTMENT 3 — SHOT PROFILE ALIGNMENT
        # Points props only. If player scores ≥35% of their pts from 3s AND
        # the opponent's 3pt defense is elite (fg3VsAvg ≤ -1.5%), apply -1.0 pt.
        # Inverse: if opponent leaks 3s (fg3VsAvg ≥ +1.5%), apply +0.5 pt bonus.
        # ─────────────────────────────────────────────────────────────────────────
        shot_delta = 0.0
        if prop_type in ("points", "pra", "pr") and scoring_row and opp_def:
            pct_3pt     = float(scoring_row.get("pctPts3pt") or 0)
            fg3_vs_avg  = float(opp_def.get("fg3VsAvg") or 0)
            is_3pt_guy  = pct_3pt >= _THREE_PT_RELY_THRESH
            elite_def   = fg3_vs_avg <= _FG3_ELITE_DEF_THRESH
            weak_def    = fg3_vs_avg >= 0.015
            if is_3pt_guy and elite_def:
                shot_delta = -1.0
                drivers.append(
                    f"Shot Profile MISMATCH — {resolved_name.title()} derives "
                    f"{pct_3pt:.0f}% of pts from 3s, but {opp_abbr} is an elite "
                    f"3pt defense ({fg3_vs_avg:+.1%} vs league avg). Penalty: −1.0 pt."
                )
            elif is_3pt_guy and weak_def:
                shot_delta = +0.5
                drivers.append(
                    f"Shot Profile BOOST — {resolved_name.title()} derives "
                    f"{pct_3pt:.0f}% of pts from 3s and {opp_abbr} leaks threes "
                    f"({fg3_vs_avg:+.1%} vs league avg). Bonus: +0.5 pts."
                )
        breakdown["shotProfileAdj"] = shot_delta
        corr = round(corr + shot_delta, 2)

        # ─────────────────────────────────────────────────────────────────────────
        # ADJUSTMENT 4 — HUSTLE / REBOUND REALIZATION RATE
        # For rebound props: calculate the player's realization rate
        # (rebChancePct vs 55% league avg). Each 10% above/below avg ≈ ±5%
        # shift in projection. Cap ±8%.
        # ─────────────────────────────────────────────────────────────────────────
        hustle_delta = 0.0
        if prop_type in ("rebounds", "pr", "pra", "ra") and tracking_row:
            gp              = int(tracking_row.get("gp") or 0)
            reb_chance_pct  = float(tracking_row.get("rebChancePct") or 0)
            total_chances   = float((tracking_row.get("orebChance") or 0) +
                                    (tracking_row.get("drebChance") or 0))
            if gp >= 2 and total_chances >= 1.0 and reb_chance_pct > 0:
                rate_vs_league = (reb_chance_pct - _LEAGUE_AVG_REB_CONV) / _LEAGUE_AVG_REB_CONV
                hustle_pct     = _soft_cap(rate_vs_league * 0.5, 0.08)   # sigmoid, was hard ±8%
                hustle_delta   = round(corr * hustle_pct, 2)
                if abs(hustle_pct) >= 0.005:
                    direction = "ABOVE" if rate_vs_league > 0 else "BELOW"
                    drivers.append(
                        f"Rebound Realization Rate — {resolved_name.title()} secures "
                        f"{reb_chance_pct:.1f}% of rebound chances vs "
                        f"{_LEAGUE_AVG_REB_CONV}% league avg "
                        f"({abs(rate_vs_league)*100:.1f}% {direction} avg, "
                        f"{total_chances:.1f} chances/gm). "
                        f"Impact: {hustle_delta:+.2f} reb."
                    )
        breakdown["hustleAdj"] = hustle_delta
        corr = round(corr + hustle_delta, 1)

        # ─────────────────────────────────────────────────────────────────────────
        # ADJUSTMENT 5 — PACE CONTEXT
        # When use_rate_base=True, pace is already baked into projected_minutes inside
        # _base_stat (Rate×Minutes baseline). Firing Adj 5 on top would double-count it.
        # When use_rate_base=False (per-game fallback), fire normally with sigmoid cap.
        # ─────────────────────────────────────────────────────────────────────────
        pace_pct = 0.0
        if prop_type in _COUNTING_PROPS and not use_rate_base:
            own_pace = float(own_team_data.get("rsPace") or 0)
            opp_pace = float(opp_team_data.get("rsPace") or 0)
            if own_pace > 50 and opp_pace > 50:
                game_pace = round((own_pace + opp_pace) / 2, 1)
                delta_pct = (game_pace - _LEAGUE_AVG_PACE) / _LEAGUE_AVG_PACE
                pace_pct  = _soft_cap(delta_pct, _PACE_CAP)   # sigmoid, was hard ±5%
                if abs(pace_pct) >= 0.005:
                    tempo = "FAST" if delta_pct > 0 else "SLOW"
                    pace_abs = round(corr * pace_pct, 2)
                    drivers.append(
                        f"Pace Context — {tempo} game expected ({game_pace} possessions/48 vs "
                        f"{_LEAGUE_AVG_PACE} league avg). Each possession = scoring opportunity. "
                        f"Impact: {pace_pct*100:+.1f}% ({pace_abs:+.2f})."
                    )
        elif use_rate_base and prop_type in _COUNTING_PROPS:
            # Pace already baked into projected minutes in the baseline
            own_pace = float(own_team_data.get("rsPace") or 0)
            opp_pace = float(opp_team_data.get("rsPace") or 0)
            if own_pace > 50 and opp_pace > 50:
                game_pace = round((own_pace + opp_pace) / 2, 1)
                drivers.append(
                    f"Pace Context — {game_pace} possessions/48 already factored into "
                    f"projected minutes (Rate×Minutes baseline active). No separate Adj 5."
                )
        breakdown["paceAdj"] = round(pace_pct * 100, 2)
        corr = round(corr * (1 + pace_pct), 2)

        # ─────────────────────────────────────────────────────────────────────────
        # ADJUSTMENT 6 — RECENT FORM (informational driver, not double-counted)
        # L5 is now baked into the BASELINE itself (35% weight in _base_stat),
        # so we don't apply a SECOND adjustment on top. Instead, expose the L5
        # divergence vs RS+PO blend as an informational driver if meaningful.
        # ─────────────────────────────────────────────────────────────────────────
        form_delta = 0.0
        if l5_avg is not None:
            try:
                l5 = float(l5_avg)
                # Compare L5 vs PO+RS-only blend (without L5) for an info-only "form" signal
                def _s2(d, k): return float(d.get(k) or 0)
                key_map_2 = {"points":"ppg","assists":"apg","rebounds":"rpg",
                             "steals":"spg","blocks":"bpg",
                             "pra":("ppg","rpg","apg"),"pa":("ppg","apg"),"pr":("ppg","rpg"),
                             "ra":("rpg","apg")}
                k2 = key_map_2.get(prop_type)
                if k2 and l5 > 0:
                    if isinstance(k2, tuple):
                        po_v2 = sum(_s2(po, x) for x in k2)
                        rs_v2 = sum(_s2(rs, x) for x in k2)
                    else:
                        po_v2 = _s2(po, k2)
                        rs_v2 = _s2(rs, k2)
                    season_blend = round(po_v2 * 0.65 + rs_v2 * 0.35, 2)
                    if season_blend > 0:
                        divergence = (l5 - season_blend) / season_blend
                        if abs(divergence) >= 0.10:  # only call out meaningful divergence
                            trend = "HOT" if divergence > 0 else "COLD"
                            drivers.append(
                                f"Recent Form — {resolved_name.title()} {trend} over L5 "
                                f"({l5:.1f} vs {season_blend:.1f} season blend, "
                                f"{divergence*100:+.1f}%). Already blended into base "
                                f"projection (L5 = 35% weight)."
                            )
            except (TypeError, ValueError):
                pass
        breakdown["recentFormAdj"] = form_delta  # always 0 now (baked into base)

        # ─────────────────────────────────────────────────────────────────────────
        # ADJUSTMENT 7 — REST DAYS (client-passed)
        # 0 days rest (back-to-back) = -3% counting stats (documented effect).
        # 2+ days rest = +1.5% (full recovery).
        # 1 day rest = baseline (no adjustment).
        # ─────────────────────────────────────────────────────────────────────────
        rest_pct = 0.0
        if rest_days is not None and prop_type in _COUNTING_PROPS:
            try:
                rd = int(rest_days)
                if rd == 0:
                    rest_pct = _REST_B2B
                    label = "BACK-TO-BACK FATIGUE"
                elif rd >= 2:
                    rest_pct = _REST_LONG
                    label = "WELL-RESTED"
                else:
                    label = None
                if label:
                    rest_abs = round(corr * rest_pct, 2)
                    drivers.append(
                        f"Rest Differential — {label} ({rd} days off). "
                        f"Documented physiological effect: {rest_pct*100:+.1f}% ({rest_abs:+.2f})."
                    )
            except (TypeError, ValueError):
                pass
        breakdown["restAdj"] = round(rest_pct * 100, 2)
        corr = round(corr * (1 + rest_pct), 2)

        # ─────────────────────────────────────────────────────────────────────────
        # ADJUSTMENT 8 — HOME / ROAD SPLIT (data-verified)
        # Pulls player's PO home vs road averages from splits cache. If the player
        # has a meaningful split for this prop, apply 40% of the differential.
        # Cap ±6%. Requires client to pass is_home and prop must be home/road relevant.
        # ─────────────────────────────────────────────────────────────────────────
        splits_pct = 0.0
        if splits_row and is_home is not None and prop_type in ("points","assists","rebounds","pra","pa","pr","ra"):
            home = splits_row.get("home") or {}
            road = splits_row.get("road") or {}
            key_map = {
                "points":"ppg", "assists":"apg", "rebounds":"rpg",
                "pra":["ppg","rpg","apg"], "pa":["ppg","apg"], "pr":["ppg","rpg"], "ra":["rpg","apg"],
            }
            k = key_map.get(prop_type)
            h_gp = int(home.get("gp") or 0)
            r_gp = int(road.get("gp") or 0)
            if h_gp >= 1 and r_gp >= 1 and k:
                if isinstance(k, list):
                    h = sum(float(home.get(x) or 0) for x in k)
                    r = sum(float(road.get(x) or 0) for x in k)
                else:
                    h = float(home.get(k) or 0)
                    r = float(road.get(k) or 0)
                if h > 0 and r > 0:
                    venue_avg = h if is_home else r
                    other_avg = r if is_home else h
                    avg = (h + r) / 2
                    delta_pct = (venue_avg - other_avg) / avg if avg > 0 else 0
                    splits_pct = _soft_cap(delta_pct * _SPLIT_BLEND_WEIGHT, _SPLIT_CAP)  # sigmoid, was hard ±6%
                    if abs(splits_pct) >= 0.005:
                        venue = "HOME" if is_home else "ROAD"
                        splits_abs = round(corr * splits_pct, 2)
                        drivers.append(
                            f"Venue Split — {resolved_name.title()} averages {venue_avg:.1f} at {venue} "
                            f"vs {other_avg:.1f} at other venue ({h_gp}H/{r_gp}R PO games). "
                            f"Impact: {splits_pct*100:+.1f}% ({splits_abs:+.2f})."
                        )
        breakdown["splitsAdj"] = round(splits_pct * 100, 2)
        corr = round(corr * (1 + splits_pct), 2)

        # ─────────────────────────────────────────────────────────────────────────
        # ADJUSTMENT 9 — CLUTCH PERFORMANCE (data-verified, high-leverage only)
        # Compares player's clutch per-minute production to regular per-minute rate.
        # Only fires when:
        #   • client flagged high_leverage=true (Game 7, elimination, etc.)
        #   • clutch sample is meaningful (≥1 min/game in clutch situations)
        #   • divergence is significant (≥20% above/below regular rate)
        # ─────────────────────────────────────────────────────────────────────────
        clutch_delta = 0.0
        # Map prop_type → (clutch_key, regular_key, stat_label, abs_cap)
        _CLUTCH_STAT_MAP = {
            "points":         ("ppg", "ppg", "pts", 0.6),
            "pra":            ("ppg", "ppg", "pts", 0.6),
            "pa":             ("ppg", "ppg", "pts", 0.6),
            "pr":             ("ppg", "ppg", "pts", 0.6),
            "three_pointers": ("ppg", "ppg", "pts", 0.6),
            "rebounds":       ("rpg", "rpg", "reb", 0.4),
            "assists":        ("apg", "apg", "ast", 0.3),
            "ra":             ("rpg", "rpg", "reb+ast", 0.4),
        }
        if high_leverage and clutch_row and prop_type in _CLUTCH_STAT_MAP:
            c_key, r_key, stat_lbl, abs_cap = _CLUTCH_STAT_MAP[prop_type]
            c_stat = float(clutch_row.get(c_key) or 0)
            c_min  = float(clutch_row.get("min") or 0)
            c_gp   = int(clutch_row.get("gp") or 0)
            r_stat = float(po.get(r_key) or rs.get(r_key) or 0)
            r_min  = float(po.get("min") or rs.get("min") or 0)
            if c_min >= 1.0 and c_gp >= 2 and r_stat > 0 and r_min > 5:
                c_per_min = c_stat / max(c_min, 0.5)
                r_per_min = r_stat / max(r_min, 0.5)
                lift = (c_per_min - r_per_min) / r_per_min if r_per_min > 0 else 0
                if abs(lift) >= _CLUTCH_LIFT_THRESH:
                    # Apply 50% of the lift, capped by stat type
                    clutch_delta = round(_soft_cap(lift * 0.5, abs_cap), 2)  # sigmoid, was hard ±abs_cap
                    label = "ELEVATES" if lift > 0 else "SHRINKS"
                    drivers.append(
                        f"Clutch Profile (HIGH-LEVERAGE) — {resolved_name.title()} {label} "
                        f"in clutch ({c_per_min*36:.1f} {stat_lbl}/36 vs {r_per_min*36:.1f} regular, "
                        f"{c_gp} clutch games). Game-7 boost: {clutch_delta:+.2f}."
                    )
        breakdown["clutchAdj"] = clutch_delta
        corr = round(corr + clutch_delta, 2)

        # ─────────────────────────────────────────────────────────────────────────
        # ADJUSTMENT 10 — USAGE × EFFICIENCY PROFILE (data-verified)
        # USG% measures % of team plays that end with this player (shot/TO/FT).
        # Combined with TS% (true shooting), this tells us:
        #   • Elite USG (>28%) + good TS (>55%) → +2% (efficient volume scorer)
        #   • Elite USG (>28%) + low TS (<50%)  → -1.5% (volume inefficiency)
        # No adjustment for league-average USG to avoid noise.
        # ─────────────────────────────────────────────────────────────────────────
        usg_pct_adj = 0.0
        po_usg = float(po.get("usg") or 0)
        po_ts  = float(po.get("ts")  or 0)
        if prop_type in _SCORING_PROPS and po_usg > 0:
            if po_usg >= _USG_ELITE_THRESH and po_ts >= _TS_GOOD_THRESH:
                usg_pct_adj = 0.02
                label = "ELITE-VOLUME EFFICIENT"
            elif po_usg >= _USG_ELITE_THRESH and po_ts < _TS_BAD_THRESH:
                usg_pct_adj = -0.015
                label = "HIGH-VOLUME INEFFICIENT"
            else:
                label = None
            if label:
                usg_abs = round(corr * usg_pct_adj, 2)
                drivers.append(
                    f"Usage Profile — {label} ({po_usg:.0f}% USG, {po_ts:.0f}% TS). "
                    f"Impact: {usg_pct_adj*100:+.1f}% ({usg_abs:+.2f})."
                )
        breakdown["usageAdj"] = round(usg_pct_adj * 100, 2)
        corr = round(corr * (1 + usg_pct_adj), 2)

        # ─────────────────────────────────────────────────────────────────────────
        # ADJUSTMENT 11 — POST-SEASON ELEVATION (data-verified, no client context)
        # Some players elevate in playoffs (Jimmy Butler, Jokic, Tatum); others
        # fade. Compares per-game PO production vs RS for the SAME prop type.
        # Apply 30% of divergence as adjustment. Cap ±6%. Requires PO sample ≥3.
        # This signal fires for almost every player — captures playoff-specific form.
        # ─────────────────────────────────────────────────────────────────────────
        elev_pct = 0.0
        po_gp_check = int(po.get("gp") or 0)
        rs_gp_check = int(rs.get("gp") or 0)
        if po_gp_check >= 3 and rs_gp_check >= 5 and prop_type in ("points","assists","rebounds","pra","pa","pr","ra","steals","blocks"):
            key_map = {
                "points":"ppg", "assists":"apg", "rebounds":"rpg",
                "steals":"spg", "blocks":"bpg",
                "pra":("ppg","rpg","apg"), "pa":("ppg","apg"), "pr":("ppg","rpg"), "ra":("rpg","apg"),
            }
            k = key_map.get(prop_type)
            if isinstance(k, tuple):
                po_v = sum(float(po.get(x) or 0) for x in k)
                rs_v = sum(float(rs.get(x) or 0) for x in k)
            else:
                po_v = float(po.get(k) or 0)
                rs_v = float(rs.get(k) or 0)
            if po_v > 0 and rs_v > 0:
                po_lift = (po_v - rs_v) / rs_v
                if abs(po_lift) >= 0.10:  # ≥10% divergence is meaningful
                    elev_pct = _soft_cap(po_lift * 0.30, 0.06)  # sigmoid, was hard ±6%
                    elev_abs = round(corr * elev_pct, 2)
                    label = "ELEVATING" if po_lift > 0 else "FADING"
                    drivers.append(
                        f"Playoff Form — {resolved_name.title()} {label} in PO "
                        f"({po_v:.1f} per game over {po_gp_check} PO games vs "
                        f"{rs_v:.1f} over {rs_gp_check} RS games, {po_lift*100:+.1f}%). "
                        f"Impact: {elev_pct*100:+.1f}% ({elev_abs:+.2f})."
                    )
        breakdown["playoffFormAdj"] = round(elev_pct * 100, 2)
        corr = round(corr * (1 + elev_pct), 2)

        # ─────────────────────────────────────────────────────────────────────────
        # ADJUSTMENT 11b — PLAYOFF DEBUT FACTOR (zero career PO games this season)
        # Players making their first playoff appearance historically outperform
        # regular-season lines. Young/role players especially show adrenaline lift.
        # Observed May 4 2026: Harper +58%, Champagnie +74% vs model in Game 1.
        # Stars and veterans excluded (handled by Adj 9 clutch + 11 PO form).
        # Rules:
        #   • po_gp == 0 (literally no playoff data this season)
        #   • rs_gp >= 10 (not a garbage-time player)
        #   • rs stat for this prop > 0 (has something to project from)
        #   • role player (rs_ppg < 22) — stars already get elite clutch adj
        #   → +5% lift (modest — one game is not a trend)
        # ─────────────────────────────────────────────────────────────────────────
        debut_pct = 0.0
        if po_gp_check == 0 and rs_gp_check >= 10:
            rs_ppg_for_debut = float(rs.get("ppg") or 0)
            if 3.0 <= rs_ppg_for_debut < 22.0:
                debut_pct = 0.05
                debut_abs = round(corr * debut_pct, 2)
                drivers.append(
                    f"Playoff Debut — {resolved_name.title()} has 0 career PO games "
                    f"this season ({rs_gp_check} RS games, {rs_ppg_for_debut:.1f} RS PPG). "
                    f"First-game adrenaline lift: {debut_pct*100:+.1f}% ({debut_abs:+.2f})."
                )
        breakdown["debutAdj"] = round(debut_pct * 100, 2)
        corr = round(corr * (1 + debut_pct), 2)

        # NOTE: Old Adj 12 (Defensive Matchup Type) removed — merged into Adj 2
        # (Defense Composite EWMA). Keeping adjustment number for reference continuity.
        breakdown["defMatchAdj"] = 0.0   # legacy key — now included in defCompositeAdj

    else:
        # XGBoost active — zero out heuristic adjustment breakdown fields
        for _k in ("astConvAdj","defCompositeAdj","shotProfileAdj","hustleAdj",
                   "paceAdj","recentFormAdj","restAdj","splitsAdj","clutchAdj",
                   "usageAdj","playoffFormAdj","debutAdj","defMatchAdj"):
            breakdown[_k] = 0.0

    # ADJUSTMENT 13 — INJURY CASCADE (teammate OUT → usage boost; own GTD → penalty)
    # v6.5.1: Reads from _build_effective_injury_map() instead of static
    # _INJURY_OVERRIDES — so live boxscore clears (auto-cleared playing players)
    # and GTD streak softeners propagate into cascade math correctly. Without
    # this, a player listed in _INJURY_OVERRIDES but auto-cleared by live
    # boxscore would still trigger phantom cascades for their teammates.
    #
    # Freed usage (PPG) redistributes proportionally to remaining players' USG%.
    # This is the #1 real-world edge: sportsbooks are slow to update lines for DNPs.
    #   • Teammate OUT with ≥8 PPG → proportional boost (cap +15%)
    #   • Player themselves is Questionable/Doubtful → conservative -8% penalty
    # ─────────────────────────────────────────────────────────────────────────
    inj_cascade_pct = 0.0
    try:
        effective_inj_map, _ = _build_effective_injury_map()
    except Exception as e:
        logging.warning("effective_injury_map failed, falling back to overrides: %s", e)
        effective_inj_map = dict(_INJURY_OVERRIDES)

    own_inj = (effective_inj_map.get(resolved_name)
               or effective_inj_map.get(player_name)
               or {})
    if own_inj.get("status") in ("Questionable", "Doubtful"):
        inj_cascade_pct = -0.08
        inj_abs = round(corr * inj_cascade_pct, 2)
        drivers.append(
            f"Injury Uncertainty — {resolved_name.title()} is {own_inj['status']} "
            f"({own_inj.get('detail','undisclosed')}). "
            f"GTD penalty: {inj_cascade_pct*100:+.1f}% ({inj_abs:+.2f})."
        )
    elif prop_type in _COUNTING_PROPS and own_team:
        # Determine which stat to track freed production by, threshold, and label
        _CASCADE_STAT_CFG = {
            "points":   ("ppg", "ppg", 8.0,  "PPG"),
            "pra":      ("ppg", "ppg", 8.0,  "PPG"),
            "pa":       ("ppg", "ppg", 8.0,  "PPG"),
            "pr":       ("ppg", "ppg", 8.0,  "PPG"),
            "ra":       ("rpg", "rpg", 4.0,  "RPG"),
            "rebounds": ("rpg", "rpg", 4.0,  "RPG"),
            "assists":  ("apg", "apg", 3.0,  "APG"),
            "steals":   ("spg", "spg", 0.5,  "SPG"),
            "blocks":   ("bpg", "bpg", 0.5,  "BPG"),
            "three_pointers": ("ppg", "ppg", 8.0, "PPG"),
        }
        stat_key, my_stat_key, thresh, stat_lbl = _CASCADE_STAT_CFG.get(
            prop_type, ("ppg", "ppg", 8.0, "PPG")
        )
        full_players = (_cache_get("players") or {}).get("players", {})
        teammates_out, remaining_usg = [], 0.0
        for tname, tdata in full_players.items():
            if tdata.get("team") != own_team or tname == resolved_name:
                continue
            # v6.5.1: read from effective map (live boxscore + ESPN + overrides),
            # not raw _INJURY_OVERRIDES. Auto-cleared players don't trigger cascade.
            t_inj  = effective_inj_map.get(tname) or {}
            t_usg  = float(tdata.get("po", {}).get("usg") or tdata.get("rs", {}).get("usg") or 0)
            t_stat = float(tdata.get("po", {}).get(stat_key) or tdata.get("rs", {}).get(stat_key) or 0)
            # Status from /api/injuries can be "Out", "Out For Season", etc. — match
            # any string starting with "out" (case-insensitive) so live source labels
            # like "nba_live_inactive" with status "Out" match correctly.
            t_status = (t_inj.get("status") or "").strip().lower()
            if t_status.startswith("out") and t_stat >= thresh:
                teammates_out.append({"name": tname, stat_lbl: t_stat, "usg": t_usg})
            else:
                remaining_usg += t_usg
        if teammates_out:
            my_usg   = float(po.get("usg") or rs.get("usg") or 0)
            pool     = remaining_usg + my_usg
            if my_usg > 0 and pool > 0:
                freed    = sum(p[stat_lbl] for p in teammates_out)
                share    = my_usg / pool
                boost    = freed * share
                my_stat  = float(po.get(my_stat_key) or rs.get(my_stat_key) or 1)
                raw_pct  = boost / max(my_stat, 1)
                inj_cascade_pct = max(0.0, _soft_cap(raw_pct, 0.20))  # sigmoid, bumped to +20% (was 0.15) — sportsbooks slow on DNPs
                if inj_cascade_pct >= 0.01:
                    boost_abs = round(corr * inj_cascade_pct, 2)
                    out_names = ", ".join(p["name"].title() for p in teammates_out[:3])
                    drivers.append(
                        f"Injury Cascade — {out_names} OUT ({freed:.1f} freed {stat_lbl}). "
                        f"{resolved_name.title()} absorbs ~{share*100:.0f}% of load "
                        f"(USG {my_usg:.0f}%). Boost: {inj_cascade_pct*100:+.1f}% ({boost_abs:+.2f})."
                    )
    breakdown["injCascadeAdj"] = round(inj_cascade_pct * 100, 2)
    corr = round(corr * (1 + inj_cascade_pct), 2)

    # ─────────────────────────────────────────────────────────────────────────
    # ADJUSTMENT 15 — BLOWOUT DISCOUNT (high-usage stars sit late in blowouts)
    # When a star (USG ≥ 25%) is on a heavily favored team (net rating diff
    # ≥ +6 pts), they sit more in the 4th. Apply minute discount.
    # Targets the SGA-18, Cade-23, Reaves-8 type misses where the model
    # over-projects because it can't see end-of-game garbage time.
    # No effect on non-stars (low USG keeps you on the floor either way).
    # ─────────────────────────────────────────────────────────────────────────
    blowout_pct = 0.0
    if (prop_type in _COUNTING_PROPS and own_team_data and opp_team_data):
        star_usg = float(po.get("usg") or rs.get("usg") or 0)
        if star_usg >= 25.0:
            own_oEFF = float(own_team_data.get("rsOEFF") or 113.0)
            own_dEFF = float(own_team_data.get("rsDEFF") or 113.0)
            opp_oEFF = float(opp_team_data.get("rsOEFF") or 113.0)
            opp_dEFF = float(opp_team_data.get("rsDEFF") or 113.0)
            own_net  = own_oEFF - own_dEFF
            opp_net  = opp_oEFF - opp_dEFF
            net_diff = own_net - opp_net   # >0 → own team favored
            if net_diff >= 6.0:
                # Linear ramp: -1.5% at +6, -3.0% at +12+
                raw_pct = -0.015 * min(2.0, net_diff / 6.0)
                blowout_pct = _soft_cap(raw_pct, 0.04)   # sigmoid, max ±4%
                if abs(blowout_pct) >= 0.005:
                    blowout_abs = round(corr * blowout_pct, 2)
                    drivers.append(
                        f"Blowout Risk — {own_team} favored by net {net_diff:+.1f} "
                        f"(own {own_net:+.1f} vs opp {opp_net:+.1f}). "
                        f"Star ({star_usg:.0f}% USG) likely sits late. "
                        f"Minutes discount: {blowout_pct*100:+.1f}% ({blowout_abs:+.2f})."
                    )
    breakdown["blowoutAdj"] = round(blowout_pct * 100, 2)
    corr = round(corr * (1 + blowout_pct), 2)

    # ─────────────────────────────────────────────────────────────────────────
    # ADJUSTMENT 14 — RESIDUAL CALIBRATION (model learns from historical errors)
    # Client sends prior_residuals: [{projected, actual, date, ctx?}].
    # Computes weighted mean bias (actual - projected / projected) and corrects it.
    #
    # CONTEXT-AWARE BUCKETING (v6.4):
    #   When residuals carry `ctx` metadata, score each one's similarity to the
    #   current game's context. If we have ≥3 context-similar samples, use ONLY
    #   those (more accurate calibration); else fall back to global mean.
    #
    # Why: if the model over-projects road games but is dead-on at home, a
    # global average dilutes both signals. Bucketing applies the right
    # correction only when conditions match.
    #
    # Similarity scoring (max ~6.0):
    #   home match    +1.5
    #   po match      +1.5
    #   b2b match     +1.0
    #   leverage match +1.0
    #   teammate-OUT overlap +1.0 per match (capped 2.0)
    # Threshold for "similar context" = 2.5 → at least 2 strong matches
    # ─────────────────────────────────────────────────────────────────────────

    def _ctx_similarity(sample_ctx, cur_ctx):
        """Returns 0..6.0 score; higher = more similar context."""
        if not isinstance(sample_ctx, dict) or not isinstance(cur_ctx, dict):
            return 0.0
        s = 0.0
        if sample_ctx.get("home") is not None and cur_ctx.get("home") is not None \
                and bool(sample_ctx["home"]) == bool(cur_ctx["home"]):
            s += 1.5
        if sample_ctx.get("po") is not None and cur_ctx.get("po") is not None \
                and bool(sample_ctx["po"]) == bool(cur_ctx["po"]):
            s += 1.5
        if sample_ctx.get("b2b") is not None and cur_ctx.get("b2b") is not None \
                and bool(sample_ctx["b2b"]) == bool(cur_ctx["b2b"]):
            s += 1.0
        if sample_ctx.get("leverage") is not None and cur_ctx.get("leverage") is not None \
                and bool(sample_ctx["leverage"]) == bool(cur_ctx["leverage"]):
            s += 1.0
        s_out = set((sample_ctx.get("out") or []))
        c_out = set((cur_ctx.get("out") or []))
        if s_out and c_out:
            overlap = len(s_out & c_out)
            s += min(2.0, overlap * 1.0)
        return s

    residual_pct  = 0.0
    residual_n    = min(len(prior_residuals), 20)
    bucket_used   = "none"
    bucket_n      = 0
    SIM_THRESHOLD = 2.5

    if residual_n >= 5:   # need ≥5 total samples before any calibration fires
        try:
            samples = prior_residuals[-15:]   # pool the last 15 for richer signal

            # Bucket: split samples into context-similar vs the rest
            context_similar = []
            if current_ctx:
                for r in samples:
                    sim = _ctx_similarity(r.get("ctx"), current_ctx)
                    if sim >= SIM_THRESHOLD:
                        context_similar.append(r)

            # Choose which pool to calibrate from:
            #   - ≥3 context-similar → use those (sharper signal)
            #   - else use global pool (current behavior, backward compat)
            if len(context_similar) >= 3:
                pool = context_similar
                bucket_used = "context_similar"
            else:
                pool = samples
                bucket_used = "global"
            bucket_n = len(pool)

            weighted_errors = []
            for i, r in enumerate(pool):
                p_val = float(r.get("projected") or 0)
                a_val = float(r.get("actual")    or 0)
                if p_val > 0.5:
                    rel_err = (a_val - p_val) / p_val
                    # Gentle linear recency — old samples still matter (1.0x → 1.0+0.1*i)
                    weight  = 1.0 + i * 0.1
                    weighted_errors.append((rel_err, weight))
            if weighted_errors:
                total_w   = sum(w for _, w in weighted_errors)
                mean_bias = sum(e * w for e, w in weighted_errors) / total_w
                if abs(mean_bias) >= 0.04:   # 4% threshold — only act on systemic bias
                    residual_pct = _soft_cap(mean_bias, 0.08)  # sigmoid, was hard ±8%
                    res_abs      = round(corr * residual_pct, 2)
                    direction    = "under" if mean_bias > 0 else "over"
                    bucket_label = "context-matched" if bucket_used == "context_similar" else "global"
                    drivers.append(
                        f"Residual Learning ({bucket_n}/{residual_n} {bucket_label} samples) — "
                        f"model has historically {direction}-projected this prop by "
                        f"{abs(mean_bias)*100:.1f}% under similar conditions. "
                        f"Calibration: {residual_pct*100:+.1f}% ({res_abs:+.2f})."
                    )
        except (TypeError, ValueError, ZeroDivisionError) as e:
            logging.warning("Residual calibration error: %s", e)
    breakdown["residualCalibAdj"] = round(residual_pct * 100, 2)
    breakdown["residualN"]        = residual_n
    breakdown["residualBucket"]   = bucket_used     # "none" | "context_similar" | "global"
    breakdown["residualBucketN"]  = bucket_n
    corr = round(corr * (1 + residual_pct), 2)

    # ── No signal found ───────────────────────────────────────────────────────
    if not [d for d in drivers if "NEUTRAL" not in d and "no adjustment" not in d.lower()]:
        drivers.append(
            f"No significant correlation signals found for {prop_type}. "
            f"Correlated projection equals blended historical base ({base})."
        )

    # ── EV Edge ───────────────────────────────────────────────────────────────
    ev_edge = round((corr / book_line) - 1, 4) if (book_line and book_line > 0) else None

    # ── Confidence band — variance-aware uncertainty range from L5 game log ───
    confidence_band = None
    monte_carlo     = None
    if l5_stat_values and len(l5_stat_values) >= 3:
        try:
            vals = [float(v) for v in l5_stat_values if v is not None]
            if len(vals) >= 3:
                confidence_band = _confidence_band(vals, corr)
                # Monte Carlo bootstrap — empirical-distribution probabilities
                # Seed by player+prop+date for reproducibility within a single
                # game day (same inputs → same probabilities all night).
                try:
                    seed = hash((resolved_name, prop_type, datetime.now().strftime("%Y%m%d"))) & 0xFFFFFFFF
                    _PROP_STAT_KEY = {
                        "points": "pts", "rebounds": "reb", "assists": "ast",
                        "steals": "stl", "blocks": "blk", "three_pointers": "fg3m",
                        "pra": None, "pa": None, "pr": None, "ra": None,
                    }
                    stat_key = _PROP_STAT_KEY.get(prop_type)
                    monte_carlo = _monte_carlo(
                        vals, corr, book_line, n_sims=10000, seed=seed,
                        game_log_ctx  = game_log_context if stat_key else None,
                        stat_key      = stat_key,
                        target_min    = float(l5_min) if l5_min is not None else None,
                        target_home   = is_home,
                    )
                except Exception as e:
                    logging.warning("Monte Carlo failed for %s/%s: %s", resolved_name, prop_type, e)
        except (TypeError, ValueError) as e:
            logging.warning("Confidence band computation failed: %s", e)

    # ── Book-line gap (model-vs-book disagreement, used in client grading) ────
    book_gap = None
    if book_line and book_line > 0:
        book_gap = round(abs(corr - book_line) / book_line, 4)

    return jsonify({
        "success":               True,
        "player":                resolved_name,
        "team":                  player.get("team", ""),
        "prop":                  prop_type,
        "opponent":              opp_abbr,
        "base_projection":       base,
        "correlated_projection": corr,
        "ev_edge":               ev_edge,
        "book_line":             book_line,
        "book_gap":              book_gap,
        "confidence_band":       confidence_band,
        "monte_carlo":           monte_carlo,
        "drivers":               drivers,
        "breakdown":             breakdown,
        "data_quality": {
            "po_gp":          int(po.get("gp") or 0),
            "rs_gp":          int(rs.get("gp") or 0),
            "has_tracking":   bool(tracking_row),
            "has_matchup":    bool(opp_delta),
            "has_scoring":    bool(scoring_row),
            "has_team_def":   bool(opp_def),
            "xgb_active":     _xgb_used,
            "xgb_models_loaded": list(_XGB_MODELS.keys()),
            "has_splits":     bool(splits_row),
            "has_clutch":     bool(clutch_row),
            "has_pace":       bool(own_team_data and opp_team_data),
            "has_l5":         l5_avg is not None,
            "has_l5_min":     l5_min is not None,
            "has_l5_games":   bool(l5_stat_values),
            "use_rate_base":  use_rate_base,
            "has_rest":       rest_days is not None,
            "is_home":        is_home,
            "high_leverage":  high_leverage,
        },
    })


def _roster_for_team(team_abbr: str) -> list:
    """
    Return player name list for a team.
    Primary: players stats cache (always current when warm).
    Fallback: CommonTeamRoster API + per-team cache (handles cold start &
              new playoff teams not yet in PO stats).
    """
    players_data = (_cache_get("players") or {}).get("players", {})
    names = [name for name, p in players_data.items() if p.get("team") == team_abbr]
    if names:
        return names

    # Stats cache miss — try dedicated roster cache first
    roster_cache_key = f"team_roster_{team_abbr}"
    cached = _cache_get(roster_cache_key)
    if cached is not None:
        return cached

    # Live lookup via CommonTeamRoster
    team_id = _TEAM_ABBR_TO_ID.get(team_abbr)
    if not team_id:
        logging.warning("_roster_for_team: unknown abbr '%s'", team_abbr)
        return []
    try:
        from nba_api.stats.endpoints import commonteamroster
        time.sleep(0.5)  # gentle rate-limit
        ep  = commonteamroster.CommonTeamRoster(team_id=str(team_id), season=SEASON)
        df  = ep.common_team_roster.get_data_frame()
        col = "PLAYER" if "PLAYER" in df.columns else (df.columns[1] if len(df.columns) > 1 else None)
        names = [str(n).lower() for n in (df[col].tolist() if col else []) if n]
        _cache_set(roster_cache_key, names)
        logging.info("_roster_for_team: CommonTeamRoster fetched %d for %s", len(names), team_abbr)
        return names
    except Exception as e:
        logging.warning("_roster_for_team: CommonTeamRoster failed for %s — %s", team_abbr, e)
        _cache_set(roster_cache_key, [])   # cache empty to avoid hammering API on repeat calls
        return []


def _fmt_series(home_wins: int, away_wins: int, home_abbr: str, away_abbr: str) -> str:
    if home_wins == away_wins:
        return f"Series tied {home_wins}-{away_wins}"
    leader = home_abbr if home_wins > away_wins else away_abbr
    return f"{leader} leads {max(home_wins, away_wins)}-{min(home_wins, away_wins)}"


def _fetch_espn_games(date_yyyymmdd, et_label_for_today=None):
    """
    Fetch games from ESPN's public scoreboard API. Unlike NBA stats API,
    ESPN includes CONDITIONAL Game 7s and the future playoff schedule
    before games are officially decided.

    date_yyyymmdd: 'YYYYMMDD' format (e.g., '20260503')
    Returns list of normalized game dicts. Returns [] safely on any error.

    CACHED: per-date, 10 min TTL during pregame/gametime, 1 hour during
    dead overnight hours (via _dynamic_ttl). The schedule endpoint is
    polled every 5 min by the frontend — without this cache, every poll
    hit ESPN twice (today + upcoming). Now hits ESPN ~once per 10 min
    per date, regardless of how many users refresh.
    """
    cache_key = f"espn_scoreboard_{date_yyyymmdd}"
    # ESPN scoreboard for a given date is stable in 10-min chunks during the
    # day; overnight it's totally stable. Use dynamic TTL with 600s base.
    cached = _cache_get(cache_key, ttl=_dynamic_ttl(600))
    if cached is not None:
        return cached

    url = (f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/"
           f"scoreboard?dates={date_yyyymmdd}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        logging.warning("ESPN fetch failed for %s: %s", date_yyyymmdd, e)
        # Cache the empty result briefly to avoid hammering on transient ESPN errors
        _cache_set(cache_key, [])
        return []

    games = []
    for event in (data.get("events") or []):
        try:
            comp_list = event.get("competitions") or []
            if not comp_list:
                continue
            comp = comp_list[0]
            competitors = comp.get("competitors") or []
            home = next((c for c in competitors if c.get("homeAway") == "home"), {}) or {}
            away = next((c for c in competitors if c.get("homeAway") == "away"), {}) or {}

            home_team = home.get("team") or {}
            away_team = away.get("team") or {}
            # Normalize ESPN short abbrs → NBA stats API abbrs (SA→SAS, NY→NYK etc.)
            home_abbr = _norm_abbr(home_team.get("abbreviation") or "")
            away_abbr = _norm_abbr(away_team.get("abbreviation") or "")
            home_full = home_team.get("displayName") or home_abbr
            away_full = away_team.get("displayName") or away_abbr

            # Series state — ESPN provides this even for conditional Game 7s
            series = comp.get("series") or {}
            series_summary = series.get("summary") or ""

            # Extract game number from notes (e.g., "East 1st Round - Game 7")
            notes = comp.get("notes") or []
            note_headline = notes[0].get("headline", "") if notes else ""
            title = "Playoff Game"
            import re as _re
            m = _re.search(r"Game\s+(\d+)", note_headline)
            if m:
                title = f"Game {m.group(1)}"
            elif series.get("currentGameNumber"):
                title = f"Game {series['currentGameNumber']}"

            # Game start time — ESPN returns ISO 8601 UTC
            iso = event.get("date") or ""
            time_str = iso  # client converts UTC→local
            if iso:
                # Convert "2026-05-03T17:00Z" → "May 3, 1:00 PM ET" for client display
                try:
                    dt_utc = datetime.fromisoformat(iso.replace("Z", "+00:00"))
                    dt_et  = dt_utc.astimezone(timezone(timedelta(hours=-4)))
                    time_str = dt_et.strftime("%a %b %-d, %-I:%M %p ET")
                except Exception:
                    # Windows strftime doesn't support %-d; fall back
                    try:
                        dt_utc = datetime.fromisoformat(iso.replace("Z", "+00:00"))
                        dt_et  = dt_utc.astimezone(timezone(timedelta(hours=-4)))
                        time_str = dt_et.strftime("%a %b %d, %I:%M %p ET").replace(" 0", " ")
                    except Exception:
                        time_str = iso

            # Get current scores for in-progress/completed games
            try:
                home_score = int(home.get("score") or 0)
                away_score = int(away.get("score") or 0)
            except (TypeError, ValueError):
                home_score = away_score = None

            status_obj = (event.get("status") or {}).get("type") or {}
            status_text = status_obj.get("shortDetail") or status_obj.get("description") or ""

            entry = {
                "id":        f"espn-{event.get('id','')}",
                "home":      home_abbr,
                "away":      away_abbr,
                "homeTeam":  home_full,
                "awayTeam":  away_full,
                "time":      time_str,
                "title":     title,
                "series":    series_summary,
                "restDays":  {},
                "homeScore": home_score,
                "awayScore": away_score,
                "status":    status_text,
            }
            if home_abbr: entry[home_abbr] = _roster_for_team(home_abbr)
            if away_abbr: entry[away_abbr] = _roster_for_team(away_abbr)
            games.append(entry)
        except Exception as e:
            logging.warning("Skipping malformed ESPN event: %s", e)
            continue

    _cache_set(cache_key, games)
    return games


def _fetch_date_games(date_str):
    """
    Fetch scheduled NBA games for a specific date using stats scoreboard.
    Returns list of game dicts with rosters auto-populated from players cache.
    date_str format: 'MM/DD/YYYY'. Returns [] safely on any error.
    """
    try:
        from nba_api.stats.endpoints import scoreboard as stats_sb
        sb  = stats_sb.ScoreBoard(game_date=date_str, league_id="00")
        hdr = sb.game_header.get_data_frame()
        ls  = sb.line_score.get_data_frame()
    except Exception as e:
        logging.warning("Stats scoreboard fetch failed for %s: %s", date_str, e)
        return []

    # Series standings is optional and often missing/malformed
    ser_idx = {}
    try:
        ser = sb.series_standings.get_data_frame() if hasattr(sb, "series_standings") else None
        if ser is not None and not ser.empty and "GAME_ID" in ser.columns:
            ser_idx = ser.set_index("GAME_ID").to_dict("index")
    except Exception as e:
        logging.warning("Series standings parse failed for %s: %s", date_str, e)

    if hdr is None or hdr.empty:
        return []

    games = []
    for _, row in hdr.iterrows():
        try:
            gid     = str(row.get("GAME_ID", "") or "")
            home_id = int(row.get("HOME_TEAM_ID") or 0)
            away_id = int(row.get("VISITOR_TEAM_ID") or 0)

            # Resolve abbr from line_score; fall back to static lookup if missing
            home_ls = ls[ls["TEAM_ID"] == home_id] if (ls is not None and not ls.empty) else None
            away_ls = ls[ls["TEAM_ID"] == away_id] if (ls is not None and not ls.empty) else None

            def _safe_get(df, col):
                try:
                    return str(df[col].values[0]) if df is not None and not df.empty and col in df.columns else ""
                except Exception:
                    return ""

            home = _safe_get(home_ls, "TEAM_ABBREVIATION") or _TEAM_ID_TO_ABBR.get(home_id, "")
            away = _safe_get(away_ls, "TEAM_ABBREVIATION") or _TEAM_ID_TO_ABBR.get(away_id, "")

            home_city = _safe_get(home_ls, "TEAM_CITY_NAME")
            home_nick = _safe_get(home_ls, "TEAM_NICKNAME")
            away_city = _safe_get(away_ls, "TEAM_CITY_NAME")
            away_nick = _safe_get(away_ls, "TEAM_NICKNAME")
            home_full = (home_city + " " + home_nick).strip() or home
            away_full = (away_city + " " + away_nick).strip() or away

            s = ser_idx.get(gid, {}) or {}
            home_w = int(s.get("HOME_TEAM_WINS") or 0)
            away_w = int(s.get("VISITOR_TEAM_WINS") or 0)
            game_num = home_w + away_w + 1
            series_str = _fmt_series(home_w, away_w, home, away) if (home_w or away_w) else ""

            status = str(row.get("GAME_STATUS_TEXT") or "TBD").strip()

            entry = {
                "id":        gid,
                "home":      home,
                "away":      away,
                "homeTeam":  home_full,
                "awayTeam":  away_full,
                "time":      status,
                "title":     f"Game {game_num}" if game_num <= 7 else "Playoff Game",
                "series":    series_str,
                "restDays":  {},
            }
            if home: entry[home] = _roster_for_team(home)
            if away: entry[away] = _roster_for_team(away)
            games.append(entry)
        except Exception as e:
            logging.warning("Skipping malformed scheduled game: %s", e)
            continue

    return games


@app.route("/api/schedule")
def get_schedule():
    """
    ESPN-only schedule. No NBA stats scoreboard dependency.
    Tonight  = ESPN games for the game-night display date.
    Upcoming = ESPN games for the next date that has games.
    Game-night date: before 6 AM ET use yesterday (games ran past midnight).
    """
    try:
        ET       = timezone(timedelta(hours=-4))
        now_et   = datetime.now(ET)
        # Game-night rollback: before 6 AM, last night is still "tonight"
        display_et    = now_et if now_et.hour >= 6 else (now_et - timedelta(days=1))
        today_iso     = display_et.strftime("%Y%m%d")
        today_label   = display_et.strftime(f"%b {display_et.day}, %Y")
        upcoming_base = display_et + timedelta(days=1)
        logging.info("Schedule: display_et=%s now_et=%s", display_et.date(), now_et.date())
    except Exception as e:
        logging.error("Schedule date setup failed: %s", e)
        return jsonify({"success": True, "today": "Today", "upcomingLabel": "Tomorrow",
                        "games": [], "todayGames": [], "upcomingGames": []})

    # Tonight — ESPN for the game-night date (date-keyed, not clock-keyed)
    today_games = _fetch_espn_games(today_iso)
    logging.info("Tonight (%s): %d games", today_iso, len(today_games))

    # Upcoming — first date after game-night date that has ESPN games
    upcoming_games = []
    upcoming_label = upcoming_base.strftime(f"%b {upcoming_base.day}")
    for offset in range(0, 6):
        d      = upcoming_base + timedelta(days=offset)
        ds_iso = d.strftime("%Y%m%d")
        games  = _fetch_espn_games(ds_iso)
        if games:
            upcoming_games = games
            upcoming_label = d.strftime(f"%b {d.day}")
            logging.info("Upcoming (%s): %d games", ds_iso, len(games))
            break

    return jsonify({
        "success":       True,
        "today":         today_label,
        "todayDate":     today_iso,
        "upcomingLabel": upcoming_label,
        "games":         today_games,
        "todayGames":    today_games,
        "upcomingGames": upcoming_games,
    })


# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC INJURY DETECTION (v6.5-live)
# ─────────────────────────────────────────────────────────────────────────────
# Three new live sources stack on top of ESPN + overrides:
#
#   1. _live_boxscore_status()  — NBA's official live boxscore feed.
#      Once a game's pregame inactive list posts (~1hr before tip), every
#      player carries an explicit status: ACTIVE / INACTIVE. Plus during
#      the game, `played` and `oncourt` flags tell us who has minutes.
#      → AUTHORITATIVE: an INACTIVE in the boxscore is officially OUT.
#      → AUTHORITATIVE: minutes>0 means they're playing (auto-clear).
#
#   2. _gtd_played_streak()  — uses /api/recent cache.
#      If a "Questionable" / "Day-To-Day" player has logged minutes in
#      4+ of their last 5 games, soften status to "Probable" with note.
#
#   3. ESPN live + static overrides  — kept as fallback for players not
#      yet covered by today's boxscore (gives lead-time on injuries that
#      surface earlier in the day before the inactive list posts).
#
# Priority (most authoritative wins):
#   live_boxscore_INACTIVE > live_boxscore_minutes>0 (OUT vs ACTIVE)
#                          > played_today
#                          > _CLEARED_PLAYERS (manual)
#                          > GTD streak softener
#                          > ESPN live
#                          > _INJURY_OVERRIDES (gap-fill)
# ─────────────────────────────────────────────────────────────────────────────

_LIVE_INJURY_TTL = 60  # 1-minute TTL — boxscore changes fast on game day


def _parse_minutes_str(m):
    """Parse 'PT15M30.00S' → 15.5 minutes. Handles missing/empty inputs."""
    if not m or not isinstance(m, str):
        return 0.0
    try:
        if m.startswith("PT"):
            m = m[2:]
        mins = 0.0
        if "M" in m:
            ms, m = m.split("M", 1)
            mins = float(ms or 0)
        if "S" in m:
            ss = m.split("S")[0]
            mins += float(ss or 0) / 60.0
        return mins
    except Exception:
        return 0.0


def _live_boxscore_status():
    """
    Walk every active live game's boxscore and bucket players by status.

    Returns:
        {
            "playing_now":  set(name)  — minutes > 0 right now (definitively playing)
            "active":       set(name)  — status=ACTIVE in pregame inactive list
            "inactive":     {name: reason}  — status=INACTIVE / DNP / NWT
            "team_of":      {name: team_abbr}
        }

    Cached 60 seconds (boxscores change fast during games).
    Returns empty buckets if no live games or API fails.
    """
    cache_key = "_live_box_status"
    with _cache_lock:
        entry = _cache.get(cache_key)
        if entry and (time.time() - entry["ts"]) < _LIVE_INJURY_TTL:
            return entry["data"]

    out = {"playing_now": set(), "active": set(), "inactive": {}, "team_of": {}}

    # Step 1: get today's games via live scoreboard
    try:
        sb = live_scoreboard.ScoreBoard()
        games = sb.get_dict().get("scoreboard", {}).get("games", [])
    except Exception as e:
        logging.warning("live_scoreboard fetch failed: %s", e)
        games = []

    # Step 2: for each game, pull boxscore (only if pregame inactives may have posted)
    # gameStatus: 1=preGame, 2=inProgress, 3=final
    for g in games:
        gid = g.get("gameId")
        gstatus = g.get("gameStatus", 0)
        if not gid:
            continue
        # Skip games > 3 hours away (inactives haven't posted yet)
        # gameStatus=1 means pregame; check if start is within 4 hours
        if gstatus == 1:
            try:
                game_et = g.get("gameTimeUTC") or ""
                # Cheap parse — accept any format; we just need to gate fetch
                # If we can't parse, just try fetching anyway.
                from datetime import datetime as _dt
                gt = _dt.fromisoformat(game_et.replace("Z", "+00:00"))
                hours_until = (gt - datetime.now(timezone.utc)).total_seconds() / 3600
                if hours_until > 4:
                    continue
            except Exception:
                pass

        try:
            bs = live_boxscore.BoxScore(game_id=gid)
            bdata = bs.get_dict().get("game", {})
        except Exception as e:
            logging.warning("live boxscore fetch failed for %s: %s", gid, e)
            continue

        for side in ("homeTeam", "awayTeam"):
            team = bdata.get(side, {}) or {}
            tri  = team.get("teamTricode", "")
            for p in (team.get("players") or []):
                name   = (p.get("name") or "").strip().lower()
                status = (p.get("status") or "").upper()
                played = bool(p.get("played"))
                stats  = p.get("statistics") or {}
                mins   = _parse_minutes_str(stats.get("minutesCalculated") or stats.get("minutes"))
                if not name:
                    continue
                if tri:
                    out["team_of"][name] = tri

                if status == "ACTIVE":
                    out["active"].add(name)
                    if played or mins > 0:
                        out["playing_now"].add(name)
                elif status in ("INACTIVE", "OUT", "DNP", "NWT"):
                    # Reason: player notes if NBA provides them, else generic
                    reason = p.get("notPlayingReason") or p.get("notPlayingDescription") or "Not active (NBA boxscore)"
                    out["inactive"][name] = reason

    with _cache_lock:
        _cache[cache_key] = {"data": out, "ts": time.time()}

    logging.info(
        "live_boxscore_status: playing=%d active=%d inactive=%d (across %d games)",
        len(out["playing_now"]), len(out["active"]), len(out["inactive"]), len(games),
    )
    return out


def _gtd_played_streak(player_name):
    """
    Returns (games_played_in_last_5, total_recent_logged) or (0, 0) if no data.

    Used to soften "Questionable" / "GTD" tags when a player has actually been
    playing through the listed concern. We check the cached gameLog in the
    /api/recent cache (per-PID) — no extra API hit.
    """
    players_data = (_cache_get("players") or {}).get("players", {})
    p = players_data.get(player_name) or {}
    pid = p.get("pid")
    if not pid:
        return (0, 0)

    # Look up the cached recent payload (set by /api/recent endpoint)
    recent_cache = _cache_get(f"recent_{pid}")
    if not recent_cache:
        return (0, 0)
    log = recent_cache.get("gameLog") or []
    if not log:
        return (0, 0)
    last5 = log[-5:]
    played = sum(1 for g in last5 if (g.get("min") or 0) > 0)
    return (played, len(last5))


_EFFECTIVE_INJURY_TTL = 60  # 1-minute cache for the fully-merged map


def _build_effective_injury_map():
    """
    Build the FULLY-MERGED, dynamic injury map.

    This is the single source of truth used by:
      • /api/injuries  (response payload)
      • Adj 13         (injury cascade math during projection)

    Without this, Adj 13 was reading raw _INJURY_OVERRIDES — which meant
    a player auto-cleared by live boxscore was still triggering teammate
    cascades. Now both consumers see the same effective state.

    Source priority (most authoritative wins):
      1. live_boxscore.INACTIVE      → forced OUT
      2. live_boxscore.PLAYING_NOW   → auto-clear (minutes > 0 right now)
      3. live_boxscore.ACTIVE        → auto-clear (officially activated tonight)
      4. _CLEARED_PLAYERS            → manual override of ESPN lag
      5. ESPN live feed              → primary upcoming injuries
      6. _INJURY_OVERRIDES           → static gap-fill for confirmed long-term outs
      7. GTD streak softener         → downgrade Questionable → Probable when 4/5 played

    Returns (merged_dict, diagnostics_dict).
    Cached 60 seconds (boxscores tick fast on game day).
    """
    cache_key = "_effective_injury_map"
    with _cache_lock:
        entry = _cache.get(cache_key)
        if entry and (time.time() - entry["ts"]) < _EFFECTIVE_INJURY_TTL:
            return entry["data"]

    merged = {}
    diag = {"espn_loaded": 0, "live_in": 0, "live_out": 0,
            "live_playing": 0, "auto_cleared": [], "gtd_softened": []}

    # ── Layer 1: ESPN live injury feed (broadest coverage) ────────────────────
    # ESPN returns a nested structure: injuries[] is an array of TEAM objects,
    # each with its own injuries[] array of player entries.
    # Player detail: entry.athlete.displayName / .team.abbreviation
    # Status detail: entry.shortComment (concise) or entry.longComment (verbose)
    try:
        url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        for team_entry in (data.get("injuries") or []):
            for entry in (team_entry.get("injuries") or []):
                try:
                    athlete   = entry.get("athlete") or {}
                    name      = (athlete.get("displayName") or "").strip().lower()
                    status    = (entry.get("status") or "").strip()
                    team_info = (athlete.get("team") or {})
                    team      = _norm_abbr(team_info.get("abbreviation", ""))
                    detail    = (entry.get("shortComment") or
                                 entry.get("longComment") or status)
                    if name and status:
                        merged[name] = {"status": status, "detail": detail,
                                        "team": team, "source": "espn_live"}
                except Exception:
                    continue
        diag["espn_loaded"] = len(merged)
        logging.info("ESPN injuries: loaded %d entries from nested feed", len(merged))
    except Exception as e:
        logging.warning("ESPN injury fetch failed: %s", e)

    # ── Layer 2: static overrides (gap-fill where ESPN has nothing) ───────────
    for name, info in _INJURY_OVERRIDES.items():
        if name not in merged:
            merged[name] = {**info, "source": "override"}

    # ── Layer 3: live boxscore — most authoritative source on game day ────────
    live = _live_boxscore_status()
    diag["live_in"]      = len(live["active"])
    diag["live_out"]     = len(live["inactive"])
    diag["live_playing"] = len(live["playing_now"])

    # 3a. Boxscore says INACTIVE → force OUT (overrides any softer status)
    for name, reason in live["inactive"].items():
        prev = merged.get(name) or {}
        merged[name] = {
            "status": "Out",
            "detail": reason,
            "team":   live["team_of"].get(name, prev.get("team", "")),
            "source": "nba_live_inactive",
            "supersedes": prev.get("source") if prev else None,
        }

    # 3b. Boxscore minutes > 0 → player is literally on the floor → auto-clear
    for name in live["playing_now"]:
        if name in merged:
            diag["auto_cleared"].append({"name": name, "reason": "playing_now"})
            merged.pop(name, None)

    # 3c. Boxscore status=ACTIVE (pregame inactive list posted, player not on it)
    for name in live["active"]:
        if name in merged and name not in live["inactive"]:
            diag["auto_cleared"].append({"name": name, "reason": "boxscore_active"})
            merged.pop(name, None)

    # ── Layer 4: manual cleared list (ESPN lag protection) ────────────────────
    for name in _CLEARED_PLAYERS:
        if name in merged:
            diag["auto_cleared"].append({"name": name, "reason": "manual_clear"})
            merged.pop(name, None)

    # ── Layer 5: GTD streak softener ──────────────────────────────────────────
    # If a player is Questionable but has played 4 of last 5, downgrade to
    # Probable. Doesn't remove the entry — just lowers Adj 13's penalty.
    for name in list(merged.keys()):
        info = merged[name]
        s = (info.get("status") or "").lower()
        if "questionable" in s or "day-to-day" in s or "gtd" in s:
            played, total = _gtd_played_streak(name)
            if played >= 4 and total >= 5:
                merged[name] = {
                    **info,
                    "status": "Probable",
                    "detail": f"{info.get('detail','')} (softened — played {played}/{total} recent)".strip(),
                    "source": (info.get("source") or "") + "+streak_softened",
                }
                diag["gtd_softened"].append({
                    "name": name, "played_l5": played, "of": total,
                })

    result = (merged, diag)
    with _cache_lock:
        _cache[cache_key] = {"data": result, "ts": time.time()}
    return result


@app.route("/api/injuries")
def get_injuries():
    """
    Public injury endpoint — returns the dynamic merged map plus diagnostics.
    Backed by _build_effective_injury_map() which is also consumed by Adj 13.
    """
    merged, diag = _build_effective_injury_map()
    return jsonify({
        "success":     True,
        "injuries":    merged,
        "count":       len(merged),
        "updated":     datetime.now(timezone.utc).strftime("%b %-d, %-I:%M %p ET"),
        "diagnostics": diag,
        "version":     SERVER_VERSION,
    })


@app.route("/api/version")
def get_version():
    return {"version": SERVER_VERSION, "ready": _warmup_done.is_set()}


@app.route("/api/model-status")
def model_status():
    """Report which XGBoost models are loaded and ready for inference."""
    meta_val = _XGB_META.get("validation_results", {})
    xgb_ver = "?"
    try:
        if _xgb_lib:
            xgb_ver = _xgb_lib.__version__
    except Exception:
        pass
    # List the model files we expect
    file_check = {}
    for prop in ("pts", "reb", "ast"):
        path = os.path.join(os.path.dirname(__file__), f"xgb_{prop}_model.json")
        try:
            file_check[prop] = {"path": path, "exists": os.path.exists(path),
                                "size": os.path.getsize(path) if os.path.exists(path) else 0}
        except Exception as e:
            file_check[prop] = {"path": path, "error": str(e)}
    return jsonify({
        "xgb_available":    _XGB_AVAILABLE,
        "xgb_version":      xgb_ver,
        "models_loaded":    list(_XGB_MODELS.keys()),
        "quantile_models":  {p: list(q.keys()) for p, q in _XGB_QUANTILE.items() if q},
        "load_errors":      _XGB_LOAD_ERRORS,
        "file_check":       file_check,
        "feature_cols":     _XGB_META.get("feature_cols", []),
        "train_seasons":    _XGB_META.get("train_seasons", []),
        "val_seasons":      _XGB_META.get("val_seasons", []),
        "validation":       meta_val,
    })


@app.route("/api/debug-schedule")
def debug_schedule():
    """Diagnostic: try MULTIPLE schedule endpoints and report what they return."""
    from nba_api.stats.endpoints import scoreboard as stats_sb
    ET = timezone(timedelta(hours=-4))
    now_et = datetime.now(ET)
    out = {"by_date_scoreboardv2": {}, "endpoints_tried": []}

    # 1. ScoreBoardV2 (stats.endpoints.scoreboard) — the one we currently use
    out["endpoints_tried"].append("nba_api.stats.endpoints.scoreboard.ScoreBoard")
    for offset in range(0, 8):
        date = now_et + timedelta(days=offset)
        ds = date.strftime("%m/%d/%Y")
        try:
            sb = stats_sb.ScoreBoard(game_date=ds, league_id="00")
            hdr = sb.game_header.get_data_frame()
            rows = int(len(hdr)) if hdr is not None else 0
            cols = hdr.columns.tolist()[:6] if hdr is not None and not hdr.empty else []
            sample = {}
            if hdr is not None and not hdr.empty:
                # Convert to JSON-safe dict (drop NaN/Timestamp)
                raw = hdr.iloc[0].to_dict()
                for k, v in raw.items():
                    try:
                        json.dumps(v)
                        sample[k] = v
                    except Exception:
                        sample[k] = str(v)
            out["by_date_scoreboardv2"][ds] = {
                "label": date.strftime("%a %b %d"),
                "rows": rows, "cols": cols, "sample_keys": list(sample.keys())[:10],
            }
        except Exception as e:
            out["by_date_scoreboardv2"][ds] = {"error": str(e), "label": date.strftime("%a %b %d")}

    # 2. Try ScheduleLeagueV2 (the actual schedule API)
    try:
        from nba_api.stats.endpoints import scheduleleaguev2
        out["endpoints_tried"].append("scheduleleaguev2")
        sched = scheduleleaguev2.ScheduleLeagueV2(season=SEASON, league_id="00")
        dfs = sched.get_data_frames()
        out["scheduleleaguev2"] = {
            "frame_count": len(dfs),
            "frames": [{"rows": len(df), "cols": df.columns.tolist()[:10]} for df in dfs[:3]],
        }
    except ImportError:
        out["scheduleleaguev2"] = "module not available in nba_api version"
    except Exception as e:
        out["scheduleleaguev2"] = {"error": str(e), "type": type(e).__name__}

    # 3. Try scheduleleaguev2int
    try:
        import json as json_lib
        from nba_api.stats.endpoints import scheduleleaguev2int
        out["endpoints_tried"].append("scheduleleaguev2int")
        sched = scheduleleaguev2int.ScheduleLeagueV2Int(season=SEASON, league_id="00")
        dfs = sched.get_data_frames()
        out["scheduleleaguev2int"] = {
            "frame_count": len(dfs),
            "frames": [{"rows": len(df), "cols": df.columns.tolist()[:10]} for df in dfs[:3]],
        }
    except ImportError:
        out["scheduleleaguev2int"] = "module not available"
    except Exception as e:
        out["scheduleleaguev2int"] = {"error": str(e), "type": type(e).__name__}

    return jsonify(out)


@app.route("/api/debug-team-defense")
def debug_team_defense():
    """Diagnostic: fetch raw LeagueDashPtTeamDefend and return actual columns + sample row."""
    out = {}
    for cat in ("3 Pointers", "Less Than 6Ft"):
        try:
            df = leaguedashptteamdefend.LeagueDashPtTeamDefend(
                season=SEASON, season_type_all_star="Playoffs",
                defense_category=cat, per_mode_simple="PerGame",
            ).get_data_frames()[0]
            sample = df.iloc[0].to_dict() if not df.empty else {}
            out[cat] = {"rows": len(df), "cols": df.columns.tolist(), "sample": sample}
        except Exception as e:
            out[cat] = {"error": str(e)}
    return jsonify(out)


@app.route("/api/debug-teams")
def debug_teams():
    """Diagnostic: try each measure type and return actual errors + column names."""
    results = {}
    for season_type in ("Playoffs", "Regular Season"):
        for measure in ("Advanced", "Opponent", "Base"):
            key = f"{season_type}/{measure}"
            try:
                df = leaguedashteamstats.LeagueDashTeamStats(
                    season=SEASON, season_type_all_star=season_type,
                    per_mode_detailed="PerGame",
                    measure_type_detailed_defense=measure,
                ).get_data_frames()[0]
                results[key] = {
                    "rows": len(df),
                    "cols": df.columns.tolist()[:15],
                    "has_team_abbr": "TEAM_ABBREVIATION" in df.columns,
                    "sample": df.iloc[0].to_dict() if not df.empty else {},
                }
            except Exception as e:
                results[key] = {"error": str(e), "type": type(e).__name__}
    return jsonify(results)


@app.route("/api/ready")
def get_ready():
    """Lightweight ping — returns {ready: true} once players cache is populated.
    Frontend polls this cheaply before firing all 9 data fetches."""
    players_cached = _cache_get("players") is not None
    return {"ready": players_cached, "warmupDone": _warmup_done.is_set()}


@app.route("/api/cache-status")
def cache_status():
    with _cache_lock:
        status = {}
        for k, v in _cache.items():
            age = time.time() - v["ts"]
            status[k] = {
                "age_seconds": round(age),
                "expires_in": round(_CACHE_TTL - age),
                "valid": age < _CACHE_TTL,
            }
    return status


# ── Game log endpoints (on-demand, not cached) ────────────────────────────────
def _parse_min(val):
    try:
        if isinstance(val, str) and ":" in val:
            parts = val.split(":")
            return round(int(parts[0]) + int(parts[1]) / 60, 1)
        return _f(val)
    except (ValueError, IndexError):
        return 0.0


def _game_log_avg(df):
    def safe_pct(col):
        val = df[col].mean()
        return round(float(val) * 100, 1) if not pd.isna(val) else 0.0
    return {
        "ppg":  _f(df["PTS"].mean()),  "rpg":  _f(df["REB"].mean()),
        "apg":  _f(df["AST"].mean()),  "spg":  _f(df["STL"].mean()),
        "bpg":  _f(df["BLK"].mean()),  "topg": _f(df["TOV"].mean()),
        "fg":   safe_pct("FG_PCT"),    "fg3":  safe_pct("FG3_PCT"),
        "ft":   safe_pct("FT_PCT"),
        "min":  _f(df["MIN"].apply(_parse_min).mean()),
        "gp":   len(df),
    }


def _game_log_array(df):
    """Per-game stat array (newest first) — feeds confidence band + residual auto-fill."""
    def _pct_val(v):
        try:
            return round(float(v) * 100, 1) if v is not None and not pd.isna(v) else 0.0
        except (TypeError, ValueError):
            return 0.0
    out = []
    for _, row in df.iterrows():
        out.append({
            # Core counting stats
            "pts":     _f(row.get("PTS")  or 0),
            "reb":     _f(row.get("REB")  or 0),
            "ast":     _f(row.get("AST")  or 0),
            "stl":     _f(row.get("STL")  or 0),
            "blk":     _f(row.get("BLK")  or 0),
            "tov":     _f(row.get("TOV")  or 0),
            "oreb":    _f(row.get("OREB") or 0),
            "dreb":    _f(row.get("DREB") or 0),
            # Shooting splits (stored as pct, e.g. 53.3)
            "fg_pct":  _pct_val(row.get("FG_PCT")),
            "fg3_pct": _pct_val(row.get("FG3_PCT")),
            "ft_pct":  _pct_val(row.get("FT_PCT")),
            # Raw makes/attempts
            "fgm":     _f(row.get("FGM")  or 0),
            "fga":     _f(row.get("FGA")  or 0),
            "fg3m":    _f(row.get("FG3M") or 0),
            "fg3a":    _f(row.get("FG3A") or 0),
            "ftm":     _f(row.get("FTM")  or 0),
            "fta":     _f(row.get("FTA")  or 0),
            # Game context
            "pm":      _f(row.get("PLUS_MINUS") or 0),
            "min":     _f(_parse_min(row.get("MIN"))),
            "wl":      str(row.get("WL")       or ""),
            "matchup": str(row.get("MATCHUP")  or ""),
            "date":    pd.to_datetime(str(row.get("GAME_DATE") or ""), errors="coerce").strftime("%Y-%m-%d") if row.get("GAME_DATE") else "",
        })
    return out


@app.route("/api/box-results/<int:player_id>")
@app.route("/api/box-results/<int:player_id>/<date_str>")
def get_box_results(player_id, date_str=None):
    """
    Full per-game box score for residual auto-fill.
    Reuses the /api/recent cache when the entry already has extended fields (fg_pct, matchup).
    Falls back to a fresh PlayerGameLog fetch otherwise.
    """
    def _to_iso(d):
        """Normalize 'May 06, 2026' or 'MAY 06, 2026' to '2026-05-06'."""
        try:
            return pd.to_datetime(str(d), errors="coerce").strftime("%Y-%m-%d")
        except Exception:
            return str(d)

    def _find(gl, ds):
        if ds:
            return next((g for g in gl if _to_iso(g.get("date", "")) == ds), None)
        return gl[0] if gl else None

    # Try cache first
    cached = _cache_get(f"recent_{player_id}")
    gl_cached = (cached or {}).get("gameLog", [])
    game = _find(gl_cached, date_str)
    if game and "fg_pct" in game:  # extended format present
        return jsonify({"success": True, "game": game, "source": "cache"})

    # Fetch fresh — grab all PO games so older dates are reachable
    _sleep()
    try:
        logs = playergamelog.PlayerGameLog(
            player_id=player_id, season=SEASON, season_type_all_star="Playoffs",
        ).get_data_frames()[0]
        if logs.empty:
            return jsonify({"success": False, "error": "no playoff games found"}), 404
        gl_full = _game_log_array(logs)
        game = _find(gl_full, date_str)
        if not game:
            label = f" on {date_str}" if date_str else ""
            return jsonify({"success": False, "error": f"no game found{label}"}), 404
        return jsonify({"success": True, "game": game, "source": "fresh"})
    except Exception as e:
        logging.error("box-results error pid=%s: %s", player_id, e)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/recent/<int:player_id>")
def get_recent(player_id):
    # Cache by PID for 30 minutes — game logs only change when a new game finishes.
    # The injury endpoint's GTD streak softener also reads from this cache.
    cache_key = f"recent_{player_id}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    try:
        logs = playergamelog.PlayerGameLog(
            player_id=player_id, season=SEASON, season_type_all_star="Playoffs",
        ).get_data_frames()[0]
        if logs.empty:
            payload = {"success": True, "recent": None, "gp": 0, "gameLog": []}
            _cache_set(cache_key, payload)
            return payload
        recent = logs.head(5)
        payload = {
            "success": True,
            "recent":  _game_log_avg(recent),    # L5 averages (always head 5)
            "gp":      len(logs),
            "gameLog": _game_log_array(recent),  # L5 per-game array (UI / L5 avg)
            "gameLogFull": _game_log_array(logs), # Full PO log for KNN Monte Carlo
        }
        _cache_set(cache_key, payload)
        return payload
    except Exception as e:
        logging.error("Error fetching recent: %s", e)
        return {"success": False, "error": str(e)}, 500


@app.route("/api/vs-opponent/<int:player_id>/<opp_abbr>")
def get_vs_opponent(player_id, opp_abbr):
    """
    Player's historical performance vs a specific opponent (PO + RS).
    CACHED per (pid, opp) — historical matchups only change when a new
    game completes vs that opponent (~once per series). Default dynamic
    TTL is fine: 4hr overnight, 30min near tipoff. Slate of 20 players
    against the same opponent now hits NBA API once per player instead
    of every page load.
    """
    cache_key = f"vsopp_{player_id}_{opp_abbr.upper()}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    try:
        po_logs = playergamelog.PlayerGameLog(
            player_id=player_id, season=SEASON, season_type_all_star="Playoffs",
        ).get_data_frames()[0]
        _sleep()

        vs = po_logs[po_logs["MATCHUP"].str.contains(opp_abbr, na=False, case=False)]
        source = "Playoffs"

        if len(vs) < 2:
            rs_logs = playergamelog.PlayerGameLog(
                player_id=player_id, season=SEASON, season_type_all_star="Regular Season",
            ).get_data_frames()[0]
            rs_vs = rs_logs[rs_logs["MATCHUP"].str.contains(opp_abbr, na=False, case=False)]
            vs = pd.concat([vs, rs_vs])
            source = "PO+RS" if not po_logs.empty else "RS"

        if vs.empty:
            payload = {"success": True, "vsOpponent": None, "gp": 0, "source": None}
            _cache_set(cache_key, payload)
            return payload

        payload = {"success": True, "vsOpponent": _game_log_avg(vs), "gp": len(vs), "source": source}
        _cache_set(cache_key, payload)
        return payload
    except Exception as e:
        logging.error("Error fetching vs-opponent: %s", e)
        return {"success": False, "error": str(e)}, 500


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
