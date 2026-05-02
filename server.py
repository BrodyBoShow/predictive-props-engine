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
from nba_api.live.nba.endpoints import scoreboard as live_scoreboard
from nba_api.stats.static import teams as nba_teams_static
from datetime import datetime, timezone, timedelta
import pandas as pd
import time
import logging
import threading
import json
import urllib.request
import urllib.error

SERVER_VERSION = "v5.4-multivar"  # 12-factor + ESPN-driven dynamic schedule

# Static TEAM_ID → abbreviation lookup (no API call needed)
_TEAM_ID_TO_ABBR = {t["id"]: t["abbreviation"] for t in nba_teams_static.get_teams()}
_TEAM_NAME_TO_ABBR = {t["full_name"]: t["abbreviation"] for t in nba_teams_static.get_teams()}

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
CORS(app)

SEASON = "2025-26"
_CACHE_TTL = 3600  # 1 hour

# ── In-memory cache ───────────────────────────────────────────────────────────
_cache: dict = {}
_cache_lock = threading.Lock()
# Event set to True once the warmup thread finishes.
# _cached_endpoint waits on this (up to 240s) instead of blocking forever
# or fast-failing with 503 — whichever the warmup finishes first.
_warmup_done = threading.Event()


def _cache_get(key):
    with _cache_lock:
        entry = _cache.get(key)
        if entry and (time.time() - entry["ts"]) < _CACHE_TTL:
            logging.info("Cache HIT: %s (age %.0fs)", key, time.time() - entry["ts"])
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
                   "pra","pa","pr","three_pointers")
_SCORING_PROPS  = ("points","pra","pa","pr","three_pointers")


def _resolve_player(name: str, players_cache: dict):
    """Exact match first, then substring, then None."""
    if name in players_cache:
        return name, players_cache[name]
    for k, v in players_cache.items():
        if name in k or k in name:
            return k, v
    return None, None


def _base_stat(po, rs, prop_type, scoring_row=None):
    """
    Return the weighted blended base stat for the given prop type.
    Blend: PO 65% / RS 35% (PO is more predictive in playoff context).
    Falls back to whichever is available.
    """
    def _s(d, k): return float(d.get(k) or 0)

    if prop_type == "pra":
        po_v = _s(po, "ppg") + _s(po, "rpg") + _s(po, "apg")
        rs_v = _s(rs, "ppg") + _s(rs, "rpg") + _s(rs, "apg")
    elif prop_type == "pa":
        po_v = _s(po, "ppg") + _s(po, "apg")
        rs_v = _s(rs, "ppg") + _s(rs, "apg")
    elif prop_type == "pr":
        po_v = _s(po, "ppg") + _s(po, "rpg")
        rs_v = _s(rs, "ppg") + _s(rs, "rpg")
    elif prop_type == "three_pointers":
        # FG3M ≈ PPG × (pctPts3pt / 100) / 3 — scoring cache required
        pct = float((scoring_row or {}).get("pctPts3pt") or 0)
        po_v = _s(po, "ppg") * (pct / 100) / 3 if pct > 0 else 0
        rs_v = _s(rs, "ppg") * (pct / 100) / 3 if pct > 0 else 0
    else:
        key_map = {"points": "ppg", "assists": "apg", "rebounds": "rpg",
                   "steals": "spg", "blocks": "bpg"}
        k = key_map.get(prop_type, "ppg")
        po_v, rs_v = _s(po, k), _s(rs, k)

    po_gp = int(po.get("gp") or 0)
    rs_gp = int(rs.get("gp") or 0)
    if po_gp >= 3 and rs_gp >= 5:
        return round(po_v * 0.65 + rs_v * 0.35, 2)
    elif po_gp >= 1:
        return round(po_v, 2)
    return round(rs_v, 2)


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
    rest_days    = body.get("rest_days")     # player's team rest days (int)
    team_abbr    = (body.get("team_abbr") or "").strip().upper()  # player's own team
    is_home      = body.get("is_home")       # bool: player playing at home?
    high_leverage = bool(body.get("high_leverage"))  # Game 7, elimination, etc.

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

    # ── Base projection ───────────────────────────────────────────────────────
    base = _base_stat(po, rs, prop_type, scoring_row)
    corr = base
    drivers   = []
    breakdown = {}

    # ─────────────────────────────────────────────────────────────────────────
    # ADJUSTMENT 1 — AST CONVERSION WEIGHT
    # Pull astConvRate from tracking. If < 0.25 (cold) → +0.8 AST (player is
    # not converting chances; mean-reversion says actual assists should rise).
    # If > 0.35 (hot) → -0.5 AST (over-performing the league model; regress).
    # Only fires for assist-bearing props (assists, pa, pra).
    # ─────────────────────────────────────────────────────────────────────────
    ast_conv_delta = 0.0
    if prop_type in ("assists", "pa", "pra") and tracking_row:
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
    # ADJUSTMENT 2 — MATCHUP DELTA (dEFF rolling momentum)
    # For every 1.0 pt increase in the opponent's L5 dEFF vs season dEFF
    # (defense has gotten WORSE recently), increase projection by +1.5%.
    # Negative delta = tightening defense → reduce projection.
    # Only fires for scoring-bearing props. Cap ±9%.
    # ─────────────────────────────────────────────────────────────────────────
    matchup_pct = 0.0
    if prop_type in ("points", "pra", "pa", "pr", "three_pointers") and opp_delta:
        deff_delta = float(opp_delta.get("dEFF_delta") or 0)
        # gp check removed — matchup delta now computed vs league avg (season-long),
        # not rolling L5, so gp is not meaningful as a quality gate.
        if abs(deff_delta) >= 0.5:
            matchup_pct = min(_MATCHUP_CAP, max(-_MATCHUP_CAP, deff_delta * _MATCHUP_SCALE))
            matchup_abs = round(corr * matchup_pct, 2)
            direction   = "above" if deff_delta > 0 else "below"
            quality     = "SOFT" if deff_delta > 0 else "ELITE"
            drivers.append(
                f"Matchup Quality — {opp_abbr} defense is {quality} "
                f"({opp_delta.get('l5_dEFF', '?'):.1f} dEFF, "
                f"{abs(deff_delta):.1f} pts {direction} league avg of "
                f"{opp_delta.get('season_dEFF', '?'):.1f}). "
                f"Impact: {matchup_pct*100:+.1f}% ({matchup_abs:+.2f} pts)."
            )
    breakdown["matchupAdj"] = round(matchup_pct * 100, 2)
    corr = round(corr * (1 + matchup_pct), 2)

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
    if prop_type in ("rebounds", "pr", "pra") and tracking_row:
        gp              = int(tracking_row.get("gp") or 0)
        reb_chance_pct  = float(tracking_row.get("rebChancePct") or 0)
        total_chances   = float((tracking_row.get("orebChance") or 0) +
                                (tracking_row.get("drebChance") or 0))
        if gp >= 2 and total_chances >= 1.0 and reb_chance_pct > 0:
            rate_vs_league = (reb_chance_pct - _LEAGUE_AVG_REB_CONV) / _LEAGUE_AVG_REB_CONV
            hustle_pct     = min(0.08, max(-0.08, rate_vs_league * 0.5))
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
    # ADJUSTMENT 5 — PACE CONTEXT (data-verified)
    # Game pace = (own_team_pace + opp_team_pace) / 2 from teams cache.
    # League PO baseline: ~96.5 possessions per 48 min.
    # More possessions = more opportunities for ALL counting stats.
    # Affects every counting prop linearly. Cap ±5%.
    # ─────────────────────────────────────────────────────────────────────────
    pace_pct = 0.0
    if prop_type in _COUNTING_PROPS:
        own_pace = float(own_team_data.get("rsPace") or 0)
        opp_pace = float(opp_team_data.get("rsPace") or 0)
        if own_pace > 50 and opp_pace > 50:  # sanity-check real values
            game_pace = round((own_pace + opp_pace) / 2, 1)
            delta_pct = (game_pace - _LEAGUE_AVG_PACE) / _LEAGUE_AVG_PACE
            pace_pct = min(_PACE_CAP, max(-_PACE_CAP, delta_pct))
            if abs(pace_pct) >= 0.005:
                tempo = "FAST" if delta_pct > 0 else "SLOW"
                pace_abs = round(corr * pace_pct, 2)
                drivers.append(
                    f"Pace Context — {tempo} game expected ({game_pace} possessions/48 vs "
                    f"{_LEAGUE_AVG_PACE} league avg). Each possession = scoring opportunity. "
                    f"Impact: {pace_pct*100:+.1f}% ({pace_abs:+.2f})."
                )
    breakdown["paceAdj"] = round(pace_pct * 100, 2)
    corr = round(corr * (1 + pace_pct), 2)

    # ─────────────────────────────────────────────────────────────────────────
    # ADJUSTMENT 6 — RECENT FORM / L5 BLEND (client-passed)
    # Client computes L5 PO average from real game logs. Server blends a portion
    # of the divergence into projection. Recent games are 2-3x more predictive
    # than full-season averages for short-term outcomes.
    # Cap ±8%.
    # ─────────────────────────────────────────────────────────────────────────
    form_delta = 0.0
    if l5_avg is not None and base > 0:
        try:
            l5 = float(l5_avg)
            if l5 > 0:
                divergence = (l5 - base) / base
                if abs(divergence) >= 0.05:  # only when ≥5% divergence
                    form_pct = min(_FORM_CAP, max(-_FORM_CAP, divergence * _FORM_BLEND_WEIGHT))
                    form_delta = round(corr * form_pct, 2)
                    trend = "HOT" if divergence > 0 else "COLD"
                    drivers.append(
                        f"Recent Form — {resolved_name.title()} {trend} over L5 "
                        f"({l5:.1f} vs {base:.1f} blended base, {divergence*100:+.1f}%). "
                        f"Recency-weighted blend: {form_delta:+.2f}."
                    )
        except (TypeError, ValueError):
            pass
    breakdown["recentFormAdj"] = form_delta
    corr = round(corr + form_delta, 2)

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
    if splits_row and is_home is not None and prop_type in ("points","assists","rebounds","pra","pa","pr"):
        home = splits_row.get("home") or {}
        road = splits_row.get("road") or {}
        key_map = {
            "points":"ppg", "assists":"apg", "rebounds":"rpg",
            "pra":["ppg","rpg","apg"], "pa":["ppg","apg"], "pr":["ppg","rpg"],
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
                splits_pct = min(_SPLIT_CAP, max(-_SPLIT_CAP, delta_pct * _SPLIT_BLEND_WEIGHT))
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
    if high_leverage and clutch_row and prop_type in _SCORING_PROPS:
        c_ppg = float(clutch_row.get("ppg") or 0)
        c_min = float(clutch_row.get("min") or 0)
        c_gp  = int(clutch_row.get("gp") or 0)
        r_ppg = float(po.get("ppg") or rs.get("ppg") or 0)
        r_min = float(po.get("min") or rs.get("min") or 0)
        if c_min >= 1.0 and c_gp >= 2 and r_ppg > 0 and r_min > 5:
            c_per_min = c_ppg / max(c_min, 0.5)
            r_per_min = r_ppg / max(r_min, 0.5)
            lift = (c_per_min - r_per_min) / r_per_min if r_per_min > 0 else 0
            if abs(lift) >= _CLUTCH_LIFT_THRESH:
                # Apply 50% of the lift, capped at ±0.6 pts in absolute terms
                clutch_delta = round(min(0.6, max(-0.6, lift * 0.5)), 2)
                label = "ELEVATES" if lift > 0 else "SHRINKS"
                drivers.append(
                    f"Clutch Profile (HIGH-LEVERAGE) — {resolved_name.title()} {label} "
                    f"in clutch ({c_per_min*36:.1f} pts/36 vs {r_per_min*36:.1f} regular, "
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
    if po_gp_check >= 3 and rs_gp_check >= 5 and prop_type in ("points","assists","rebounds","pra","pa","pr","steals","blocks"):
        key_map = {
            "points":"ppg", "assists":"apg", "rebounds":"rpg",
            "steals":"spg", "blocks":"bpg",
            "pra":("ppg","rpg","apg"), "pa":("ppg","apg"), "pr":("ppg","rpg"),
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
                elev_pct = min(0.06, max(-0.06, po_lift * 0.30))
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
    # ADJUSTMENT 12 — DEFENSIVE MATCHUP TYPE (data-verified, no client context)
    # For SCORING props, factor in opponent's overall dEFF (separately from
    # the matchup_delta which uses league-avg comparison). This uses the
    # blended RS+PO dEFF for the most stable signal.
    # Elite D (≤108) → -2%; Bad D (≥115) → +2%; cap ±2.5%.
    # ─────────────────────────────────────────────────────────────────────────
    def_match_pct = 0.0
    if prop_type in _SCORING_PROPS and opp_team_data:
        opp_deff = opp_team_data.get("dEFF") or opp_team_data.get("rsDEFF")
        if opp_deff:
            opp_deff = float(opp_deff)
            # Center around 112 (typical PO league avg). Apply 1% per 1.5 dEFF point delta.
            deff_delta_pts = opp_deff - 112.0
            def_match_pct = min(0.025, max(-0.025, deff_delta_pts * (0.01 / 1.5)))
            if abs(def_match_pct) >= 0.005:
                grade = ("ELITE" if opp_deff <= 108 else
                         "STRONG" if opp_deff <= 110 else
                         "AVERAGE" if opp_deff <= 114 else
                         "WEAK")
                def_abs = round(corr * def_match_pct, 2)
                drivers.append(
                    f"Defensive Tier — {opp_abbr} grades {grade} defense "
                    f"({opp_deff:.1f} dEFF, blended RS+PO). "
                    f"Impact: {def_match_pct*100:+.1f}% ({def_abs:+.2f})."
                )
    breakdown["defMatchAdj"] = round(def_match_pct * 100, 2)
    corr = round(corr * (1 + def_match_pct), 2)

    # ── No signal found ───────────────────────────────────────────────────────
    if not [d for d in drivers if "NEUTRAL" not in d and "no adjustment" not in d.lower()]:
        drivers.append(
            f"No significant correlation signals found for {prop_type}. "
            f"Correlated projection equals blended historical base ({base})."
        )

    # ── EV Edge ───────────────────────────────────────────────────────────────
    ev_edge = round((corr / book_line) - 1, 4) if (book_line and book_line > 0) else None

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
        "drivers":               drivers,
        "breakdown":             breakdown,
        "data_quality": {
            "po_gp":          int(po.get("gp") or 0),
            "rs_gp":          int(rs.get("gp") or 0),
            "has_tracking":   bool(tracking_row),
            "has_matchup":    bool(opp_delta),
            "has_scoring":    bool(scoring_row),
            "has_team_def":   bool(opp_def),
            "has_splits":     bool(splits_row),
            "has_clutch":     bool(clutch_row),
            "has_pace":       bool(own_team_data and opp_team_data),
            "has_l5":         l5_avg is not None,
            "has_rest":       rest_days is not None,
            "is_home":        is_home,
            "high_leverage":  high_leverage,
        },
    })


def _roster_for_team(team_abbr: str) -> list:
    """Return list of player names from players cache for a given team abbreviation."""
    players_data = (_cache_get("players") or {}).get("players", {})
    return [name for name, p in players_data.items() if p.get("team") == team_abbr]


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
    """
    url = (f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/"
           f"scoreboard?dates={date_yyyymmdd}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        logging.warning("ESPN fetch failed for %s: %s", date_yyyymmdd, e)
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
            home_abbr = home_team.get("abbreviation") or ""
            away_abbr = away_team.get("abbreviation") or ""
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
    Returns today's games (from live scoreboard) + tomorrow's scheduled games.
    Always returns 200 with safe empty arrays if NBA APIs fail — never crashes
    the frontend, which depends on this for game selection.
    """
    try:
        ET = timezone(timedelta(hours=-4))   # EDT (Apr–Oct)
        now_et = datetime.now(ET)
        today_label    = now_et.strftime(f"%b {now_et.day}, %Y")
        tomorrow_et    = now_et + timedelta(days=1)
        tomorrow_label = tomorrow_et.strftime(f"%b {tomorrow_et.day}")
        today_str      = now_et.strftime("%m/%d/%Y")
        tomorrow_str   = tomorrow_et.strftime("%m/%d/%Y")
    except Exception as e:
        logging.error("Schedule date computation failed: %s", e)
        return jsonify({
            "success": True, "today": "Today", "upcomingLabel": "Tomorrow",
            "games": [], "todayGames": [], "upcomingGames": [],
        })

    # ── Today: try live scoreboard first ───────────────────────────────────
    today_games = []
    try:
        board = live_scoreboard.ScoreBoard()
        try:
            game_dicts = board.games.get_dict()
        except Exception:
            game_dicts = []
        for g in (game_dicts or []):
            try:
                away = g.get("awayTeam") or {}
                home = g.get("homeTeam") or {}
                home_abbr = home.get("teamTricode") or ""
                away_abbr = away.get("teamTricode") or ""
                entry = {
                    "id":        g.get("gameId") or "",
                    "away":      away_abbr,
                    "home":      home_abbr,
                    "awayTeam":  away.get("teamName") or away_abbr,
                    "homeTeam":  home.get("teamName") or home_abbr,
                    "time":      g.get("gameStatusText") or "TBD",
                    "title":     "Playoff Game",
                    "series":    "",
                    "restDays":  {},
                    "awayScore": away.get("score"),
                    "homeScore": home.get("score"),
                    "period":    g.get("period"),
                }
                if home_abbr: entry[home_abbr] = _roster_for_team(home_abbr)
                if away_abbr: entry[away_abbr] = _roster_for_team(away_abbr)
                today_games.append(entry)
            except Exception as e:
                logging.warning("Skipping malformed live game: %s", e)
                continue
    except Exception as e:
        logging.error("Live scoreboard error: %s", e)

    # ── If live scoreboard returned nothing, try ESPN, then NBA stats API ──
    if not today_games:
        today_iso = now_et.strftime("%Y%m%d")
        today_games = _fetch_espn_games(today_iso)
    if not today_games:
        try:
            today_games = _fetch_date_games(today_str)
        except Exception as e:
            logging.warning("Today stats scoreboard fallback failed: %s", e)
            today_games = []

    # ── Upcoming: scan next 4 days; prefer ESPN (has conditional Game 7s) ──
    # ESPN posts the playoff schedule including conditional Game 7s well before
    # NBA stats API officially confirms them. Falls back to NBA stats API.
    upcoming_games = []
    upcoming_date  = tomorrow_str
    upcoming_label = tomorrow_label
    for offset in range(1, 5):
        d = now_et + timedelta(days=offset)
        ds_us  = d.strftime("%m/%d/%Y")
        ds_iso = d.strftime("%Y%m%d")
        # 1. Try ESPN first (best for conditional/predicted games)
        games = _fetch_espn_games(ds_iso)
        # 2. Fall back to NBA stats API
        if not games:
            try:
                games = _fetch_date_games(ds_us)
            except Exception as e:
                logging.warning("NBA stats fallback failed for %s: %s", ds_us, e)
                games = []
        if games:
            upcoming_games = games
            upcoming_date  = ds_us
            upcoming_label = d.strftime(f"%b {d.day}")
            logging.info("Upcoming games found for %s: %d", ds_us, len(games))
            break

    return jsonify({
        "success":       True,
        "today":         today_label,
        "todayDate":     today_str,
        "upcomingLabel": upcoming_label,
        "upcomingDate":  upcoming_date,
        "games":         today_games,
        "todayGames":    today_games,
        "upcomingGames": upcoming_games,
    })


@app.route("/api/version")
def get_version():
    return {"version": SERVER_VERSION, "ready": _warmup_done.is_set()}


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


@app.route("/api/recent/<int:player_id>")
def get_recent(player_id):
    try:
        logs = playergamelog.PlayerGameLog(
            player_id=player_id, season=SEASON, season_type_all_star="Playoffs",
        ).get_data_frames()[0]
        if logs.empty:
            return {"success": True, "recent": None, "gp": 0}
        recent = logs.head(5)
        return {"success": True, "recent": _game_log_avg(recent), "gp": len(recent)}
    except Exception as e:
        logging.error("Error fetching recent: %s", e)
        return {"success": False, "error": str(e)}, 500


@app.route("/api/vs-opponent/<int:player_id>/<opp_abbr>")
def get_vs_opponent(player_id, opp_abbr):
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
            return {"success": True, "vsOpponent": None, "gp": 0, "source": None}

        return {"success": True, "vsOpponent": _game_log_avg(vs), "gp": len(vs), "source": source}
    except Exception as e:
        logging.error("Error fetching vs-opponent: %s", e)
        return {"success": False, "error": str(e)}, 500


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
