from flask import Flask, jsonify
from flask_cors import CORS
from nba_api.stats.endpoints import (
    leaguedashplayerstats,
    leaguedashteamstats,
)
from nba_api.live.nba.endpoints import scoreboard as live_scoreboard
import time
import logging

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
CORS(app)

SEASON = "2025-26"


def _sleep():
    time.sleep(0.8)  # nba_api rate limit


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


def _fetch_player_stats(season_type, measure="Base"):
    return leaguedashplayerstats.LeagueDashPlayerStats(
        season=SEASON,
        season_type_all_star=season_type,
        per_mode_detailed="PerGame",
        measure_type_detailed_defense=measure,
    ).get_data_frames()[0]


@app.route("/api/players")
def get_players():
    try:
        po_base = _fetch_player_stats("Playoffs")
        _sleep()
        po_adv = _fetch_player_stats("Playoffs", "Advanced")
        _sleep()
        rs_base = _fetch_player_stats("Regular Season")
        _sleep()
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
                "ppg": _f(row["PTS"]),
                "rpg": _f(row["REB"]),
                "apg": _f(row["AST"]),
                "spg": _f(row["STL"]),
                "bpg": _f(row["BLK"]),
                "topg": _f(row["TOV"]),
                "fg": _pct(row.get("FG_PCT")),
                "fg3": _pct(row.get("FG3_PCT")),
                "ft": _pct(row.get("FT_PCT")),
                "min": _f(row["MIN"]),
                "gp": _i(row["GP"]),
                "usg": _pct(pa.get("USG_PCT")) if pa else None,
                "ts": _pct(pa.get("TS_PCT")) if pa else None,
            }

            if rb:
                rs = {
                    "ppg": _f(rb.get("PTS", 0)),
                    "rpg": _f(rb.get("REB", 0)),
                    "apg": _f(rb.get("AST", 0)),
                    "spg": _f(rb.get("STL", 0)),
                    "bpg": _f(rb.get("BLK", 0)),
                    "topg": _f(rb.get("TOV", 0)),
                    "fg": _pct(rb.get("FG_PCT")),
                    "fg3": _pct(rb.get("FG3_PCT")),
                    "ft": _pct(rb.get("FT_PCT")),
                    "min": _f(rb.get("MIN", 0)),
                    "gp": _i(rb.get("GP", 0)),
                    "usg": _pct(ra.get("USG_PCT")) if ra else None,
                    "ts": _pct(ra.get("TS_PCT")) if ra else None,
                }
            else:
                rs = po  # mid-season trade — no RS data for this team

            players[name] = {
                "team": team,
                "pid": pid,
                "rs": rs,
                "po": po,
            }

        return jsonify({"success": True, "players": players, "count": len(players)})

    except Exception as e:
        logging.error("Error fetching players: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/teams")
def get_teams():
    try:
        rs_adv = leaguedashteamstats.LeagueDashTeamStats(
            season=SEASON,
            season_type_all_star="Regular Season",
            per_mode_detailed="PerGame",
            measure_type_detailed_defense="Advanced",
        ).get_data_frames()[0]
        _sleep()

        po_adv = leaguedashteamstats.LeagueDashTeamStats(
            season=SEASON,
            season_type_all_star="Playoffs",
            per_mode_detailed="PerGame",
            measure_type_detailed_defense="Advanced",
        ).get_data_frames()[0]

        po_idx = po_adv.set_index("TEAM_ABBREVIATION").to_dict("index")

        teams_data = {}
        for _, row in rs_adv.iterrows():
            abbr = row["TEAM_ABBREVIATION"]
            po = po_idx.get(abbr, {})
            teams_data[abbr] = {
                "fullName": row["TEAM_NAME"],
                "rsPace": _f(row.get("PACE", 100.0)),
                "oEFF": _f(po.get("OFF_RATING")) if po else None,
                "dEFF": _f(po.get("DEF_RATING")) if po else None,
                "eDIFF": _f(po.get("NET_RATING")) if po else None,
            }

        return jsonify({"success": True, "teams": teams_data})

    except Exception as e:
        logging.error("Error fetching teams: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/schedule")
def get_schedule():
    try:
        board = live_scoreboard.ScoreBoard()
        games_raw = board.games.get_dict()

        games = []
        for g in games_raw:
            away = g.get("awayTeam", {})
            home = g.get("homeTeam", {})
            games.append({
                "id": g.get("gameId"),
                "away": away.get("teamTricode"),
                "home": home.get("teamTricode"),
                "awayTeam": away.get("teamName"),
                "homeTeam": home.get("teamName"),
                "status": g.get("gameStatusText"),
                "awayScore": away.get("score"),
                "homeScore": home.get("score"),
                "period": g.get("period"),
            })

        return jsonify({"success": True, "games": games})

    except Exception as e:
        logging.error("Error fetching schedule: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
