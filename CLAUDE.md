# Claude Code Working Rules — predictive-props-engine (backend)

These rules exist to keep token usage low. Follow them strictly.

## File reading
- **Use Grep, not Read,** to verify existence/location of code. Only Read when you need to edit.
- When Reading any file with >500 lines, **always pass `offset` + `limit`**. Never read whole.
- **Never re-read a file you already saw this session.** Work from memory + Grep.
- `server.py` is the main Flask app. Sub-modules live alongside it (see structure below); look there first.

## Tone
- **No summary tables, no markdown recaps.** 1-3 lines unless explicitly asked.
- After deploys: one line (e.g., `Deployed v6.X.`).
- Skip "verified live" smoke-test commentary unless something failed.
- Don't recap completed work.

## Tool use
- For broad searches, prefer the `Explore` subagent.
- Batch Bash calls with `&&`; batch parallel tool calls in one message.
- Don't dump verbose JSON into context; filter with `python -c "...print only what matters..."`.

## File structure (so you can grep instead of read)
- `server.py` — Flask app, routes, warmup thread, projection endpoint glue
- `injuries.py` — `_CLEARED_PLAYERS`, `_INJURY_OVERRIDES`, live boxscore + ESPN merge logic, `_build_effective_injury_map`, `_gtd_played_streak`, `_live_boxscore_status`
- `cache.py` — `_cache`, `_cache_get`, `_cache_set`, `_dynamic_ttl`, `_cache_lock`
- (Future splits if needed: `projection.py`, `data_builders.py`, `schedule.py`)

## Deployment
- This is a worktree; `git push render deploy-v6:main` deploys to Render.
- After push, poll `https://nba-props-api-43yl.onrender.com/api/version` until version matches.
- Bump `SERVER_VERSION` constant on every deploy.

## Model
- Default to Sonnet for execution, edits, deploys, mechanical refactors.
- Only escalate to Opus for architecture decisions, novel algorithms, hard debugging.

## Caching architecture (don't break)
- `_cache_get(key, ttl=None)` — uses `_dynamic_ttl(_CACHE_TTL)` by default; pass explicit ttl for short-lived caches.
- Injury caches use 60s hardcoded TTL via direct `_cache` dict access (bypass dynamic TTL).
- ESPN scoreboard cached per-date via `_fetch_espn_games`, dynamic TTL on a 600s base.
- `/api/recent` and `/api/vs-opponent` cache by player_id key.
