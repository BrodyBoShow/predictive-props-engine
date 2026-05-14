"""
train_models.py
───────────────
Train XGBoost regressors for points, rebounds, and assists.

Per prop, trains FOUR parallel models:
  • Poisson median  — primary point estimate (corr base)
  • q=0.25          — RESIDUAL lower bound (offset from Poisson, not absolute)
  • q=0.50          — RESIDUAL median offset (blended 50/50 with Poisson at inference)
  • q=0.75          — RESIDUAL upper bound (offset from Poisson, not absolute)

Quantile models predict the residual error (actual − Poisson) rather than the
raw target. At inference, add the offset onto the final adjusted projection
(post Adj-13 injury cascade + Adj-14 residual calibration) so bands widen
correctly when injury scratches push the base up.

Strict chronological split — train on older seasons, validate on the most
recent season only. This mirrors actual sportsbook conditions.

Usage:
    python train_models.py

Outputs (in the same directory):
    xgb_pts_model.json          (Poisson point estimate)
    xgb_pts_q25.json / q50 / q75
    xgb_reb_model.json  + q25/q50/q75
    xgb_ast_model.json  + q25/q50/q75
    model_meta.json
"""

import json
import sys
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ── Config ────────────────────────────────────────────────────────────────────
TRAINING_FILE  = "training_data.parquet"
TRAIN_SEASONS  = ["2022-23", "2023-24"]
VAL_SEASONS    = ["2024-25"]

FEATURE_COLS = [
    # Flat-window rolling averages
    "l5_pts", "l5_reb", "l5_ast", "l5_min", "l5_ts",
    "l10_pts_std", "l10_min_std",
    "std_pts", "std_reb", "std_ast", "std_min",
    "gp_prior", "is_home", "rest_days",
    # Binary rest-context flags (cleaner tree splits than raw rest_days alone)
    "is_b2b", "is_well_rested",
    "opp_def_roll10", "opp_pace_roll10",
    # EWMA recency features (halflife=3 — recent games weighted 2× vs 3-games-ago)
    "ewma_pts", "ewma_reb", "ewma_ast", "ewma_min",
    # Usage redistribution pool (sum of absent teammates' rolling USG proxy)
    "inactive_usg_pool",
    # Opponent scheme concessions
    "fg3_vs_avg", "rim_vs_avg",
    # Tracking-derived features — prior-season archetypes at training; live values at inference
    # xPPS_base:                  prior-season shot-diet quality (Drives/PullUp/CatchShoot weighted eFG%)
    # efficiency_delta:           l5_ts - xPPS_base  (efficiency regression signal)
    # l5_potential_ast:           prior-season potentialAst/g (creation volume)
    # inactive_potential_ast_pool: sum of absent teammates' prior-season potentialAst/g
    # inactive_drives_pool:        sum of absent teammates' prior-season drives/g
    "xPPS_base",
    "efficiency_delta",
    "l5_potential_ast",
    "inactive_potential_ast_pool",
    "inactive_drives_pool",
    # Context and interaction features
    # leverage_index:        0.0=RS, 1.0=playoffs (game stakes)
    # paint_overlay:         pct_pts_paint × rim_vs_avg (interior archetype × interior defense)
    # perimeter_overlay:     pct_pts_3pt × fg3_vs_avg (perimeter archetype × arc defense)
    # creation_absorption:   prior_pot_ast × inactive_potential_ast_pool (who absorbs the creation void)
    # slashing_absorption:   prior_drives × inactive_drives_pool (who absorbs the drive void)
    "leverage_index",
    "paint_overlay",
    "perimeter_overlay",
    "creation_absorption",
    "slashing_absorption",
]

TARGETS = {
    "pts": "target_pts",
    "reb": "target_reb",
    "ast": "target_ast",
}

# Shared base hyperparameters
XGB_BASE = dict(
    max_depth          = 4,
    eta                = 0.05,
    subsample          = 0.8,
    colsample_bytree   = 0.8,
    min_child_weight   = 10,
    n_estimators       = 800,
    early_stopping_rounds = 40,
    tree_method        = "hist",
    random_state       = 42,
)

# Primary Poisson objectives (non-negative counting stats)
PROP_OBJECTIVES = {
    "pts": "count:poisson",
    "reb": "count:poisson",
    "ast": "count:poisson",
}

# Quantile levels to train in parallel
QUANTILES = [0.25, 0.50, 0.75]


def _impute(df: pd.DataFrame, medians: dict) -> pd.DataFrame:
    df = df.copy()
    for col, med in medians.items():
        if col in df.columns:
            df[col] = df[col].fillna(med)
    return df


def _train_quantile(X_train, y_train, X_val, y_val, alpha: float, prop: str) -> xgb.XGBRegressor:
    """Train one quantile model. Quantile loss doesn't support early stopping cleanly
    without a matching eval metric, so we use a fixed n_estimators with a held-out check."""
    params = {k: v for k, v in XGB_BASE.items() if k not in ("early_stopping_rounds", "eval_metric")}
    params["n_estimators"] = 600   # fixed; quantile loss eval metric is non-standard
    model = xgb.XGBRegressor(
        objective      = "reg:quantileerror",
        quantile_alpha = alpha,
        eval_metric    = "mae",
        **params,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def main():
    df = pd.read_parquet(TRAINING_FILE)
    print(f"Loaded {len(df):,} rows | seasons: {sorted(df['season'].unique())}")

    train_mask = df["season"].isin(TRAIN_SEASONS)
    val_mask   = df["season"].isin(VAL_SEASONS)
    train_raw  = df[train_mask]
    val_raw    = df[val_mask]

    medians = {c: float(train_raw[c].median()) for c in FEATURE_COLS if c in train_raw.columns}

    train = _impute(train_raw, medians)
    val   = _impute(val_raw,   medians)

    print(f"Train: {len(train):,} rows | Val: {len(val):,} rows")

    results = {}

    for prop, target_col in TARGETS.items():
        print(f"\n{'='*50}")
        print(f"  Training: {prop.upper()}")
        print(f"{'='*50}")

        X_train = train[FEATURE_COLS].values
        y_train = train[target_col].values.clip(min=0)
        X_val   = val[FEATURE_COLS].values
        y_val   = val[target_col].values.clip(min=0)

        # ── Primary Poisson model ─────────────────────────────────────────────
        params = {**XGB_BASE, "objective": PROP_OBJECTIVES[prop], "eval_metric": "rmse"}
        n_est  = params.pop("n_estimators")
        model  = xgb.XGBRegressor(n_estimators=n_est, **params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

        preds    = model.predict(X_val).clip(min=0)
        rmse     = float(np.sqrt(mean_squared_error(y_val, preds)))
        mae      = float(mean_absolute_error(y_val, preds))
        within_3 = float(np.mean(np.abs(preds - y_val) <= 3.0))

        print(f"\n  [Poisson] Val RMSE: {rmse:.3f}  MAE: {mae:.3f}  ±3: {within_3:.1%}")

        fi = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
        print(f"  Top features:")
        for feat, imp in fi.head(6).items():
            print(f"    {feat:<20} {imp:.4f}")

        model_file = f"xgb_{prop}_model.json"
        model.save_model(model_file)
        print(f"  Saved → {model_file}")

        results[prop] = {
            "rmse": rmse, "mae": mae, "within_3": within_3,
            "best_iteration": int(model.best_iteration),
            "model_file": model_file,
        }

        # ── Residual Quantile models (q25 / q50 / q75) ───────────────────────
        # Train quantile models to predict the RESIDUAL ERROR (actual - poisson),
        # not the raw target. During inference, add the offset onto the final
        # adjusted projection (after Adj 13 injury cascade + Adj 14 residual cal).
        # This means bands correctly widen when teammate scratches push base up.
        print(f"\n  Training residual quantile models (q25/q50/q75)…")
        train_resid = y_train - model.predict(X_train)   # residuals on train set
        val_resid   = y_val   - model.predict(X_val)     # residuals on val set

        for alpha in QUANTILES:
            tag = f"q{int(alpha*100)}"
            qmodel = _train_quantile(X_train, train_resid, X_val, val_resid, alpha, prop)

            # Evaluate: apply offset to Poisson preds, compare to actuals
            resid_preds = qmodel.predict(X_val)
            final_preds = (preds + resid_preds).clip(min=0)   # preds = poisson val preds
            qmae        = float(mean_absolute_error(y_val, final_preds))

            qfile = f"xgb_{prop}_{tag}.json"
            qmodel.save_model(qfile)
            print(f"    [{tag}] residual MAE: {qmae:.3f}  (raw resid mean={resid_preds.mean():.3f})  → {qfile}")
            results[prop][f"{tag}_mae"]  = qmae
            results[prop][f"{tag}_file"] = qfile

    # ── model_meta.json ───────────────────────────────────────────────────────
    meta = {
        "feature_cols":        FEATURE_COLS,
        "medians":             medians,
        "validation_results":  results,
        "train_seasons":       TRAIN_SEASONS,
        "val_seasons":         VAL_SEASONS,
        "quantiles":           QUANTILES,
        "quantile_mode":       "residual",   # q25/q50/q75 predict offset from Poisson
    }
    with open("model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("\nSaved → model_meta.json")

    print("\n── Summary ──────────────────────────────────────────")
    for prop, r in results.items():
        print(f"  {prop.upper():4s}  RMSE={r['rmse']:.2f}  MAE={r['mae']:.2f}  ±3={r['within_3']:.1%}  "
              f"q50_MAE={r.get('q50_mae', '?')}")
    print("Done.")


if __name__ == "__main__":
    main()
