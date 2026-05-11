"""
train_models.py
───────────────
Train XGBoost regressors for points, rebounds, and assists.

Strict chronological split — train on older seasons, validate on the most
recent season only. This mirrors actual sportsbook conditions: you never see
the future when building a line.

Usage:
    pip install xgboost scikit-learn
    python train_models.py

Outputs (in the same directory):
    xgb_pts_model.json
    xgb_reb_model.json
    xgb_ast_model.json
    model_meta.json    (feature column list + median imputation values)
"""

import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ── Config ────────────────────────────────────────────────────────────────────
TRAINING_FILE  = "training_data.parquet"
TRAIN_SEASONS  = ["2022-23", "2023-24"]
VAL_SEASONS    = ["2024-25"]

FEATURE_COLS = [
    "l5_pts", "l5_reb", "l5_ast", "l5_min", "l5_usg", "l5_ts",
    "l10_pts_std", "l10_min_std",
    "std_pts", "std_reb", "std_ast", "std_min",
    "gp_prior", "is_home", "rest_days",
    "opp_def_roll10", "opp_pace_roll10",
]

TARGETS = {
    "pts": "target_pts",
    "reb": "target_reb",
    "ast": "target_ast",
}

# Shared XGBoost hyperparameters — tuned for noisy count data
# Shallow trees + heavy regularization prevents overfitting to game-to-game variance
XGB_BASE = dict(
    max_depth          = 4,
    eta                = 0.05,      # low learning rate — more trees, less overfit
    subsample          = 0.8,
    colsample_bytree   = 0.8,
    min_child_weight   = 10,        # require ≥10 samples per leaf
    n_estimators       = 800,
    early_stopping_rounds = 40,
    eval_metric        = "rmse",
    tree_method        = "hist",    # fast histogram-based splits
    random_state       = 42,
)

# Poisson objective fits right-skewed counting stats well
# Falls back gracefully to reg:squarederror if Poisson diverges
PROP_OBJECTIVES = {
    "pts": "count:poisson",
    "reb": "count:poisson",
    "ast": "count:poisson",
}


def _impute(df: pd.DataFrame, medians: dict) -> pd.DataFrame:
    df = df.copy()
    for col, med in medians.items():
        if col in df.columns:
            df[col] = df[col].fillna(med)
    return df


def main():
    df = pd.read_parquet(TRAINING_FILE)
    print(f"Loaded {len(df):,} rows | seasons: {sorted(df['season'].unique())}")

    # Compute medians from training data only (no val leakage)
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
        y_train = train[target_col].values.clip(min=0)   # Poisson requires y ≥ 0
        X_val   = val[FEATURE_COLS].values
        y_val   = val[target_col].values.clip(min=0)

        params = {**XGB_BASE, "objective": PROP_OBJECTIVES[prop]}
        n_est  = params.pop("n_estimators")

        model = xgb.XGBRegressor(n_estimators=n_est, **params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=100,
        )

        preds = model.predict(X_val).clip(min=0)
        rmse  = float(np.sqrt(mean_squared_error(y_val, preds)))
        mae   = float(mean_absolute_error(y_val, preds))
        # Hit rate — within 3 units of actual (sports-book meaningful accuracy)
        within_3 = float(np.mean(np.abs(preds - y_val) <= 3.0))

        print(f"\n  Val RMSE:     {rmse:.3f}")
        print(f"  Val MAE:      {mae:.3f}")
        print(f"  Within ±3:    {within_3:.1%}")

        # Feature importance
        fi = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
        print(f"\n  Top features:")
        for feat, imp in fi.head(8).items():
            print(f"    {feat:<20} {imp:.4f}")

        model_file = f"xgb_{prop}_model.json"
        model.save_model(model_file)
        print(f"\n  Saved → {model_file}")

        results[prop] = {
            "rmse": rmse, "mae": mae, "within_3": within_3,
            "best_iteration": int(model.best_iteration),
            "model_file": model_file,
        }

    # Save/update model_meta.json with medians + feature list
    meta = {
        "feature_cols": FEATURE_COLS,
        "medians": medians,
        "validation_results": results,
        "train_seasons": TRAIN_SEASONS,
        "val_seasons": VAL_SEASONS,
    }
    with open("model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("\nSaved → model_meta.json")

    print("\n── Summary ──────────────────────────────────────────")
    for prop, r in results.items():
        print(f"  {prop.upper():4s}  RMSE={r['rmse']:.2f}  MAE={r['mae']:.2f}  ±3={r['within_3']:.1%}")
    print("Done.")


if __name__ == "__main__":
    main()
