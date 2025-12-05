"""
train_pregame_model.py

Train an XGBoost model for pregame win probability using historical NFL games.

EXPECTED INPUT DATA (CSV):
- One row per game
- MUST include at least:
    season
    week
    home_win  (1 if home team won, 0 otherwise)

- It WILL TRY to use these feature columns if present, and will
  create them with default values if they are missing:

    home_pts_for_avg
    home_pts_against_avg
    away_pts_for_avg
    away_pts_against_avg
    vegas_spread
    vegas_total
    home_moneyline
    away_moneyline
    home_rest_days
    away_rest_days
    is_primetime
    is_divisional
    is_dome
    weather_temp
    weather_wind

USAGE:

1) Put your CSV at:
   data/games_with_features.csv

2) Install dependencies:
   pip3 install xgboost pandas numpy scikit-learn

3) Run:
   python3 train_pregame_model.py

4) Output:
   - models/pregame_xgb.json
   - models/pregame_feature_order.json
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Tuple

from sklearn.metrics import accuracy_score, brier_score_loss
from xgboost import XGBClassifier

# ---------------------------
# CONFIG
# ---------------------------

DATA_CSV_PATH = "data/games_with_features.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "pregame_xgb.json")
FEATURE_ORDER_PATH = os.path.join(MODEL_DIR, "pregame_feature_order.json")

TARGET_COL = "home_win"  # 1 if home team won, 0 otherwise

# These must match how we’ll build features in the ML server.
# If some are missing in the CSV, we’ll create them with defaults.
FEATURE_COLS: List[str] = [
    "home_pts_for_avg",
    "home_pts_against_avg",
    "away_pts_for_avg",
    "away_pts_against_avg",
    "vegas_spread",
    "vegas_total",
    "home_moneyline",
    "away_moneyline",
    "home_rest_days",
    "away_rest_days",
    "is_primetime",
    "is_divisional",
    "is_dome",
    "weather_temp",
    "weather_wind",
]

SEASON_COL = "season"
WEEK_COL = "week"

# Train on seasons < VALIDATION_MIN_SEASON, validate on >= VALIDATION_MIN_SEASON
VALIDATION_MIN_SEASON = 2023


# ---------------------------
# DATA LOADING & SPLITTING
# ---------------------------

def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found at {path}. Please export your data there.")

    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows from {path}")

    # Check for absolutely required columns
    required_base_cols = [TARGET_COL, SEASON_COL, WEEK_COL]
    missing_base = [c for c in required_base_cols if c not in df.columns]
    if missing_base:
        raise ValueError(f"CSV is missing required base columns: {missing_base}")

    # Drop rows with missing target
    before = len(df)
    df = df.dropna(subset=[TARGET_COL])
    print(f"Dropped {before - len(df)} rows with missing {TARGET_COL}")

    # Ensure target is 0/1
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    # Handle missing feature columns by creating them with defaults
    missing_features = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_features:
        print("\nWARNING: The following feature columns are missing from the CSV:")
        for c in missing_features:
            print(f"  - {c}")
        print("They will be created with default values so training can continue.\n")

    for col in FEATURE_COLS:
        if col not in df.columns:
            # Choose reasonable defaults
            if col in ["is_primetime", "is_divisional", "is_dome"]:
                default_value = 0
            elif col in ["weather_temp"]:
                default_value = 70.0  # neutral football weather
            elif col in ["weather_wind"]:
                default_value = 5.0   # light wind
            elif col in ["home_rest_days", "away_rest_days"]:
                default_value = 7
            else:
                default_value = 0.0

            df[col] = default_value

    return df


def time_based_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prefer time-based split; if that fails, fall back to random split."""
    train_df = df[df[SEASON_COL] < VALIDATION_MIN_SEASON].copy()
    val_df = df[df[SEASON_COL] >= VALIDATION_MIN_SEASON].copy()

    if train_df.empty or val_df.empty:
        # Fallback: random 80/20 split if seasons don’t support a clean boundary
        print(
            f"Time-based split failed (train or val empty). "
            f"Falling back to random 80/20 split."
        )
        df_shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        split_idx = int(0.8 * len(df_shuffled))
        train_df = df_shuffled.iloc[:split_idx].copy()
        val_df = df_shuffled.iloc[split_idx:].copy()

    print(
        f"Train set: {len(train_df):,} rows, "
        f"Val set: {len(val_df):,} rows"
    )

    return train_df, val_df


def build_feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    X = df[FEATURE_COLS].copy()

    # Convert boolean-like fields to 0/1 if needed
    for col in ["is_primetime", "is_divisional", "is_dome"]:
        if col in X.columns:
            if X[col].dtype == bool:
                X[col] = X[col].astype(int)
            if X[col].dtype == object:
                X[col] = X[col].astype(str).str.lower().map({"true": 1, "false": 0})
            X[col] = X[col].fillna(0)

    # Numeric fill
    X = X.fillna(0.0)

    y = df[TARGET_COL].values.astype(int)

    return X.values.astype(float), y


# ---------------------------
# TRAINING
# ---------------------------

def train_xgb_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray
) -> XGBClassifier:
    pos_rate = y_train.mean()
    print(f"Positive class rate in train data (home_win=1): {pos_rate:.3f}")

    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=4,
        reg_lambda=1.0,
        reg_alpha=0.0,
        tree_method="hist",
        random_state=42,
    )

    eval_set = [(X_train, y_train), (X_val, y_val)]

    print("Training XGBoost model...")
    # Compatible with XGBoost 2.x: no early_stopping_rounds kwarg
    model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        verbose=True,
    )

    return model


# ---------------------------
# EVALUATION
# ---------------------------

def evaluate_model(model: XGBClassifier, X: np.ndarray, y: np.ndarray, split_name: str) -> None:
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y, y_pred)
    brier = brier_score_loss(y, y_prob)

    print(f"[{split_name}] Accuracy: {acc:.4f}, Brier score: {brier:.4f}")


# ---------------------------
# SAVE ARTIFACTS
# ---------------------------

def save_artifacts(model: XGBClassifier, feature_cols: List[str]) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"Saving model to {MODEL_PATH}")
    model.save_model(MODEL_PATH)

    print(f"Saving feature order to {FEATURE_ORDER_PATH}")
    with open(FEATURE_ORDER_PATH, "w") as f:
        json.dump(feature_cols, f, indent=2)

    print("Done. Artifacts ready for ML service.")


# ---------------------------
# MAIN
# ---------------------------

def main():
    df = load_data(DATA_CSV_PATH)
    train_df, val_df = time_based_split(df)

    X_train, y_train = build_feature_matrix(train_df)
    X_val, y_val = build_feature_matrix(val_df)

    print(f"Training matrix: {X_train.shape[0]:,} rows x {X_train.shape[1]} features")
    print(f"Validation matrix: {X_val.shape[0]:,} rows x {X_val.shape[1]} features")

    model = train_xgb_model(X_train, y_train, X_val, y_val)

    evaluate_model(model, X_train, y_train, "Train")
    evaluate_model(model, X_val, y_val, "Validation")

    save_artifacts(model, FEATURE_COLS)


if __name__ == "__main__":
    main()
