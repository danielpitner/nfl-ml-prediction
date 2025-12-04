"""
train_pregame_model.py

Train an XGBoost model for pregame win probability using historical NFL games.

EXPECTED INPUT DATA (CSV):
- One row per game
- Columns MUST include at least:

    game_id (optional, for logging)
    season
    week
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
    home_win  (TARGET: 1 if home team won, 0 otherwise)

USAGE:

1) Export a CSV from your database (Supabase / Lovable) to e.g.:
   data/games_with_features.csv

2) Install dependencies locally:
   pip install xgboost pandas numpy scikit-learn

3) Run:
   python train_pregame_model.py

4) Output:
   - models/pregame_xgb.json         (XGBoost model)
   - models/pregame_feature_order.json  (list of feature names in correct order)

These files will be loaded by your FastAPI ML service.
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

# These must match your GameFeatures fields in server.py
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

# You can tweak this to hold out the last N seasons for validation
VALIDATION_MIN_SEASON = 2023  # e.g., train on <= 2022, validate on >= 2023


# ---------------------------
# DATA LOADING & SPLITTING
# ---------------------------

def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found at {path}. Please export your data there.")

    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows from {path}")

    # Basic sanity check
    missing_cols = [c for c in FEATURE_COLS + [TARGET_COL, SEASON_COL, WEEK_COL] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in CSV: {missing_cols}")

    # Drop rows with missing target
    before = len(df)
    df = df.dropna(subset=[TARGET_COL])
    print(f"Dropped {before - len(df)} rows with missing {TARGET_COL}")

    # Ensure target is 0/1
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    return df


def time_based_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train on seasons < VALIDATION_MIN_SEASON, validate on seasons >= VALIDATION_MIN_SEASON.
    This avoids peeking into the future.
    """
    train_df = df[df[SEASON_COL] < VALIDATION_MIN_SEASON].copy()
    val_df = df[df[SEASON_COL] >= VALIDATION_MIN_SEASON].copy()

    if train_df.empty or val_df.empty:
        raise ValueError(
            f"Train or validation split is empty. "
            f"Check VALIDATION_MIN_SEASON={VALIDATION_MIN_SEASON} and your data's {SEASON_COL} range."
        )

    print(
        f"Train set: {len(train_df):,} rows (seasons <= {VALIDATION_MIN_SEASON - 1}), "
        f"Val set: {len(val_df):,} rows (seasons >= {VALIDATION_MIN_SEASON})"
    )

    return train_df, val_df


def build_feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert DataFrame into X (features) and y (target) with fixed FEATURE_COLS order.
    """
    X = df[FEATURE_COLS].copy()

    # Convert boolean-like fields to 0/1 if needed
    for col in ["is_primetime", "is_divisional", "is_dome"]:
        if col in X.columns:
            if X[col].dtype == bool:
                X[col] = X[col].astype(int)
            # If it's strings like "true"/"false", map them
            if X[col].dtype == object:
                X[col] = X[col].str.lower().map({"true": 1, "false": 0})
            X[col] = X[col].fillna(0)

    # Basic numeric fill
    X = X.fillna(0.0)

    y = df[TARGET_COL].values.astype(int)

    return X.values.astype(float), y


# ---------------------------
# TRAINING
# ---------------------------

def train_xgb_model(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> XGBClassifier:
    """
    Train an XGBoost classifier with reasonable defaults and early stopping.
    """

    # Rough class balance
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
        tree_method="hist",  # good default for CPU
        random_state=42,
    )

    eval_set = [(X_train, y_train), (X_val, y_val)]

    print("Training XGBoost model...")
    model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        verbose=True,
        early_stopping_rounds=30,
    )

    return model


# ---------------------------
# EVALUATION
# ---------------------------

def evaluate_model(model: XGBClassifier, X: np.ndarray, y: np.ndarray, split_name: str) -> None:
    """
    Print accuracy and Brier score for a given split.
    """
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
    # 1) Load data
    df = load_data(DATA_CSV_PATH)

    # 2) Split train/val by season
    train_df, val_df = time_based_split(df)

    # 3) Build feature matrices
    X_train, y_train = build_feature_matrix(train_df)
    X_val, y_val = build_feature_matrix(val_df)

    print(f"Training matrix: {X_train.shape[0]:,} rows x {X_train.shape[1]} features")
    print(f"Validation matrix: {X_val.shape[0]:,} rows x {X_val.shape[1]} features")

    # 4) Train model
    model = train_xgb_model(X_train, y_train, X_val, y_val)

    # 5) Evaluate
    evaluate_model(model, X_train, y_train, "Train")
    evaluate_model(model, X_val, y_val, "Validation")

    # 6) Save artifacts
    save_artifacts(model, FEATURE_COLS)


if __name__ == "__main__":
    main()
