from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
import numpy as np
import os
import json

from xgboost import XGBClassifier
import uvicorn

# -------------------------------------------------
# CONFIG & MODEL LOADING
# -------------------------------------------------

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "pregame_xgb.json")
FEATURE_ORDER_PATH = os.path.join(MODEL_DIR, "pregame_feature_order.json")

app = FastAPI(title="NFL ML Prediction Service", version="1.0.0")

model: Optional[XGBClassifier] = None
feature_order: List[str] = []
MODEL_VERSION = "xgboost-pregame-v1"


def load_model_artifacts():
    global model, feature_order

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. "
            f"Train and save the model first (train_pregame_model.py)."
        )
    if not os.path.exists(FEATURE_ORDER_PATH):
        raise RuntimeError(
            f"Feature order file not found at {FEATURE_ORDER_PATH}. "
            f"Expected JSON list of feature names."
        )

    print(f"Loading model from {MODEL_PATH} ...")
    model_local = XGBClassifier()
    model_local.load_model(MODEL_PATH)

    print(f"Loading feature order from {FEATURE_ORDER_PATH} ...")
    with open(FEATURE_ORDER_PATH, "r") as f:
        feature_order_json = json.load(f)

    if not isinstance(feature_order_json, list):
        raise RuntimeError("feature_order JSON is not a list.")

    feature_order_local = [str(c) for c in feature_order_json]

    # Only assign to globals after everything loaded successfully
    global model, feature_order
    model = model_local
    feature_order = feature_order_local

    print(f"Model and feature order loaded. {len(feature_order)} features.")


@app.on_event("startup")
def on_startup():
    try:
        load_model_artifacts()
    except Exception as e:
        # Log but don't crash app startup so /v1/health can still report failure
        print(f"Error loading model artifacts on startup: {e}")


# -------------------------------------------------
# SCHEMAS
# -------------------------------------------------

class GameFeaturesPregame(BaseModel):
    # These should match what Lovable / Supabase sends for pregame features.
    # If extra fields are present in the JSON, FastAPI will ignore them.
    season: int
    week: int
    home_pts_for_avg: float = 0.0
    home_pts_against_avg: float = 0.0
    away_pts_for_avg: float = 0.0
    away_pts_against_avg: float = 0.0
    vegas_spread: float = 0.0
    vegas_total: float = 44.5
    home_moneyline: float = 0.0
    away_moneyline: float = 0.0
    home_rest_days: int = 7
    away_rest_days: int = 7
    is_primetime: bool = False
    is_divisional: bool = False
    is_dome: bool = False
    weather_temp: float = 70.0
    weather_wind: float = 5.0


class EnsembleDetails(BaseModel):
    xgboost_prob: Optional[float] = None
    lightgbm_prob: Optional[float] = None
    bayesian_prob: Optional[float] = None
    lstm_form_prob: Optional[float] = None
    ft_transformer_prob: Optional[float] = None
    ensemble_method: str = "xgboost_only"
    meta_learner_weights: Optional[Dict[str, float]] = None


class BoundsDetails(BaseModel):
    probability_lower: float
    probability_upper: float
    spread_lower: float
    spread_upper: float
    total_lower: float
    total_upper: float
    coverage_level: float = 0.8


class MLPredictResponse(BaseModel):
    win_probability: float
    ats_margin: float
    total_points: float
    home_team_total: float
    away_team_total: float
    confidence: float
    model_version: str
    ensemble: Optional[EnsembleDetails] = None
    bounds: Optional[BoundsDetails] = None


class BatchPregameRequest(BaseModel):
    games: List[GameFeaturesPregame]
    include_ensemble: bool = False
    include_vegas_edge: bool = False  # placeholder


class BatchPregameResponse(BaseModel):
    predictions: List[MLPredictResponse]
    model_version: str


# -------------------------------------------------
# UTILS
# -------------------------------------------------

def build_feature_vector(game: GameFeaturesPregame) -> np.ndarray:
    """
    Construct feature vector in the exact order the model was trained on.
    Missing fields fall back to defaults defined in the GameFeaturesPregame model.
    """
    if model is None or not feature_order:
        raise RuntimeError("Model or feature order not loaded.")

    game_dict = game.dict()

    row = []
    for feat_name in feature_order:
        if feat_name not in game_dict:
            # If the feature didn’t exist in schema (e.g., future features),
            # just default to 0.0.
            row.append(0.0)
        else:
            val = game_dict[feat_name]
            if isinstance(val, bool):
                val = 1.0 if val else 0.0
            row.append(float(val))

    return np.array(row, dtype=float).reshape(1, -1)


def predict_pregame_internal(game: GameFeaturesPregame) -> MLPredictResponse:
    X = build_feature_vector(game)
    y_prob = float(model.predict_proba(X)[:, 1][0])  # probability home team wins

    # Simple heuristic for ATS/total based on probabilities:
    ats_margin = (y_prob - 0.5) * 10.0  # ~10 point swing from 0–1

    total_points = float(getattr(game, "vegas_total", 44.5))

    # Split total into home/away prediction biased by win probability
    home_team_total = total_points * (0.5 + (y_prob - 0.5) * 0.3)
    away_team_total = total_points - home_team_total

    confidence = 0.5 + abs(y_prob - 0.5)

    return MLPredictResponse(
        win_probability=y_prob,
        ats_margin=float(ats_margin),
        total_points=total_points,
        home_team_total=float(home_team_total),
        away_team_total=float(away_team_total),
        confidence=float(confidence),
        model_version=MODEL_VERSION,
        ensemble=EnsembleDetails(
            xgboost_prob=y_prob,
            ensemble_method="xgboost_only",
            meta_learner_weights={"xgboost": 1.0},
        ),
        bounds=BoundsDetails(
            probability_lower=max(0.0, y_prob - 0.15),
            probability_upper=min(1.0, y_prob + 0.15),
            spread_lower=ats_margin - 7.0,
            spread_upper=ats_margin + 7.0,
            total_lower=total_points - 10.0,
            total_upper=total_points + 10.0,
            coverage_level=0.8,
        ),
    )


# -------------------------------------------------
# ENDPOINTS
# -------------------------------------------------

@app.get("/v1/health")
def health_check():
    return {
        "status": "healthy" if model is not None else "uninitialized",
        "models_loaded": ["pregame_xgboost"] if model is not None else [],
        "model_version": MODEL_VERSION,
        "feature_count": len(feature_order),
    }


@app.post("/v1/predict/pregame", response_model=MLPredictResponse)
def predict_pregame(game: GameFeaturesPregame):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    try:
        return predict_pregame_internal(game)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/predict/ensemble", response_model=MLPredictResponse)
def predict_ensemble(game: GameFeaturesPregame):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    return predict_pregame_internal(game)


@app.post("/v1/predict/batch", response_model=BatchPregameResponse)
def predict_batch(request: BatchPregameRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    preds: List[MLPredictResponse] = []
    for g in request.games:
        preds.append(predict_pregame_internal(g))

    return BatchPregameResponse(
        predictions=preds,
        model_version=MODEL_VERSION,
    )


# -------------------------------------------------
# LOCAL ENTRY POINT (for Render if it runs `python server.py`)
# -------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("server:app", host="0.0.0.0", port=port)
