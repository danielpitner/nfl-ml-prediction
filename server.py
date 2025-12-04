from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict

import math

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(title="NFL ML Prediction Server", version="0.1.0")

# ---------------------------
# Optional API key auth
# ---------------------------
ML_API_KEY = None  # set from env later if you want auth


def verify_api_key(authorization: Optional[str] = Header(default=None)):
    """
    Optional simple API key check.
    If you don't want auth yet, leave ML_API_KEY = None.
    """
    if ML_API_KEY is None:
        return

    if authorization != f"Bearer {ML_API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ---------------------------
# Schemas matching Lovable's spec
# ---------------------------
class EnsembleData(BaseModel):
    xgboost_prob: Optional[float] = None
    lightgbm_prob: Optional[float] = None
    bayesian_prob: Optional[float] = None
    lstm_form_prob: Optional[float] = None
    ft_transformer_prob: Optional[float] = None
    ensemble_method: str = "placeholder"
    meta_learner_weights: Optional[Dict[str, float]] = None


class BoundsData(BaseModel):
    probability_lower: float
    probability_upper: float
    spread_lower: float
    spread_upper: float
    total_lower: float
    total_upper: float
    coverage_level: float


class MLPredictResponse(BaseModel):
    win_probability: float
    ats_margin: float
    total_points: float
    home_team_total: float
    away_team_total: float
    confidence: float
    model_version: str
    ensemble: Optional[EnsembleData] = None
    bounds: Optional[BoundsData] = None


class GameFeatures(BaseModel):
    game_id: Optional[str] = None
    home_team_id: Optional[str] = None
    away_team_id: Optional[str] = None
    season: int
    week: int
    home_pts_for_avg: float
    home_pts_against_avg: float
    away_pts_for_avg: float
    away_pts_against_avg: float
    vegas_spread: float
    vegas_total: float
    home_moneyline: float
    away_moneyline: float
    home_rest_days: int
    away_rest_days: int
    is_primetime: bool
    is_divisional: bool
    is_dome: bool
    weather_temp: float
    weather_wind: float


class BatchPredictRequest(BaseModel):
    games: List[GameFeatures]
    include_ensemble: bool = True
    include_vegas_edge: bool = False  # reserved
    include_bounds: bool = True


class VegasEdgeResponse(BaseModel):
    model_probability: float
    vegas_probability: float
    disagreement_score: float
    bet_signal: str
    sharp_money_indicator: bool
    line_movement_direction: str
    model_version: str


# ---------------------------
# Helper functions (placeholder "models")
# ---------------------------
def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def implied_prob_from_moneyline(ml: float) -> float:
    # Simple American odds → implied probability
    if ml > 0:
        return 100.0 / (ml + 100.0)
    else:
        return -ml / (-ml + 100.0)


def base_prediction(features: GameFeatures) -> MLPredictResponse:
    """
    Placeholder ML logic that roughly mimics a weighted model.
    This is where you'll later call real XGBoost / LightGBM.
    """
    # Offense / defense strength
    home_off = features.home_pts_for_avg
    home_def = features.home_pts_against_avg
    away_off = features.away_pts_for_avg
    away_def = features.away_pts_against_avg

    # Basic strength diff
    offensive_edge = home_off - away_off
    defensive_edge = (away_def - home_def)  # lower is better

    # Rest advantage
    rest_diff = features.home_rest_days - features.away_rest_days

    # Vegas info
    vegas_spread = features.vegas_spread  # home - away
    vegas_total = features.vegas_total

    # Moneyline implied probs
    home_ml_prob = implied_prob_from_moneyline(features.home_moneyline)
    away_ml_prob = implied_prob_from_moneyline(features.away_moneyline)

    # Primetime / dome adjustments
    primetime_bonus = 0.03 if features.is_primetime else 0.0
    dome_bonus = 0.02 if features.is_dome else 0.0
    divisional_noise = -0.01 if features.is_divisional else 0.0

    # Wind penalty for passing / scoring
    wind_penalty = -0.0008 * max(0.0, features.weather_wind - 8.0)

    # Build a simple logit score for home win
    logit = 0.0
    logit += -0.35 * vegas_spread               # if home is favorite (negative), boosts prob
    logit += 0.05 * offensive_edge
    logit += 0.05 * defensive_edge
    logit += 0.02 * rest_diff
    logit += 2.0 * (home_ml_prob - 0.5)
    logit += primetime_bonus + dome_bonus + divisional_noise
    logit += wind_penalty

    win_prob = sigmoid(logit)

    # Predict spread (home minus away)
    ats_margin = -vegas_spread + (win_prob - 0.5) * 8.0

    # Predict total points around vegas_total with small adjustments
    offensive_total = (home_off + away_off) / 2.0
    defensive_total = (home_def + away_def) / 2.0
    total_bias = 0.15 * (offensive_total - defensive_total)
    total_bias += -0.15 * max(0.0, features.weather_wind - 10.0)
    total_bias += 0.05 * (features.weather_temp - 50.0) / 10.0

    total_points = vegas_total + total_bias

    # Split home/away totals
    home_team_total = total_points / 2.0 + ats_margin / 2.0
    away_team_total = total_points - home_team_total

    # Clamp to reasonable ranges
    home_team_total = max(7.0, min(45.0, home_team_total))
    away_team_total = max(7.0, min(45.0, away_team_total))
    total_points = home_team_total + away_team_total

    # Confidence: how far from coin flip + how strong vegas + rest signals are
    confidence = abs(win_prob - 0.5) * 1.4
    confidence += 0.1 * min(1.0, abs(vegas_spread) / 7.0)
    confidence += 0.05 * min(1.0, abs(rest_diff) / 7.0)
    confidence = max(0.1, min(0.99, confidence))

    response = MLPredictResponse(
        win_probability=round(win_prob, 4),
        ats_margin=round(ats_margin, 2),
        total_points=round(total_points, 1),
        home_team_total=round(home_team_total, 1),
        away_team_total=round(away_team_total, 1),
        confidence=round(confidence, 3),
        model_version="python-placeholder-v0.1",
    )

    return response


def add_ensemble_and_bounds(base: MLPredictResponse) -> MLPredictResponse:
    # Fake ensemble probs around base.win_probability
    p = base.win_probability
    ensemble = EnsembleData(
        xgboost_prob=round(p + 0.01, 4),
        lightgbm_prob=round(p - 0.01, 4),
        bayesian_prob=round(0.5 * p + 0.25, 4),
        lstm_form_prob=None,
        ft_transformer_prob=None,
        ensemble_method="weighted_average",
        meta_learner_weights={
            "xgboost": 0.4,
            "lightgbm": 0.3,
            "bayesian": 0.3,
        },
    )

    # Simple +/- bands for conformal bounds (placeholder)
    spread = 0.08
    bounds = BoundsData(
        probability_lower=max(0.0, round(p - spread, 4)),
        probability_upper=min(1.0, round(p + spread, 4)),
        spread_lower=round(base.ats_margin - 4.0, 2),
        spread_upper=round(base.ats_margin + 4.0, 2),
        total_lower=round(base.total_points - 6.0, 1),
        total_upper=round(base.total_points + 6.0, 1),
        coverage_level=0.8,
    )

    base.ensemble = ensemble
    base.bounds = bounds
    base.model_version = "python-ensemble-placeholder-v0.1"
    return base


def vegas_edge_from_prediction(features: GameFeatures, pred: MLPredictResponse) -> VegasEdgeResponse:
    # Model prob vs vegas implied
    home_ml_prob = implied_prob_from_moneyline(features.home_moneyline)
    vegas_prob = home_ml_prob

    model_prob = pred.win_probability
    disagreement = model_prob - vegas_prob

    # Bet signal
    abs_diff = abs(disagreement)
    if abs_diff > 0.1:
        bet_signal = "strong_bet"
    elif abs_diff > 0.05:
        bet_signal = "weak_bet"
    else:
        bet_signal = "no_bet"

    # Placeholder sharp money indicator + line movement
    sharp_money_indicator = abs_diff > 0.12
    line_movement_direction = "stable"

    return VegasEdgeResponse(
        model_probability=round(model_prob, 4),
        vegas_probability=round(vegas_prob, 4),
        disagreement_score=round(disagreement, 4),
        bet_signal=bet_signal,
        sharp_money_indicator=sharp_money_indicator,
        line_movement_direction=line_movement_direction,
        model_version=pred.model_version,
    )


# ---------------------------
# Endpoints
# ---------------------------
@app.get("/v1/health")
def health():
    return {
        "status": "healthy",
        "models_loaded": [
            "pregame_placeholder",
            "ensemble_placeholder",
        ],
        "version": "0.1.0",
    }


@app.post("/v1/predict/pregame", dependencies=[Depends(verify_api_key)])
def predict_pregame(features: GameFeatures) -> MLPredictResponse:
    """
    Single game prediction (pregame model).
    """
    return base_prediction(features)


@app.post("/v1/predict/ensemble", dependencies=[Depends(verify_api_key)])
def predict_ensemble(features: GameFeatures) -> MLPredictResponse:
    """
    Full ensemble prediction (base + ensemble + bounds).
    """
    base = base_prediction(features)
    enriched = add_ensemble_and_bounds(base)
    return enriched


@app.post("/v1/predict/vegas-edge", dependencies=[Depends(verify_api_key)])
def predict_vegas_edge(features: GameFeatures) -> VegasEdgeResponse:
    """
    Vegas edge analysis based on model vs vegas implied.
    """
    base = base_prediction(features)
    return vegas_edge_from_prediction(features, base)


@app.post("/v1/predict/batch", dependencies=[Depends(verify_api_key)])
def predict_batch(request: BatchPredictRequest):
    """
    Batch predictions for multiple games.
    """
    predictions: List[MLPredictResponse] = []
    for g in request.games:
        pred = base_prediction(g)
        if request.include_ensemble:
            pred = add_ensemble_and_bounds(pred)
        predictions.append(pred)

    return {
        "predictions": predictions,
        "model_version": "python-batch-placeholder-v0.1",
    }
