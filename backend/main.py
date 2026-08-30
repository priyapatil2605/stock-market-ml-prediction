import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb


from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# ============================================================
# PATH CONFIGURATION
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(
    os.path.join(CURRENT_DIR, "..")
)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Stock Market ML Prediction API",
    description="Machine Learning API for stock movement prediction",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# FEATURE NAMES
# ============================================================

FEATURE_NAMES = [
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_width",
    "bb_position",
    "atr_14",
    "volatility_10",
    "volatility_20",
    "volume_ratio",
    "price_vs_sma_10",
    "price_vs_sma_20",
    "price_vs_sma_50",
    "ma_10_20_ratio",
    "daily_return",
    "return_lag_1",
    "return_lag_2",
    "return_lag_3",
    "return_lag_5",
    "return_lag_10"
]

LABELS = {
    0: "DOWN",
    1: "FLAT",
    2: "UP"
}


# ============================================================
# LOAD MODEL
# ============================================================

def load_xgboost_model():

    model_path = os.path.join(
        MODELS_DIR,
        "xgboost_model.json"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    model = xgb.XGBClassifier()
    model.load_model(model_path)

    return model


MODEL = load_xgboost_model()


# ============================================================
# REQUEST MODEL
# ============================================================

class PredictionRequest(BaseModel):

    features: list[float]


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "XGBoost",
        "features_required": len(FEATURE_NAMES)
    }


# ============================================================
# PREDICT ENDPOINT
# ============================================================

@app.post("/predict")
def predict(request: PredictionRequest):

    if len(request.features) != len(FEATURE_NAMES):

        raise HTTPException(
            status_code=400,
            detail=(
                f"Expected {len(FEATURE_NAMES)} features, "
                f"received {len(request.features)}"
            )
        )

    features = np.array(
        request.features
    ).reshape(1, -1)

    probabilities = MODEL.predict_proba(
        features
    )[0]

    prediction = int(
        np.argmax(probabilities)
    )

    return {
        "prediction": LABELS[prediction],
        "prediction_class": prediction,
        "probabilities": {
            "DOWN": float(probabilities[0]),
            "FLAT": float(probabilities[1]),
            "UP": float(probabilities[2])
        }
    }


# ============================================================
# MODEL COMPARISON ENDPOINT
# ============================================================

@app.get("/compare")
def compare_models():

    path = os.path.join(
        RESULTS_DIR,
        "model_comparison.json"
    )

    if not os.path.exists(path):

        raise HTTPException(
            status_code=404,
            detail="Model comparison results not found"
        )

    with open(path, "r") as file:

        data = json.load(file)

    return data


# ============================================================
# BACKTEST ENDPOINT
# ============================================================

@app.get("/backtest")
def get_backtest():

    path = os.path.join(
        RESULTS_DIR,
        "backtesting",
        "backtest_results.csv"
    )

    if not os.path.exists(path):

        raise HTTPException(
            status_code=404,
            detail="Backtest results not found"
        )

    df = pd.read_csv(path)

    return df.to_dict(
        orient="records"
    )


# ============================================================
# EXPLAINABILITY ENDPOINT
# ============================================================

@app.get("/explain")
def explain_model():

    path = os.path.join(
        RESULTS_DIR,
        "shap",
        "shap_feature_importance.csv"
    )

    if not os.path.exists(path):

        raise HTTPException(
            status_code=404,
            detail="SHAP results not found"
        )

    df = pd.read_csv(path)

    return {
        "model": "XGBoost",
        "feature_importance": df.to_dict(
            orient="records"
        )
    }


# ============================================================
# EQUITY CURVE ENDPOINT (time series for charting)
# ============================================================

@app.get("/equity-curve")
def get_equity_curve():

    files = {
        "Logistic Regression": "logistic_daily_returns.csv",
        "XGBoost": "xgboost_daily_returns.csv",
        "Equal Weight Benchmark": "benchmark_daily_returns.csv",
    }

    series = {}

    for label, filename in files.items():
        path = os.path.join(RESULTS_DIR, "backtesting", filename)

        if not os.path.exists(path):
            raise HTTPException(
                status_code=404,
                detail=f"Equity curve data not found: {filename}"
            )

        df = pd.read_csv(path)[["Date", "Equity"]].rename(
            columns={"Equity": label}
        )
        series[label] = df

    # Inner-join on Date so every row has a value for every strategy —
    # some strategies can have one fewer trading day than others (e.g. if
    # a ticker was missing data on the final date), and a naive outer join
    # would leave NaN gaps that break JSON parsing on the frontend.
    combined = None
    for df in series.values():
        combined = df if combined is None else combined.merge(df, on="Date", how="inner")

    combined = combined.sort_values("Date").reset_index(drop=True)

    return {
        "data_start": combined["Date"].iloc[0],
        "data_end": combined["Date"].iloc[-1],
        "points": combined.to_dict(orient="records"),
    }


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Stock Market ML Prediction API",
        "endpoints": [
            "/health",
            "/predict",
            "/compare",
            "/backtest",
            "/equity-curve",
            "/explain"
        ]
    }