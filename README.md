# Stock Market Movement Prediction using Machine Learning

A machine learning-based stock market prediction system that predicts short-term stock movement using technical indicators, compares multiple ML models, performs walk-forward validation, evaluates trading strategies through portfolio-level backtesting, and explains XGBoost predictions using SHAP.

---

## Project Overview

This project builds an end-to-end machine learning pipeline for stock market movement prediction.

The system performs:

- Historical stock data collection
- Data cleaning and preprocessing
- Technical indicator generation
- Target label creation
- Logistic Regression training
- XGBoost training
- LSTM sequence modeling
- Time-based train-test splitting
- Walk-forward validation
- Portfolio-level backtesting
- Model comparison
- SHAP explainability
- FastAPI backend
- React dashboard

The objective is not only to build predictive models, but also to evaluate whether model predictions translate into useful investment performance.

---

# Project Architecture

```text
Stock Market Data
       |
       v
Data Collection
       |
       v
Data Cleaning
       |
       v
Feature Engineering
       |
       v
Target Generation
       |
       v
+-----------------------------+
| Machine Learning Models     |
|                             |
| Logistic Regression         |
| XGBoost                     |
| LSTM                        |
+-----------------------------+
       |
       v
Model Evaluation
       |
       +-------------------+
       |                   |
       v                   v
Walk-Forward          SHAP Analysis
Validation
       |
       v
Portfolio Backtesting
       |
       v
FastAPI Backend
       |
       v
React Dashboard
```

---

## Setup

```bash
git clone https://github.com/priyapatil2605/stock-market-ml-prediction.git
cd stock-market-ml-prediction
pip install -r requirements.txt

# Pull fresh data and rebuild everything (data/ and models/ are gitignored,
# so you need to regenerate them locally)
python src/data/data_pipeline.py
python src/features/build_features.py      # adjust to your actual script names
python src/models/train_models.py
python src/models/walk_forward_validation.py
python src/backtesting/run_backtest.py

# Backend
cd backend && uvicorn main:app --reload

# Frontend
cd frontend && npm install && npm run dev
```

---

## Results & Interpretation

**Model comparison (held-out test set)**

| Model | Accuracy | F1 (macro) |
|---|---|---|
| LSTM | 47.25% | 42.86% |
| Logistic Regression | 46.92% | 41.07% |
| XGBoost | 44.81% | 42.58% |

**Walk-forward validation (rolling, chronological folds — no lookahead)**

| Model | Accuracy | F1 (macro) |
|---|---|---|
| Logistic Regression | 45.01% | 38.56% |
| XGBoost | 43.33% | 39.91% |

Accuracy sits close to a 3-class random baseline (~33-45%), which is expected and honest for daily equity direction prediction — this isn't a data leak, it's what real market efficiency looks like.

**Portfolio backtest, net of a 10 bps/turnover transaction cost assumption**

| Strategy | Total Return | Ann. Return | Sharpe | Max Drawdown | Win Rate |
|---|---|---|---|---|---|
| Logistic Regression | 6.24% | 3.12% | 0.256 | -37.28% | 42.66% |
| XGBoost | -23.93% | -12.95% | -0.329 | -56.14% | 49.30% |
| Equal-Weight Benchmark | 95.80% | 40.69% | 1.541 | -25.56% | 56.85% |

**Both strategies substantially underperform a simple equal-weight buy-and-hold benchmark once realistic transaction costs are included.** Daily rebalancing across a small universe (10 tickers) produces high turnover, and at an assumed 10 bps cost per unit of turnover this consumes 59-60% of gross returns over the backtest period.

This is the expected, honest outcome of a leak-free pipeline — a pipeline with lookahead bias would show unrealistically strong returns. The value of this project is the rigor of the pipeline (walk-forward validation, no lookahead, realistic cost modeling), not a claim that it beats the market.

---

## Limitations

- Universe limited to 10 liquid large-cap tickers; results may not generalize to small caps or other asset classes
- Transaction cost is a flat 10 bps/turnover assumption, not modeled per-ticker liquidity or actual bid-ask spreads
- Daily bar data only — no intraday signals or execution modeling
- No regime detection (bull/bear/sideways) — a single global model is used across the full period
- Labeling threshold (forward return > X%) is a fixed global value, not tuned per ticker's typical volatility