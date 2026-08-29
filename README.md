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