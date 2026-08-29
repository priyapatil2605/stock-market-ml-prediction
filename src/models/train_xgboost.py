from pathlib import Path
import numpy as np
import joblib

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)


# ==========================================
# PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "ml_ready"

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# LOAD DATA
# ==========================================

def load_data():

    print("Loading ML-ready data...")

    X_train = np.load(DATA_DIR / "X_train.npy")
    X_test = np.load(DATA_DIR / "X_test.npy")

    y_train = np.load(DATA_DIR / "y_train.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")

    return X_train, X_test, y_train, y_test


# ==========================================
# MAIN
# ==========================================

def main():

    # --------------------------------------
    # LOAD DATA
    # --------------------------------------

    X_train, X_test, y_train, y_test = load_data()

    print("\nDataset shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test:  {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test:  {y_test.shape}")


    # --------------------------------------
    # CREATE MODEL
    # --------------------------------------

    print("\nTraining XGBoost...")

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,

        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,

        subsample=0.8,
        colsample_bytree=0.8,

        eval_metric="mlogloss",

        random_state=42,
        n_jobs=-1
    )


    # --------------------------------------
    # TRAIN
    # --------------------------------------

    model.fit(
        X_train,
        y_train
    )


    # --------------------------------------
    # PREDICT
    # --------------------------------------

    print("Making predictions...")

    y_pred = model.predict(
        X_test
    )


    # --------------------------------------
    # EVALUATE
    # --------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted"
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )


    print("\n" + "=" * 55)
    print("XGBOOST RESULTS")
    print("=" * 55)

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Weighted F1-score: {f1:.4f}")


    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "DOWN",
                "FLAT",
                "UP"
            ]
        )
    )


    print("Confusion Matrix:")

    print(cm)


    # --------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------

    print("\nFeature Importance:")

    feature_names = [
        "daily_return",
        "return_lag_1",
        "return_lag_2",
        "return_lag_3",
        "return_lag_5",
        "return_lag_10",
        "price_vs_sma_10",
        "price_vs_sma_20",
        "price_vs_sma_50",
        "ma_10_20_ratio",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_width",
        "bb_position",
        "volatility_10",
        "volatility_20",
        "atr_14",
        "volume_ratio"
    ]

    importances = model.feature_importances_

    importance_pairs = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True
    )

    for feature, importance in importance_pairs:

        print(
            f"{feature:25s} "
            f"{importance:.4f}"
        )


    # --------------------------------------
    # SAVE MODEL
    # --------------------------------------

    model_path = MODEL_DIR / "xgboost_model.json"

    model.save_model(
        model_path
    )


    # --------------------------------------
    # SAVE RESULTS
    # --------------------------------------

    results = {
        "model": "XGBoost",
        "accuracy": float(accuracy),
        "weighted_f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "feature_importance": {
            feature: float(importance)
            for feature, importance in importance_pairs
        }
    }

    joblib.dump(
        results,
        MODEL_DIR / "xgboost_results.pkl"
    )


    print("\n" + "=" * 55)
    print("XGBOOST TRAINING COMPLETE")
    print("=" * 55)

    print(
        f"\nModel saved to:\n"
        f"{model_path}"
    )


if __name__ == "__main__":
    main()