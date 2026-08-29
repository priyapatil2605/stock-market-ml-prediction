from pathlib import Path
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
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

    X_train = np.load(
        DATA_DIR / "X_train.npy"
    )

    X_test = np.load(
        DATA_DIR / "X_test.npy"
    )

    y_train = np.load(
        DATA_DIR / "y_train.npy"
    )

    y_test = np.load(
        DATA_DIR / "y_test.npy"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ==========================================
# MAIN
# ==========================================

def main():

    # --------------------------------------
    # LOAD DATA
    # --------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = load_data()


    print("\nDataset shapes:")

    print(
        f"X_train: {X_train.shape}"
    )

    print(
        f"X_test:  {X_test.shape}"
    )

    print(
        f"y_train: {y_train.shape}"
    )

    print(
        f"y_test:  {y_test.shape}"
    )


    # --------------------------------------
    # TRAIN MODEL
    # --------------------------------------

    print(
        "\nTraining Logistic Regression..."
    )

    model = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )


    # --------------------------------------
    # PREDICTIONS
    # --------------------------------------

    print(
        "Making predictions..."
    )

    y_pred = model.predict(
        X_test
    )


    # --------------------------------------
    # EVALUATION
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
    print("LOGISTIC REGRESSION RESULTS")
    print("=" * 55)

    print(
        f"\nAccuracy: {accuracy:.4f}"
    )

    print(
        f"Weighted F1-score: {f1:.4f}"
    )


    print(
        "\nClassification Report:"
    )

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


    print(
        "Confusion Matrix:"
    )

    print(
        cm
    )


    # --------------------------------------
    # SAVE MODEL
    # --------------------------------------

    model_path = (
        MODEL_DIR
        / "logistic_regression.pkl"
    )

    joblib.dump(
        model,
        model_path
    )


    # --------------------------------------
    # SAVE RESULTS
    # --------------------------------------

    results = {
        "model": "Logistic Regression",
        "accuracy": accuracy,
        "weighted_f1": f1,
        "confusion_matrix": cm.tolist()
    }

    joblib.dump(
        results,
        MODEL_DIR
        / "logistic_results.pkl"
    )


    print("\n" + "=" * 55)
    print("MODEL TRAINING COMPLETE")
    print("=" * 55)

    print(
        f"\nModel saved to:\n"
        f"{model_path}"
    )


if __name__ == "__main__":
    main()