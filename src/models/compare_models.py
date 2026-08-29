import os
import json
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

from xgboost import XGBClassifier


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

ML_READY_DIR = os.path.join(BASE_DIR, "data", "ml_ready")
LSTM_READY_DIR = os.path.join(BASE_DIR, "data", "lstm_ready")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)


# =========================================================
# LSTM MODEL ARCHITECTURE
# MUST MATCH train_lstm.py
# =========================================================

class LSTMClassifier(nn.Module):

    def __init__(
        self,
        input_size=20,
        hidden_size=64,
        num_layers=2,
        num_classes=3
    ):
        super(LSTMClassifier, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            num_classes
        )

    def forward(self, x):

        lstm_out, (hidden, cell) = self.lstm(x)

        last_hidden = hidden[-1]

        output = self.fc(last_hidden)

        return output


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    # -----------------------------------------------------
    # LOAD STANDARD ML DATA
    # -----------------------------------------------------

    print("\nLoading Logistic Regression and XGBoost test data...")

    X_test = np.load(
        os.path.join(ML_READY_DIR, "X_test.npy")
    )

    y_test = np.load(
        os.path.join(ML_READY_DIR, "y_test.npy")
    )


    # -----------------------------------------------------
    # LOAD LOGISTIC REGRESSION
    # -----------------------------------------------------

    print("\nLoading Logistic Regression...")

    logistic_path = os.path.join(
        MODELS_DIR,
        "logistic_regression.pkl"
    )

    logistic_model = joblib.load(logistic_path)

    logistic_predictions = logistic_model.predict(X_test)


    # -----------------------------------------------------
    # LOAD XGBOOST
    # -----------------------------------------------------

    print("Loading XGBoost...")

    xgb_path = os.path.join(
        MODELS_DIR,
        "xgboost_model.json"
    )

    xgb_model = XGBClassifier()

    xgb_model.load_model(xgb_path)

    xgb_predictions = xgb_model.predict(X_test)


    # -----------------------------------------------------
    # LOAD LSTM DATA
    # -----------------------------------------------------

    print("Loading LSTM test data...")

    X_test_lstm = np.load(
        os.path.join(LSTM_READY_DIR, "X_test_lstm.npy")
    )

    y_test_lstm = np.load(
        os.path.join(LSTM_READY_DIR, "y_test_lstm.npy")
    )


    # -----------------------------------------------------
    # LOAD LSTM MODEL
    # -----------------------------------------------------

    print("Loading LSTM...")

    device = torch.device("cpu")

    lstm_model = LSTMClassifier(
        input_size=20,
        hidden_size=64,
        num_layers=2,
        num_classes=3
    )

    lstm_path = os.path.join(
        MODELS_DIR,
        "lstm_model.pt"
    )

    lstm_model.load_state_dict(
        torch.load(
            lstm_path,
            map_location=device
        )
    )

    lstm_model.eval()


    # -----------------------------------------------------
    # LSTM PREDICTIONS
    # -----------------------------------------------------

    X_tensor = torch.tensor(
        X_test_lstm,
        dtype=torch.float32
    )

    with torch.no_grad():

        outputs = lstm_model(X_tensor)

        lstm_predictions = torch.argmax(
            outputs,
            dim=1
        ).numpy()


    # -----------------------------------------------------
    # METRICS FUNCTION
    # -----------------------------------------------------

    def calculate_metrics(y_true, predictions):

        return {

            "Accuracy":
                accuracy_score(
                    y_true,
                    predictions
                ),

            "Precision":
                precision_score(
                    y_true,
                    predictions,
                    average="weighted",
                    zero_division=0
                ),

            "Recall":
                recall_score(
                    y_true,
                    predictions,
                    average="weighted",
                    zero_division=0
                ),

            "F1 Score":
                f1_score(
                    y_true,
                    predictions,
                    average="weighted",
                    zero_division=0
                )
        }


    # -----------------------------------------------------
    # CALCULATE RESULTS
    # -----------------------------------------------------

    logistic_metrics = calculate_metrics(
        y_test,
        logistic_predictions
    )

    xgb_metrics = calculate_metrics(
        y_test,
        xgb_predictions
    )

    lstm_metrics = calculate_metrics(
        y_test_lstm,
        lstm_predictions
    )


    # -----------------------------------------------------
    # CREATE COMPARISON TABLE
    # -----------------------------------------------------

    results = pd.DataFrame(
        [
            {
                "Model": "Logistic Regression",
                **logistic_metrics
            },

            {
                "Model": "XGBoost",
                **xgb_metrics
            },

            {
                "Model": "LSTM",
                **lstm_metrics
            }
        ]
    )


    # Convert to percentages

    metric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]

    results[metric_columns] = (
        results[metric_columns] * 100
    )


    # -----------------------------------------------------
    # DISPLAY RESULTS
    # -----------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)

    print(
        results.to_string(
            index=False,
            float_format="%.2f"
        )
    )


    # -----------------------------------------------------
    # FIND BEST MODEL
    # -----------------------------------------------------

    best_model = results.loc[
        results["F1 Score"].idxmax()
    ]

    print("\n" + "=" * 70)

    print("BEST MODEL BASED ON WEIGHTED F1 SCORE")

    print("=" * 70)

    print(
        f"\nModel: {best_model['Model']}"
    )

    print(
        f"F1 Score: "
        f"{best_model['F1 Score']:.2f}%"
    )


    # -----------------------------------------------------
    # SAVE RESULTS
    # -----------------------------------------------------

    csv_path = os.path.join(
        RESULTS_DIR,
        "model_comparison.csv"
    )

    results.to_csv(
        csv_path,
        index=False
    )


    json_path = os.path.join(
        RESULTS_DIR,
        "model_comparison.json"
    )

    results.to_json(
        json_path,
        orient="records",
        indent=4
    )


    print("\nResults saved to:")

    print(csv_path)

    print(json_path)


    print("\n" + "=" * 70)
    print("MODEL COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()