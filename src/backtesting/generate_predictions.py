import os
import sys
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn

from xgboost import XGBClassifier


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

sys.path.append(BASE_DIR)

ML_READY_DIR = os.path.join(BASE_DIR, "data", "ml_ready")
LSTM_READY_DIR = os.path.join(BASE_DIR, "data", "lstm_ready")
MODEL_DATA_DIR = os.path.join(BASE_DIR, "data", "model_data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)


# =========================================================
# LSTM ARCHITECTURE
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
        super().__init__()

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

        _, (hidden, _) = self.lstm(x)

        last_hidden = hidden[-1]

        return self.fc(last_hidden)


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 65)
    print("GENERATING MODEL PREDICTIONS FOR BACKTESTING")
    print("=" * 65)


    # =====================================================
    # LOAD ORIGINAL ML DATA
    # =====================================================

    print("\nLoading ML dataset...")

    df = pd.read_csv(
        os.path.join(
            MODEL_DATA_DIR,
            "market_ml_dataset.csv"
        )
    )

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values(
        ["Ticker", "Date"]
    ).reset_index(drop=True)


    # =====================================================
    # CREATE SAME TIME-BASED TEST PERIOD
    # =====================================================

    test_start = pd.Timestamp("2024-01-01")

    test_df = df[
        df["Date"] >= test_start
    ].copy()

    print(
        f"Total test rows available: {len(test_df)}"
    )

    print(
        f"Test period: "
        f"{test_df['Date'].min().date()} "
        f"to "
        f"{test_df['Date'].max().date()}"
    )


    # =====================================================
    # LOAD STANDARD TEST FEATURES
    # =====================================================

    print("\nLoading Logistic/XGBoost test data...")

    X_test = np.load(
        os.path.join(
            ML_READY_DIR,
            "X_test.npy"
        )
    )

    y_test = np.load(
        os.path.join(
            ML_READY_DIR,
            "y_test.npy"
        )
    )


    # =====================================================
    # LOAD LOGISTIC MODEL
    # =====================================================

    print("\nLoading Logistic Regression...")

    logistic_model = joblib.load(
        os.path.join(
            MODELS_DIR,
            "logistic_regression.pkl"
        )
    )

    logistic_predictions = logistic_model.predict(
        X_test
    )


    # =====================================================
    # LOAD XGBOOST MODEL
    # =====================================================

    print("Loading XGBoost...")

    xgb_model = XGBClassifier()

    xgb_model.load_model(
        os.path.join(
            MODELS_DIR,
            "xgboost_model.json"
        )
    )

    xgb_predictions = xgb_model.predict(
        X_test
    )


    # =====================================================
    # IMPORTANT ALIGNMENT CHECK
    # =====================================================

    print("\nChecking prediction alignment...")

    print(
        f"Test dataframe rows: {len(test_df)}"
    )

    print(
        f"Standard ML predictions: "
        f"{len(logistic_predictions)}"
    )


    # The standard models should align with
    # the time-based test dataset.

    if len(test_df) != len(logistic_predictions):

        print(
            "\nWARNING: Dataset alignment mismatch!"
        )

        print(
            "This can happen because prepare_data.py "
            "used an exact split boundary."
        )

        print(
            "We will reconstruct the exact test rows "
            "using the saved dataset order."
        )

        # Use the final N rows according to the
        # original preparation split.

        test_df = df.tail(
            len(logistic_predictions)
        ).copy()

        print(
            f"Reconstructed test rows: {len(test_df)}"
        )


    # =====================================================
    # ADD STANDARD MODEL PREDICTIONS
    # =====================================================

    test_df = test_df.reset_index(
        drop=True
    )

    test_df["logistic_prediction"] = (
        logistic_predictions
    )

    test_df["xgboost_prediction"] = (
        xgb_predictions
    )


    # =====================================================
    # LABEL MAPPING
    # =====================================================

    label_map = {

        0: "DOWN",
        1: "FLAT",
        2: "UP"

    }

    test_df["logistic_signal"] = (
        test_df["logistic_prediction"]
        .map(label_map)
    )

    test_df["xgboost_signal"] = (
        test_df["xgboost_prediction"]
        .map(label_map)
    )


    # =====================================================
    # SAVE INITIAL PREDICTION DATA
    # =====================================================

    output_path = os.path.join(
        RESULTS_DIR,
        "backtest_predictions_standard.csv"
    )

    test_df.to_csv(
        output_path,
        index=False
    )


    print("\n" + "=" * 65)
    print("STANDARD MODEL PREDICTIONS GENERATED")
    print("=" * 65)

    print(
        f"\nRows: {len(test_df)}"
    )

    print(
        f"\nSaved to:\n{output_path}"
    )


    print("\nSample:")

    print(
        test_df[
            [
                "Date",
                "Ticker",
                "Close",
                "future_return",
                "target",
                "logistic_signal",
                "xgboost_signal"
            ]
        ].head(10)
    )


if __name__ == "__main__":
    main()