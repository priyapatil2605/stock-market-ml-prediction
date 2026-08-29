import os
import sys
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from xgboost import XGBClassifier


warnings.filterwarnings("ignore")


# ============================================================
# PATH CONFIGURATION
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(
    os.path.join(CURRENT_DIR, "..", "..")
)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "model_data",
    "market_ml_dataset.csv"
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "walk_forward"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# FEATURE CONFIGURATION
# ============================================================

FEATURE_COLUMNS = [
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


TARGET_MAPPING = {
    "DOWN": 0,
    "FLAT": 1,
    "UP": 2
}


# ============================================================
# METRICS FUNCTION
# ============================================================

def calculate_metrics(y_true, y_pred):

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ),
        "f1_score": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )
    }


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print("=" * 75)
    print("WALK-FORWARD VALIDATION")
    print("=" * 75)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("\nLoading ML dataset...")

    if not os.path.exists(DATA_PATH):
        print(f"\nERROR: Dataset not found:\n{DATA_PATH}")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values(
        ["Date", "Ticker"]
    ).reset_index(drop=True)

    print(f"Dataset shape: {df.shape}")
    print(
        f"Date range: "
        f"{df['Date'].min().date()} "
        f"to "
        f"{df['Date'].max().date()}"
    )

    # --------------------------------------------------------
    # VALIDATE FEATURES
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in df.columns
    ]

    if missing_features:

        print("\nERROR: Missing feature columns:")

        for feature in missing_features:
            print(f"- {feature}")

        sys.exit(1)

    # --------------------------------------------------------
    # PREPARE TARGET
    # --------------------------------------------------------

    print("\nEncoding target labels...")

    df["target_encoded"] = (
        df["target"]
        .map(TARGET_MAPPING)
    )

    if df["target_encoded"].isna().sum() > 0:

        print(
            "\nERROR: Unknown target labels found."
        )

        print(
            df.loc[
                df["target_encoded"].isna(),
                "target"
            ].unique()
        )

        sys.exit(1)

    # --------------------------------------------------------
    # CREATE YEAR-BASED FOLDS
    # --------------------------------------------------------

    years = sorted(
        df["Date"].dt.year.unique()
    )

    print("\nAvailable years:")
    print(years)

    # Minimum 2 years of training
    # Then move forward year by year

    folds = []

    for test_year in years[2:]:

        train_years = [
            year
            for year in years
            if year < test_year
        ]

        folds.append({
            "train_years": train_years,
            "test_year": test_year
        })

    print("\nWalk-forward folds:")

    for i, fold in enumerate(folds, start=1):

        print(
            f"Fold {i}: "
            f"Train {fold['train_years'][0]}"
            f"-{fold['train_years'][-1]} "
            f"| Test {fold['test_year']}"
        )

    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    results = []

    # ========================================================
    # WALK-FORWARD LOOP
    # ========================================================

    for fold_number, fold in enumerate(
        folds,
        start=1
    ):

        print("\n" + "=" * 75)
        print(
            f"FOLD {fold_number}"
        )
        print("=" * 75)

        train_years = fold["train_years"]
        test_year = fold["test_year"]

        # ----------------------------------------------------
        # SPLIT DATA
        # ----------------------------------------------------

        train_df = df[
            df["Date"].dt.year.isin(
                train_years
            )
        ].copy()

        test_df = df[
            df["Date"].dt.year == test_year
        ].copy()

        X_train = train_df[
            FEATURE_COLUMNS
        ].values

        X_test = test_df[
            FEATURE_COLUMNS
        ].values

        y_train = train_df[
            "target_encoded"
        ].values

        y_test = test_df[
            "target_encoded"
        ].values

        print(
            f"\nTraining period: "
            f"{train_df['Date'].min().date()} "
            f"to "
            f"{train_df['Date'].max().date()}"
        )

        print(
            f"Testing period: "
            f"{test_df['Date'].min().date()} "
            f"to "
            f"{test_df['Date'].max().date()}"
        )

        print(
            f"Training samples: "
            f"{len(train_df)}"
        )

        print(
            f"Testing samples: "
            f"{len(test_df)}"
        )

        # ----------------------------------------------------
        # SCALE FEATURES
        # ----------------------------------------------------

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(
            X_train
        )

        X_test_scaled = scaler.transform(
            X_test
        )

        # ====================================================
        # LOGISTIC REGRESSION
        # ====================================================

        print(
            "\nTraining Logistic Regression..."
        )

        logistic_model = LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            random_state=42
        )

        logistic_model.fit(
            X_train_scaled,
            y_train
        )

        logistic_predictions = (
            logistic_model.predict(
                X_test_scaled
            )
        )

        logistic_metrics = calculate_metrics(
            y_test,
            logistic_predictions
        )

        print(
            f"Logistic Accuracy: "
            f"{logistic_metrics['accuracy']:.4f}"
        )

        print(
            f"Logistic F1: "
            f"{logistic_metrics['f1_score']:.4f}"
        )

        results.append({

            "fold": fold_number,

            "train_start":
                train_df["Date"].min().strftime(
                    "%Y-%m-%d"
                ),

            "train_end":
                train_df["Date"].max().strftime(
                    "%Y-%m-%d"
                ),

            "test_year":
                test_year,

            "model":
                "Logistic Regression",

            "accuracy":
                logistic_metrics["accuracy"],

            "precision":
                logistic_metrics["precision"],

            "recall":
                logistic_metrics["recall"],

            "f1_score":
                logistic_metrics["f1_score"]
        })

        # ====================================================
        # XGBOOST
        # ====================================================

        print(
            "\nTraining XGBoost..."
        )

        xgb_model = XGBClassifier(

            objective="multi:softprob",

            num_class=3,

            n_estimators=200,

            max_depth=5,

            learning_rate=0.05,

            subsample=0.8,

            colsample_bytree=0.8,

            random_state=42,

            eval_metric="mlogloss",

            n_jobs=-1
        )

        xgb_model.fit(
            X_train,
            y_train
        )

        xgb_predictions = (
            xgb_model.predict(
                X_test
            )
        )

        xgb_metrics = calculate_metrics(
            y_test,
            xgb_predictions
        )

        print(
            f"XGBoost Accuracy: "
            f"{xgb_metrics['accuracy']:.4f}"
        )

        print(
            f"XGBoost F1: "
            f"{xgb_metrics['f1_score']:.4f}"
        )

        results.append({

            "fold": fold_number,

            "train_start":
                train_df["Date"].min().strftime(
                    "%Y-%m-%d"
                ),

            "train_end":
                train_df["Date"].max().strftime(
                    "%Y-%m-%d"
                ),

            "test_year":
                test_year,

            "model":
                "XGBoost",

            "accuracy":
                xgb_metrics["accuracy"],

            "precision":
                xgb_metrics["precision"],

            "recall":
                xgb_metrics["recall"],

            "f1_score":
                xgb_metrics["f1_score"]
        })

    # ========================================================
    # RESULTS DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    print("\n" + "=" * 75)
    print("WALK-FORWARD VALIDATION RESULTS")
    print("=" * 75)

    print(
        "\nResults for each fold:\n"
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    # ========================================================
    # AVERAGE RESULTS
    # ========================================================

    summary = (
        results_df
        .groupby("model")[
            [
                "accuracy",
                "precision",
                "recall",
                "f1_score"
            ]
        ]
        .mean()
        .reset_index()
    )

    summary = summary.sort_values(
        "f1_score",
        ascending=False
    )

    print("\n" + "=" * 75)
    print("AVERAGE WALK-FORWARD PERFORMANCE")
    print("=" * 75)

    summary_display = summary.copy()

    for column in [
        "accuracy",
        "precision",
        "recall",
        "f1_score"
    ]:

        summary_display[column] = (
            summary_display[column] * 100
        )

    print()

    print(
        summary_display.to_string(
            index=False,
            formatters={
                "accuracy":
                    "{:.2f}%".format,

                "precision":
                    "{:.2f}%".format,

                "recall":
                    "{:.2f}%".format,

                "f1_score":
                    "{:.2f}%".format
            }
        )
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_path = os.path.join(
        RESULTS_DIR,
        "walk_forward_results.csv"
    )

    summary_path = os.path.join(
        RESULTS_DIR,
        "walk_forward_summary.csv"
    )

    json_path = os.path.join(
        RESULTS_DIR,
        "walk_forward_summary.json"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    summary_records = (
        summary
        .round(6)
        .to_dict(
            orient="records"
        )
    )

    with open(
        json_path,
        "w"
    ) as file:

        json.dump(
            summary_records,
            file,
            indent=4
        )

    print("\n" + "=" * 75)
    print("WALK-FORWARD VALIDATION COMPLETE")
    print("=" * 75)

    print(
        "\nDetailed results saved to:"
    )

    print(results_path)

    print(
        "\nSummary saved to:"
    )

    print(summary_path)

    print(
        "\nJSON summary saved to:"
    )

    print(json_path)


if __name__ == "__main__":
    main()