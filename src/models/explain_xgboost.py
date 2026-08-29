import os
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

ML_READY_DIR = os.path.join(PROJECT_ROOT, "data", "ml_ready")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "shap")

os.makedirs(RESULTS_DIR, exist_ok=True)


def main():

    print("=" * 70)
    print("XGBOOST MODEL EXPLAINABILITY USING SHAP")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("\nLoading test data...")

    X_test = np.load(
        os.path.join(ML_READY_DIR, "X_test.npy")
    )

    feature_names = [
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

    X_test_df = pd.DataFrame(
        X_test,
        columns=feature_names
    )

    print(f"Test samples: {X_test.shape[0]}")
    print(f"Features: {X_test.shape[1]}")

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print("\nLoading XGBoost model...")

    model_path = os.path.join(
        MODEL_DIR,
        "xgboost_model.json"
    )

    model = xgb.XGBClassifier()

    model.load_model(model_path)

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # USE SAMPLE FOR FASTER SHAP
    # --------------------------------------------------------

    sample_size = min(1000, len(X_test_df))

    X_sample = X_test_df.sample(
        n=sample_size,
        random_state=42
    )

    print(f"\nUsing {sample_size} samples for SHAP analysis...")

    # --------------------------------------------------------
    # CREATE SHAP EXPLAINER
    # --------------------------------------------------------

    print("\nCalculating SHAP values...")

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_sample)

    print("SHAP values calculated.")

    # --------------------------------------------------------
    # GLOBAL FEATURE IMPORTANCE
    # --------------------------------------------------------

    print("\nCalculating global feature importance...")

    shap_array = np.array(shap_values)

    # Handle multiclass output
    if shap_array.ndim == 3:

        # Shape may be:
        # (samples, features, classes)
        # or
        # (classes, samples, features)

        if shap_array.shape[0] == sample_size:
            importance = np.mean(
                np.abs(shap_array),
                axis=(0, 2)
            )
        else:
            importance = np.mean(
                np.abs(shap_array),
                axis=(0, 1)
            )

    else:
        importance = np.mean(
            np.abs(shap_array),
            axis=0
        )

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Absolute_SHAP": importance
    })

    importance_df = importance_df.sort_values(
        "Mean_Absolute_SHAP",
        ascending=False
    )

    print("\n" + "=" * 70)
    print("GLOBAL SHAP FEATURE IMPORTANCE")
    print("=" * 70)

    print(
        importance_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    csv_path = os.path.join(
        RESULTS_DIR,
        "shap_feature_importance.csv"
    )

    importance_df.to_csv(
        csv_path,
        index=False
    )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE BAR CHART
    # --------------------------------------------------------

    plt.figure(figsize=(10, 8))

    top_features = importance_df.head(15)

    plt.barh(
        top_features["Feature"][::-1],
        top_features["Mean_Absolute_SHAP"][::-1]
    )

    plt.xlabel("Mean Absolute SHAP Value")
    plt.ylabel("Feature")

    plt.title(
        "XGBoost Global Feature Importance (SHAP)"
    )

    plt.tight_layout()

    bar_path = os.path.join(
        RESULTS_DIR,
        "shap_feature_importance.png"
    )

    plt.savefig(
        bar_path,
        dpi=300
    )

    plt.close()

    print(f"\nFeature importance chart saved to:")
    print(bar_path)

    # --------------------------------------------------------
    # SAVE SHAP SUMMARY PLOTS
    # --------------------------------------------------------

    print("\nGenerating SHAP summary plots...")

    try:

        if isinstance(shap_values, list):

            # Multiclass models
            shap.summary_plot(
                shap_values[0],
                X_sample,
                show=False
            )

        else:

            # New SHAP versions
            shap.summary_plot(
                shap_values,
                X_sample,
                show=False
            )

        plt.tight_layout()

        summary_path = os.path.join(
            RESULTS_DIR,
            "shap_summary_plot.png"
        )

        plt.savefig(
            summary_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print("SHAP summary plot saved.")

    except Exception as e:

        print("\nSummary plot warning:")
        print(e)

        print(
            "Feature importance results were still saved successfully."
        )

    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SHAP EXPLAINABILITY COMPLETE")
    print("=" * 70)

    print("\nFiles generated:")

    print(
        os.path.join(
            RESULTS_DIR,
            "shap_feature_importance.csv"
        )
    )

    print(
        os.path.join(
            RESULTS_DIR,
            "shap_feature_importance.png"
        )
    )

    print(
        os.path.join(
            RESULTS_DIR,
            "shap_summary_plot.png"
        )
    )


if __name__ == "__main__":
    main()