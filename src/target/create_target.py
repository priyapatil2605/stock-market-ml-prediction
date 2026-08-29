from pathlib import Path
import numpy as np
import pandas as pd


# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "market_features.csv"
)

TARGET_DIR = (
    PROJECT_ROOT
    / "data"
    / "model_data"
)

TARGET_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_PATH = (
    TARGET_DIR
    / "market_ml_dataset.csv"
)


# ==========================================
# SETTINGS
# ==========================================

FORECAST_DAYS = 5

UP_THRESHOLD = 0.02
DOWN_THRESHOLD = -0.02


# ==========================================
# MAIN
# ==========================================

def main():

    print("Loading feature dataset...")

    df = pd.read_csv(
        INPUT_PATH,
        parse_dates=["Date"]
    )

    print(
        f"Initial dataset shape: {df.shape}"
    )


    # ======================================
    # SORT DATA
    # ======================================

    df = (
        df
        .sort_values(
            by=["Ticker", "Date"]
        )
        .reset_index(drop=True)
    )


    # ======================================
    # FUTURE CLOSE PRICE
    # ======================================

    print(
        f"\nCreating {FORECAST_DAYS}-day "
        f"future prices..."
    )

    df["future_close"] = (
        df
        .groupby("Ticker")["Close"]
        .shift(-FORECAST_DAYS)
    )


    # ======================================
    # FUTURE RETURN
    # ======================================

    df["future_return"] = (
        (
            df["future_close"]
            - df["Close"]
        )
        / df["Close"]
    )


    # ======================================
    # CREATE TARGET LABEL
    # ======================================

    print(
        "\nCreating target labels..."
    )

    conditions = [
        df["future_return"] > UP_THRESHOLD,

        df["future_return"] < DOWN_THRESHOLD
    ]

    choices = [
        "UP",
        "DOWN"
    ]

    df["target"] = np.select(
        conditions,
        choices,
        default="FLAT"
    )


    # ======================================
    # REMOVE ROWS WITHOUT FUTURE DATA
    # ======================================

    before_drop = len(df)

    df = (
        df
        .dropna(
            subset=[
                "future_close",
                "future_return"
            ]
        )
        .reset_index(drop=True)
    )

    removed_rows = (
        before_drop
        - len(df)
    )


    # ======================================
    # VALIDATION
    # ======================================

    print("\n" + "=" * 50)
    print("TARGET VALIDATION")
    print("=" * 50)

    print(
        f"Rows removed (no future data): "
        f"{removed_rows}"
    )

    print(
        f"Final dataset shape: "
        f"{df.shape}"
    )

    print(
        f"Unique tickers: "
        f"{df['Ticker'].nunique()}"
    )

    print(
        "\nTarget distribution:"
    )

    print(
        df["target"]
        .value_counts()
    )

    print(
        "\nTarget percentages:"
    )

    print(
        (
            df["target"]
            .value_counts(
                normalize=True
            )
            * 100
        )
        .round(2)
    )


    # ======================================
    # CHECK MISSING VALUES
    # ======================================

    print(
        f"\nTotal missing values: "
        f"{df.isna().sum().sum()}"
    )


    # ======================================
    # SAMPLE DATA
    # ======================================

    print(
        "\nSample target data:"
    )

    print(
        df[
            [
                "Date",
                "Ticker",
                "Close",
                "future_close",
                "future_return",
                "target"
            ]
        ].head(10)
    )


    # ======================================
    # SAVE
    # ======================================

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )


    print("\n" + "=" * 50)
    print("PHASE 3 COMPLETE")
    print("=" * 50)

    print(
        f"\nSaved ML dataset to:\n"
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()