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
    / "processed"
    / "market_data_clean.csv"
)

FEATURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
)

FEATURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_PATH = (
    FEATURE_DIR
    / "market_features.csv"
)


# ==========================================
# FEATURE ENGINEERING FUNCTION
# ==========================================

def create_features(group):

    # Save the ticker name from the group
    ticker = group.name

    # Create a copy to avoid modifying original data
    group = group.copy()

    # Restore Ticker because include_groups=False
    # removes the grouping column
    group["Ticker"] = ticker


    # ======================================
    # 1. RETURN FEATURES
    # ======================================

    # Daily percentage return
    group["daily_return"] = (
        group["Close"].pct_change()
    )

    # Lagged returns
    for lag in [1, 2, 3, 5, 10]:

        group[f"return_lag_{lag}"] = (
            group["daily_return"].shift(lag)
        )


    # ======================================
    # 2. MOVING AVERAGES
    # ======================================

    for window in [5, 10, 20, 50]:

        group[f"sma_{window}"] = (
            group["Close"]
            .rolling(
                window=window
            )
            .mean()
        )


    # Price relative to moving averages
    for window in [10, 20, 50]:

        group[f"price_vs_sma_{window}"] = (
            group["Close"]
            / group[f"sma_{window}"]
            - 1
        )


    # Moving average ratio
    group["ma_10_20_ratio"] = (
        group["sma_10"]
        / group["sma_20"]
    )


    # ======================================
    # 3. RSI (14)
    # ======================================

    delta = (
        group["Close"].diff()
    )

    gain = (
        delta.clip(lower=0)
    )

    loss = (
        -delta.clip(upper=0)
    )

    avg_gain = (
        gain
        .rolling(window=14)
        .mean()
    )

    avg_loss = (
        loss
        .rolling(window=14)
        .mean()
    )

    rs = (
        avg_gain / avg_loss
    )

    group["rsi_14"] = (
        100
        - (
            100
            / (1 + rs)
        )
    )


    # ======================================
    # 4. MACD
    # ======================================

    ema_12 = (
        group["Close"]
        .ewm(
            span=12,
            adjust=False
        )
        .mean()
    )

    ema_26 = (
        group["Close"]
        .ewm(
            span=26,
            adjust=False
        )
        .mean()
    )

    group["macd"] = (
        ema_12 - ema_26
    )

    group["macd_signal"] = (
        group["macd"]
        .ewm(
            span=9,
            adjust=False
        )
        .mean()
    )

    group["macd_hist"] = (
        group["macd"]
        - group["macd_signal"]
    )


    # ======================================
    # 5. BOLLINGER BANDS
    # ======================================

    bb_middle = (
        group["Close"]
        .rolling(window=20)
        .mean()
    )

    bb_std = (
        group["Close"]
        .rolling(window=20)
        .std()
    )

    group["bb_middle"] = (
        bb_middle
    )

    group["bb_upper"] = (
        bb_middle
        + (2 * bb_std)
    )

    group["bb_lower"] = (
        bb_middle
        - (2 * bb_std)
    )

    group["bb_width"] = (
        (
            group["bb_upper"]
            - group["bb_lower"]
        )
        / bb_middle
    )

    group["bb_position"] = (
        (
            group["Close"]
            - group["bb_lower"]
        )
        /
        (
            group["bb_upper"]
            - group["bb_lower"]
        )
    )


    # ======================================
    # 6. VOLATILITY FEATURES
    # ======================================

    group["volatility_10"] = (
        group["daily_return"]
        .rolling(window=10)
        .std()
    )

    group["volatility_20"] = (
        group["daily_return"]
        .rolling(window=20)
        .std()
    )


    # ======================================
    # 7. ATR (14)
    # ======================================

    previous_close = (
        group["Close"].shift(1)
    )

    high_low = (
        group["High"]
        - group["Low"]
    )

    high_previous_close = (
        (
            group["High"]
            - previous_close
        )
        .abs()
    )

    low_previous_close = (
        (
            group["Low"]
            - previous_close
        )
        .abs()
    )

    true_range = pd.concat(
        [
            high_low,
            high_previous_close,
            low_previous_close
        ],
        axis=1
    ).max(axis=1)

    group["atr_14"] = (
        true_range
        .rolling(window=14)
        .mean()
    )


    # ======================================
    # 8. VOLUME FEATURES
    # ======================================

    group["volume_sma_10"] = (
        group["Volume"]
        .rolling(window=10)
        .mean()
    )

    group["volume_sma_20"] = (
        group["Volume"]
        .rolling(window=20)
        .mean()
    )

    group["volume_ratio"] = (
        group["Volume"]
        / group["volume_sma_20"]
    )


    # ======================================
    # RETURN FEATURED GROUP
    # ======================================

    return group


# ==========================================
# MAIN FUNCTION
# ==========================================

def main():

    print("Loading cleaned market data...")

    df = pd.read_csv(
        INPUT_PATH,
        parse_dates=["Date"]
    )

    print(
        f"Initial dataset shape: "
        f"{df.shape}"
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
    # CREATE FEATURES PER TICKER
    # ======================================

    print(
        "\nCreating features for each ticker..."
    )

    feature_df = (
        df
        .groupby(
            "Ticker",
            group_keys=False
        )
        .apply(
            create_features,
            include_groups=False
        )
        .reset_index(drop=True)
    )


    # ======================================
    # CHECK DATA BEFORE CLEANING NaNs
    # ======================================

    print(
        f"\nShape before removing NaNs: "
        f"{feature_df.shape}"
    )


    # ======================================
    # REMOVE INFINITE VALUES
    # ======================================

    feature_df = (
        feature_df
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )


    # ======================================
    # REMOVE ROWS WITHOUT ENOUGH HISTORY
    # ======================================

    before_dropna = (
        len(feature_df)
    )

    feature_df = (
        feature_df
        .dropna()
        .reset_index(drop=True)
    )

    removed_rows = (
        before_dropna
        - len(feature_df)
    )


    # ======================================
    # FINAL SORT
    # ======================================

    feature_df = (
        feature_df
        .sort_values(
            by=["Ticker", "Date"]
        )
        .reset_index(drop=True)
    )


    # ======================================
    # FEATURE VALIDATION
    # ======================================

    print("\n" + "=" * 50)
    print("FEATURE VALIDATION")
    print("=" * 50)

    print(
        f"Rows removed due to "
        f"feature history: {removed_rows}"
    )

    print(
        f"Final dataset shape: "
        f"{feature_df.shape}"
    )

    print(
        f"Unique tickers: "
        f"{feature_df['Ticker'].nunique()}"
    )


    # --------------------------------------
    # Missing Values
    # --------------------------------------

    missing_values = (
        feature_df
        .isna()
        .sum()
        .sum()
    )

    print(
        f"\nMissing values: "
        f"{missing_values}"
    )


    # --------------------------------------
    # Infinite Values
    # --------------------------------------

    numeric_df = (
        feature_df
        .select_dtypes(
            include=[np.number]
        )
    )

    infinite_values = (
        np.isinf(numeric_df)
        .sum()
        .sum()
    )

    print(
        f"Infinite values: "
        f"{infinite_values}"
    )


    # --------------------------------------
    # Rows Per Ticker
    # --------------------------------------

    print(
        "\nRows per ticker:"
    )

    print(
        feature_df["Ticker"]
        .value_counts()
        .sort_index()
    )


    # ======================================
    # SAVE DATASET
    # ======================================

    feature_df.to_csv(
        OUTPUT_PATH,
        index=False
    )


    # ======================================
    # FINAL REPORT
    # ======================================

    print("\n" + "=" * 50)
    print("PHASE 2 FEATURE ENGINEERING COMPLETE")
    print("=" * 50)

    print(
        f"\nSaved feature dataset to:\n"
        f"{OUTPUT_PATH}"
    )

    print(
        f"\nTotal columns: "
        f"{len(feature_df.columns)}"
    )

    print(
        f"Total rows: "
        f"{len(feature_df)}"
    )

    print(
        "\nFeature columns created:"
    )

    feature_columns = [
        column
        for column in feature_df.columns
        if column not in [
            "Date",
            "Ticker"
        ]
    ]

    for column in feature_columns:

        print(
            f"- {column}"
        )


if __name__ == "__main__":
    main()