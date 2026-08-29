from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler


# ==================================================
# PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "model_data"
    / "market_ml_dataset.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "lstm_ready"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==================================================
# SETTINGS
# ==================================================

SEQUENCE_LENGTH = 20


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


# ==================================================
# CREATE SEQUENCES
# ==================================================

def create_sequences(
    df,
    feature_columns,
    sequence_length
):

    X_sequences = []
    y_sequences = []

    print(
        "\nCreating sequences for each ticker..."
    )


    for ticker in df["Ticker"].unique():

        ticker_df = df[
            df["Ticker"] == ticker
        ].copy()

        ticker_df = ticker_df.sort_values(
            "Date"
        ).reset_index(
            drop=True
        )


        features = ticker_df[
            feature_columns
        ].values


        targets = ticker_df[
            "target_encoded"
        ].values


        for i in range(
            sequence_length,
            len(ticker_df)
        ):

            sequence = features[
                i - sequence_length:i
            ]

            target = targets[i]


            X_sequences.append(
                sequence
            )

            y_sequences.append(
                target
            )


        print(
            f"{ticker}: "
            f"{len(ticker_df) - sequence_length} "
            f"sequences"
        )


    return (
        np.array(X_sequences),
        np.array(y_sequences)
    )


# ==================================================
# MAIN
# ==================================================

def main():

    print(
        "Loading ML dataset..."
    )

    df = pd.read_csv(
        INPUT_FILE
    )


    # ----------------------------------------------
    # PREPARE DATA
    # ----------------------------------------------

    df["Date"] = pd.to_datetime(
        df["Date"]
    )


    df = df.sort_values(
        ["Ticker", "Date"]
    ).reset_index(
        drop=True
    )


    print(
        f"Initial shape: {df.shape}"
    )


    # ----------------------------------------------
    # ENCODE TARGET
    # ----------------------------------------------

    print(
        "\nEncoding target labels..."
    )


    df["target_encoded"] = df[
        "target"
    ].map(
        TARGET_MAPPING
    )


    # ----------------------------------------------
    # TIME SPLIT
    # ----------------------------------------------

    print(
        "\nCreating time-based split..."
    )


    split_date = pd.Timestamp(
        "2024-01-01"
    )


    train_df = df[
        df["Date"] < split_date
    ].copy()


    test_df = df[
        df["Date"] >= split_date
    ].copy()


    print(
        f"Training rows: {len(train_df)}"
    )

    print(
        f"Testing rows: {len(test_df)}"
    )


    # ----------------------------------------------
    # SCALE FEATURES
    # ----------------------------------------------

    print(
        "\nScaling features..."
    )


    scaler = StandardScaler()


    train_df[
        FEATURE_COLUMNS
    ] = scaler.fit_transform(
        train_df[
            FEATURE_COLUMNS
        ]
    )


    test_df[
        FEATURE_COLUMNS
    ] = scaler.transform(
        test_df[
            FEATURE_COLUMNS
        ]
    )


    # ----------------------------------------------
    # CREATE TRAIN SEQUENCES
    # ----------------------------------------------

    X_train, y_train = create_sequences(
        train_df,
        FEATURE_COLUMNS,
        SEQUENCE_LENGTH
    )


    # ----------------------------------------------
    # CREATE TEST SEQUENCES
    # ----------------------------------------------

    X_test, y_test = create_sequences(
        test_df,
        FEATURE_COLUMNS,
        SEQUENCE_LENGTH
    )


    # ----------------------------------------------
    # VALIDATION
    # ----------------------------------------------

    print(
        "\n" + "=" * 55
    )

    print(
        "LSTM DATA PREPARATION COMPLETE"
    )

    print(
        "=" * 55
    )


    print(
        f"\nSequence length: "
        f"{SEQUENCE_LENGTH}"
    )


    print(
        f"Features per day: "
        f"{len(FEATURE_COLUMNS)}"
    )


    print(
        f"\nX_train shape: "
        f"{X_train.shape}"
    )


    print(
        f"y_train shape: "
        f"{y_train.shape}"
    )


    print(
        f"\nX_test shape: "
        f"{X_test.shape}"
    )


    print(
        f"y_test shape: "
        f"{y_test.shape}"
    )


    # ----------------------------------------------
    # SAVE DATA
    # ----------------------------------------------

    np.save(
        OUTPUT_DIR / "X_train_lstm.npy",
        X_train
    )


    np.save(
        OUTPUT_DIR / "X_test_lstm.npy",
        X_test
    )


    np.save(
        OUTPUT_DIR / "y_train_lstm.npy",
        y_train
    )


    np.save(
        OUTPUT_DIR / "y_test_lstm.npy",
        y_test
    )


    joblib.dump(
        scaler,
        OUTPUT_DIR / "lstm_scaler.pkl"
    )


    joblib.dump(
        FEATURE_COLUMNS,
        OUTPUT_DIR / "lstm_feature_names.pkl"
    )


    print(
        f"\nSaved LSTM-ready data to:"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()