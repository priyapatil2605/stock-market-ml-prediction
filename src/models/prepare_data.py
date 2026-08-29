from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler


# ==========================================
# PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "model_data"
    / "market_ml_dataset.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "ml_ready"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================
# FEATURE COLUMNS
# ==========================================

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


# ==========================================
# MAIN
# ==========================================

def main():

    print("Loading ML dataset...")

    df = pd.read_csv(
        INPUT_PATH,
        parse_dates=["Date"]
    )

    print(f"Initial shape: {df.shape}")


    # ======================================
    # SORT CHRONOLOGICALLY
    # ======================================

    df = (
        df
        .sort_values(
            by=["Date", "Ticker"]
        )
        .reset_index(drop=True)
    )


    # ======================================
    # VALIDATE FEATURES
    # ======================================

    missing_features = [
        col for col in FEATURE_COLUMNS
        if col not in df.columns
    ]

    if missing_features:

        raise ValueError(
            f"Missing features: {missing_features}"
        )


    # ======================================
    # CREATE X AND y
    # ======================================

    X = df[FEATURE_COLUMNS].copy()

    y = df["target"].copy()


    # ======================================
    # ENCODE TARGET
    # ======================================

    print("\nEncoding target labels...")

    label_encoder = LabelEncoder()

    y_encoded = label_encoder.fit_transform(y)

    print(
        "Target mapping:"
    )

    for label, value in zip(
        label_encoder.classes_,
        range(len(label_encoder.classes_))
    ):

        print(
            f"{label} -> {value}"
        )


    # ======================================
    # TIME-BASED TRAIN / TEST SPLIT
    # ======================================

    print(
        "\nCreating time-based split..."
    )

    split_date = pd.Timestamp(
        "2024-01-01"
    )

    train_mask = (
        df["Date"] < split_date
    )

    test_mask = (
        df["Date"] >= split_date
    )


    X_train = (
        X.loc[train_mask]
        .copy()
    )

    X_test = (
        X.loc[test_mask]
        .copy()
    )

    y_train = (
        y_encoded[train_mask]
    )

    y_test = (
        y_encoded[test_mask]
    )


    # ======================================
    # SCALE FEATURES
    # ======================================

    print(
        "\nScaling features..."
    )

    scaler = StandardScaler()

    # IMPORTANT:
    # Fit only on training data
    X_train_scaled = (
        scaler.fit_transform(
            X_train
        )
    )

    # Test data only transforms
    X_test_scaled = (
        scaler.transform(
            X_test
        )
    )


    # ======================================
    # SAVE DATA
    # ======================================

    np.save(
        OUTPUT_DIR / "X_train.npy",
        X_train_scaled
    )

    np.save(
        OUTPUT_DIR / "X_test.npy",
        X_test_scaled
    )

    np.save(
        OUTPUT_DIR / "y_train.npy",
        y_train
    )

    np.save(
        OUTPUT_DIR / "y_test.npy",
        y_test
    )

    joblib.dump(
        scaler,
        OUTPUT_DIR / "scaler.pkl"
    )

    joblib.dump(
        label_encoder,
        OUTPUT_DIR / "label_encoder.pkl"
    )


    # ======================================
    # SAVE METADATA
    # ======================================

    metadata = {
        "features": FEATURE_COLUMNS,
        "split_date": str(split_date.date()),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "train_start": str(
            df.loc[train_mask, "Date"]
            .min()
            .date()
        ),
        "train_end": str(
            df.loc[train_mask, "Date"]
            .max()
            .date()
        ),
        "test_start": str(
            df.loc[test_mask, "Date"]
            .min()
            .date()
        ),
        "test_end": str(
            df.loc[test_mask, "Date"]
            .max()
            .date()
        )
    }

    pd.Series(
        metadata
    ).to_json(
        OUTPUT_DIR / "metadata.json",
        indent=4
    )


    # ======================================
    # VALIDATION REPORT
    # ======================================

    print("\n" + "=" * 50)
    print("ML DATA PREPARATION COMPLETE")
    print("=" * 50)

    print(
        f"\nFeatures used: "
        f"{len(FEATURE_COLUMNS)}"
    )

    print(
        f"Training samples: "
        f"{len(X_train)}"
    )

    print(
        f"Testing samples: "
        f"{len(X_test)}"
    )

    print(
        f"\nTraining period: "
        f"{metadata['train_start']} "
        f"to {metadata['train_end']}"
    )

    print(
        f"Testing period: "
        f"{metadata['test_start']} "
        f"to {metadata['test_end']}"
    )

    print(
        f"\nX_train shape: "
        f"{X_train_scaled.shape}"
    )

    print(
        f"X_test shape: "
        f"{X_test_scaled.shape}"
    )

    print(
        f"\nSaved ML-ready data to:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()