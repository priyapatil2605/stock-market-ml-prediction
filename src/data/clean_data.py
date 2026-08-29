from pathlib import Path
import pandas as pd


# ---------------------------------------
# Project Paths
# ---------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "all_tickers_raw.csv"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------
# Cleaning Pipeline
# ---------------------------------------

def main():

    print("Loading raw dataset...")

    df = pd.read_csv(RAW_PATH)

    print(f"Initial rows: {len(df)}")

    # -----------------------------------
    # 1. Convert Date to datetime
    # -----------------------------------

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    # Check invalid dates
    invalid_dates = df["Date"].isna().sum()

    print(
        f"Invalid dates found: "
        f"{invalid_dates}"
    )

    # Remove invalid dates if any
    df = df.dropna(
        subset=["Date"]
    )

    # -----------------------------------
    # 2. Sort by Ticker and Date
    # -----------------------------------

    df = df.sort_values(
        by=["Ticker", "Date"]
    )

    # -----------------------------------
    # 3. Remove duplicate records
    # -----------------------------------

    before_duplicates = len(df)

    df = df.drop_duplicates(
        subset=["Ticker", "Date"],
        keep="first"
    )

    duplicates_removed = (
        before_duplicates - len(df)
    )

    print(
        f"Duplicates removed: "
        f"{duplicates_removed}"
    )

    # -----------------------------------
    # 4. Remove missing OHLCV values
    # -----------------------------------

    before_missing = len(df)

    df = df.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]
    )

    missing_rows_removed = (
        before_missing - len(df)
    )

    print(
        f"Rows removed due to "
        f"missing OHLCV values: "
        f"{missing_rows_removed}"
    )

    # -----------------------------------
    # 5. Reset index
    # -----------------------------------

    df = df.reset_index(
        drop=True
    )

    # -----------------------------------
    # 6. Save processed dataset
    # -----------------------------------

    output_path = (
        PROCESSED_DIR
        / "market_data_clean.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    # -----------------------------------
    # Final Report
    # -----------------------------------

    print("\n" + "=" * 45)
    print("CLEANING COMPLETE")
    print("=" * 45)

    print(
        f"Final rows: {len(df)}"
    )

    print(
        f"Final columns: "
        f"{len(df.columns)}"
    )

    print(
        f"Date range: "
        f"{df['Date'].min().date()} "
        f"to "
        f"{df['Date'].max().date()}"
    )

    print(
        f"Unique tickers: "
        f"{df['Ticker'].nunique()}"
    )

    print(
        f"\nSaved processed dataset to:"
    )

    print(output_path)


if __name__ == "__main__":
    main()