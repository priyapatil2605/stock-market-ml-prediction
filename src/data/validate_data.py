from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT /
    "data" /
    "raw" /
    "all_tickers_raw.csv"
)


REQUIRED_COLUMNS = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Ticker"
]


def validate_columns(df):

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        print(
            "Missing Columns:"
        )

        print(
            missing_columns
        )

        return False

    print(
        "Column Validation: PASSED"
    )

    return True


def validate_missing_values(df):

    missing = (
        df[
            REQUIRED_COLUMNS
        ]
        .isnull()
        .sum()
    )

    print(
        "\nMissing Values:"
    )

    print(
        missing
    )


def validate_duplicates(df):

    duplicate_count = (
        df.duplicated(
            subset=[
                "Date",
                "Ticker"
            ]
        )
        .sum()
    )

    print(
        "\nDuplicate Date-Ticker Records:"
    )

    print(
        duplicate_count
    )


def validate_prices(df):

    invalid_prices = df[
        (
            df["Open"] <= 0
        )
        |
        (
            df["High"] <= 0
        )
        |
        (
            df["Low"] <= 0
        )
        |
        (
            df["Close"] <= 0
        )
    ]

    print(
        "\nInvalid Price Rows:"
    )

    print(
        len(
            invalid_prices
        )
    )


def validate_ohlc_logic(df):

    invalid_rows = df[
        (
            df["High"]
            < df["Low"]
        )
        |
        (
            df["High"]
            < df["Open"]
        )
        |
        (
            df["High"]
            < df["Close"]
        )
        |
        (
            df["Low"]
            > df["Open"]
        )
        |
        (
            df["Low"]
            > df["Close"]
        )
    ]

    print(
        "\nInvalid OHLC Relationships:"
    )

    print(
        len(
            invalid_rows
        )
    )


def main():

    df = pd.read_csv(
        DATA_PATH
    )

    print(
        "\nDATA VALIDATION REPORT"
    )

    print(
        "=" * 40
    )

    validate_columns(df)

    validate_missing_values(df)

    validate_duplicates(df)

    validate_prices(df)

    validate_ohlc_logic(df)


if __name__ == "__main__":
    main()