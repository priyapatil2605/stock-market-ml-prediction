from pathlib import Path
import time

import yfinance as yf
import pandas as pd


# ---------------------------------------
# Configuration
# ---------------------------------------

TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "JPM",
    "SPY",
    "QQQ"
]

START_DATE = "2019-01-01"
END_DATE = "2026-01-01"

MAX_RETRIES = 3
RETRY_DELAY = 5


# ---------------------------------------
# Project Paths
# ---------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

RAW_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------
# Download Function
# ---------------------------------------

def download_ticker_data(ticker: str):

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            print(
                f"\nDownloading {ticker} "
                f"(Attempt {attempt}/{MAX_RETRIES})..."
            )

            data = yf.download(
                ticker,
                start=START_DATE,
                end=END_DATE,
                interval="1d",
                auto_adjust=False,
                progress=False,
                timeout=30
            )

            if not data.empty:

                data = data.reset_index()

                # Handle MultiIndex columns
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = (
                        data.columns
                        .get_level_values(0)
                    )

                data["Ticker"] = ticker

                print(
                    f"Successfully downloaded "
                    f"{ticker}"
                )

                return data

            print(
                f"No data received for {ticker}"
            )

        except Exception as error:

            print(
                f"Error downloading {ticker}:"
            )

            print(error)

        # Wait before retrying
        if attempt < MAX_RETRIES:

            print(
                f"Waiting {RETRY_DELAY} seconds "
                f"before retry..."
            )

            time.sleep(RETRY_DELAY)

    print(
        f"\nFAILED to download {ticker} "
        f"after {MAX_RETRIES} attempts."
    )

    return None


# ---------------------------------------
# Main Pipeline
# ---------------------------------------

def main():

    all_data = []

    successful_tickers = []
    failed_tickers = []

    for ticker in TICKERS:

        ticker_data = download_ticker_data(
            ticker
        )

        if ticker_data is not None:

            file_path = (
                RAW_DATA_DIR /
                f"{ticker}.csv"
            )

            ticker_data.to_csv(
                file_path,
                index=False
            )

            print(
                f"Saved {ticker}: "
                f"{len(ticker_data)} rows"
            )

            all_data.append(
                ticker_data
            )

            successful_tickers.append(
                ticker
            )

        else:

            failed_tickers.append(
                ticker
            )

        # Small delay between requests
        time.sleep(2)


    # -----------------------------------
    # Combine All Data
    # -----------------------------------

    if all_data:

        combined_data = pd.concat(
            all_data,
            ignore_index=True
        )

        combined_data = (
            combined_data
            .sort_values(
                by=[
                    "Ticker",
                    "Date"
                ]
            )
            .reset_index(
                drop=True
            )
        )

        combined_path = (
            RAW_DATA_DIR /
            "all_tickers_raw.csv"
        )

        combined_data.to_csv(
            combined_path,
            index=False
        )


    # -----------------------------------
    # Final Report
    # -----------------------------------

    print("\n")
    print("=" * 50)

    print(
        "DOWNLOAD REPORT"
    )

    print("=" * 50)

    print(
        f"\nSuccessful ({len(successful_tickers)}):"
    )

    print(
        successful_tickers
    )

    print(
        f"\nFailed ({len(failed_tickers)}):"
    )

    print(
        failed_tickers
    )

    if all_data:

        print(
            f"\nTotal Rows: "
            f"{len(combined_data)}"
        )

        print(
            "\nCombined dataset saved to:"
        )

        print(
            combined_path
        )


if __name__ == "__main__":
    main()