from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "market_features.csv"
)


def main():

    df = pd.read_csv(
        INPUT_PATH
    )

    print("\nFEATURE DATASET REPORT")
    print("=" * 50)

    print(
        f"\nShape: {df.shape}"
    )

    print(
        f"\nUnique tickers: "
        f"{df['Ticker'].nunique()}"
    )

    print(
        f"\nDate range:"
    )

    print(
        df["Date"].min()
    )

    print(
        df["Date"].max()
    )

    print(
        "\nMissing values:"
    )

    print(
        df.isna().sum().sum()
    )

    print(
        "\nFirst 5 rows:"
    )

    print(
        df.head()
    )

    print(
        "\nFeature columns:"
    )

    for column in df.columns:

        print(
            f"- {column}"
        )


if __name__ == "__main__":
    main()