from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT /
    "data" /
    "raw" /
    "all_tickers_raw.csv"
)


def main():

    df = pd.read_csv(DATA_PATH)

    print("\nDATASET SHAPE")
    print(df.shape)

    print("\nCOLUMNS")
    print(df.columns.tolist())

    print("\nFIRST 10 ROWS")
    print(df.head(10))

    print("\nDATA TYPES")
    print(df.dtypes)

    print("\nMISSING VALUES")
    print(df.isnull().sum())

    print("\nTICKER COUNTS")
    print(
        df["Ticker"]
        .value_counts()
    )

    print("\nDATE RANGE")

    print(
        df["Date"].min()
    )

    print(
        df["Date"].max()
    )


if __name__ == "__main__":
    main()