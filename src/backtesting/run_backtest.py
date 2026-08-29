import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

PREDICTIONS_PATH = os.path.join(
    BASE_DIR,
    "results",
    "backtest_predictions_standard.csv"
)

ML_DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "model_data",
    "market_ml_dataset.csv"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results",
    "backtesting"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# PERFORMANCE METRICS
# ============================================================

def calculate_metrics(returns):

    returns = pd.Series(returns).dropna()

    if len(returns) == 0:
        return {
            "Total Return (%)": 0,
            "Annualized Return (%)": 0,
            "Sharpe Ratio": 0,
            "Max Drawdown (%)": 0,
            "Win Rate (%)": 0,
            "Trading Periods": 0
        }

    # Equity curve
    equity_curve = (1 + returns).cumprod()

    # Total return
    total_return = (
        equity_curve.iloc[-1] - 1
    ) * 100

    # Annualized return
    periods = len(returns)
    years = periods / 252

    if years > 0 and equity_curve.iloc[-1] > 0:

        annualized_return = (
            equity_curve.iloc[-1] ** (1 / years) - 1
        ) * 100

    else:
        annualized_return = 0

    # Sharpe ratio
    if returns.std() != 0:

        sharpe_ratio = (
            returns.mean()
            / returns.std()
        ) * np.sqrt(252)

    else:
        sharpe_ratio = 0

    # Maximum drawdown
    running_max = equity_curve.cummax()

    drawdown = (
        equity_curve / running_max
    ) - 1

    max_drawdown = (
        drawdown.min() * 100
    )

    # Win rate
    win_rate = (
        (returns > 0).mean() * 100
    )

    return {

        "Total Return (%)":
            round(total_return, 2),

        "Annualized Return (%)":
            round(annualized_return, 2),

        "Sharpe Ratio":
            round(sharpe_ratio, 3),

        "Max Drawdown (%)":
            round(max_drawdown, 2),

        "Win Rate (%)":
            round(win_rate, 2),

        "Trading Periods":
            len(returns)
    }


# ============================================================
# CREATE NEXT-DAY RETURNS
# ============================================================

def create_next_day_returns(df):

    """
    Calculate actual next-day return.

    For each ticker:

    next_day_return =
    (Next Day Close / Current Close) - 1
    """

    df = df.copy()

    df = df.sort_values(
        ["Ticker", "Date"]
    )

    df["next_close"] = (
        df.groupby("Ticker")["Close"]
        .shift(-1)
    )

    df["next_day_return"] = (
        df["next_close"] / df["Close"]
    ) - 1

    return df


# ============================================================
# CREATE STRATEGY PORTFOLIO RETURNS
# ============================================================

def create_strategy_returns(df, signal_column):

    """
    Daily rebalanced portfolio.

    For each date:

    1. Select stocks predicted as UP
    2. Equally weight selected stocks
    3. Calculate average NEXT-DAY return

    If no stock is predicted UP:
    Portfolio stays in cash.
    Return = 0
    """

    strategy_results = []

    for date, group in df.groupby("Date"):

        selected_stocks = group[
            group[signal_column] == "UP"
        ]

        selected_stocks = selected_stocks.dropna(
            subset=["next_day_return"]
        )

        if len(selected_stocks) > 0:

            portfolio_return = (
                selected_stocks[
                    "next_day_return"
                ].mean()
            )

            number_positions = len(
                selected_stocks
            )

        else:

            portfolio_return = 0
            number_positions = 0

        strategy_results.append({

            "Date":
                date,

            "Return":
                portfolio_return,

            "Positions":
                number_positions
        })

    return pd.DataFrame(
        strategy_results
    )


# ============================================================
# EQUAL-WEIGHT BENCHMARK
# ============================================================

def create_buy_hold_returns(df):

    """
    Daily equal-weight benchmark.

    For every date:

    Portfolio return =
    average NEXT-DAY return of all stocks.
    """

    benchmark = (

        df.dropna(
            subset=["next_day_return"]
        )

        .groupby("Date")[
            "next_day_return"
        ]

        .mean()

        .reset_index()

    )

    benchmark.columns = [

        "Date",

        "Return"

    ]

    return benchmark


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RUNNING CORRECTED PORTFOLIO BACKTEST")
    print("=" * 70)


    # ========================================================
    # LOAD PREDICTIONS
    # ========================================================

    print("\nLoading predictions...")

    predictions_df = pd.read_csv(
        PREDICTIONS_PATH
    )

    predictions_df["Date"] = pd.to_datetime(
        predictions_df["Date"]
    )


    # ========================================================
    # LOAD MARKET DATA
    # ========================================================

    print("Loading market data...")

    market_df = pd.read_csv(
        ML_DATA_PATH
    )

    market_df["Date"] = pd.to_datetime(
        market_df["Date"]
    )

    market_df = market_df.sort_values(
        ["Ticker", "Date"]
    ).reset_index(
        drop=True
    )


    # ========================================================
    # CREATE ACTUAL NEXT-DAY RETURNS
    # ========================================================

    print(
        "Calculating actual next-day returns..."
    )

    market_df = create_next_day_returns(
        market_df
    )


    # ========================================================
    # MERGE PREDICTIONS WITH MARKET DATA
    # ========================================================

    print(
        "Merging predictions with actual returns..."
    )

    prediction_columns = [

        "Date",
        "Ticker",
        "logistic_signal",
        "xgboost_signal"

    ]

    predictions_df = predictions_df[
        prediction_columns
    ]

    df = pd.merge(

        predictions_df,

        market_df[
            [
                "Date",
                "Ticker",
                "Close",
                "next_day_return"
            ]
        ],

        on=[
            "Date",
            "Ticker"
        ],

        how="inner"

    )


    # ========================================================
    # SORT DATA
    # ========================================================

    df = df.sort_values(

        ["Date", "Ticker"]

    ).reset_index(

        drop=True

    )


    print(
        f"\nTotal rows: {len(df)}"
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
        f"Trading dates: "
        f"{df['Date'].nunique()}"
    )


    # ========================================================
    # LOGISTIC REGRESSION STRATEGY
    # ========================================================

    print(
        "\nBacktesting Logistic Regression..."
    )

    logistic_df = (

        create_strategy_returns(

            df,

            "logistic_signal"

        )

    )

    logistic_metrics = (

        calculate_metrics(

            logistic_df["Return"]

        )

    )


    # ========================================================
    # XGBOOST STRATEGY
    # ========================================================

    print(
        "Backtesting XGBoost..."
    )

    xgboost_df = (

        create_strategy_returns(

            df,

            "xgboost_signal"

        )

    )

    xgboost_metrics = (

        calculate_metrics(

            xgboost_df["Return"]

        )

    )


    # ========================================================
    # EQUAL-WEIGHT BENCHMARK
    # ========================================================

    print(
        "Calculating Equal-Weight Benchmark..."
    )

    buy_hold_df = (

        create_buy_hold_returns(

            df

        )

    )

    buy_hold_metrics = (

        calculate_metrics(

            buy_hold_df["Return"]

        )

    )


    # ========================================================
    # RESULTS TABLE
    # ========================================================

    results = pd.DataFrame([

        {

            "Strategy":
                "Logistic Regression",

            **logistic_metrics

        },

        {

            "Strategy":
                "XGBoost",

            **xgboost_metrics

        },

        {

            "Strategy":
                "Equal Weight Benchmark",

            **buy_hold_metrics

        }

    ])


    print("\n")

    print("=" * 90)

    print("CORRECTED BACKTEST RESULTS")

    print("=" * 90)


    print(

        results.to_string(

            index=False

        )

    )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_path = os.path.join(

        RESULTS_DIR,

        "backtest_results.csv"

    )

    results.to_csv(

        results_path,

        index=False

    )


    # ========================================================
    # CREATE EQUITY CURVES
    # ========================================================

    logistic_df["Equity"] = (

        1 +

        logistic_df["Return"]

    ).cumprod()


    xgboost_df["Equity"] = (

        1 +

        xgboost_df["Return"]

    ).cumprod()


    buy_hold_df["Equity"] = (

        1 +

        buy_hold_df["Return"]

    ).cumprod()


    # ========================================================
    # PLOT EQUITY CURVES
    # ========================================================

    plt.figure(

        figsize=(12, 6)

    )


    plt.plot(

        logistic_df["Date"],

        logistic_df["Equity"],

        label="Logistic Regression"

    )


    plt.plot(

        xgboost_df["Date"],

        xgboost_df["Equity"],

        label="XGBoost"

    )


    plt.plot(

        buy_hold_df["Date"],

        buy_hold_df["Equity"],

        label="Equal Weight Benchmark"

    )


    plt.title(

        "Corrected Portfolio Backtest Equity Curves"

    )


    plt.xlabel(

        "Date"

    )


    plt.ylabel(

        "Portfolio Value"

    )


    plt.legend()


    plt.grid(

        True

    )


    plt.tight_layout()


    plot_path = os.path.join(

        RESULTS_DIR,

        "equity_curves.png"

    )


    plt.savefig(

        plot_path,

        dpi=300

    )


    plt.close()


    # ========================================================
    # SAVE DETAILED RETURNS
    # ========================================================

    logistic_df.to_csv(

        os.path.join(

            RESULTS_DIR,

            "logistic_daily_returns.csv"

        ),

        index=False

    )


    xgboost_df.to_csv(

        os.path.join(

            RESULTS_DIR,

            "xgboost_daily_returns.csv"

        ),

        index=False

    )


    buy_hold_df.to_csv(

        os.path.join(

            RESULTS_DIR,

            "benchmark_daily_returns.csv"

        ),

        index=False

    )


    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("\n")

    print("=" * 70)

    print("CORRECTED BACKTEST COMPLETE")

    print("=" * 70)


    print(

        "\nResults saved to:"

    )


    print(

        results_path

    )


    print(

        "\nEquity curve saved to:"

    )


    print(

        plot_path

    )


if __name__ == "__main__":

    main()