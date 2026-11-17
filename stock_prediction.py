"""Reliance stock forecasting project.

This is an educational project. It uses the information available after a
trading day closes to estimate the following day's return and direction.
It is not investment advice.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# I kept the settings here so another student/recruiter can quickly change the
# company or time period and rerun the same analysis.
SYMBOL = "RELIANCE.NS"
START_DATE = "2015-01-01"
IMAGE_DIR = Path("images")
RANDOM_STATE = 42


def download_stock_data(symbol: str, start_date: str) -> pd.DataFrame:
    """Download daily OHLCV data from Yahoo Finance."""
    print(f"Downloading {symbol} data from {start_date}...")
    data = yf.download(symbol, start=start_date, progress=False, auto_adjust=False)

    # yfinance can return MultiIndex columns in some versions. This makes the
    # column names simple so the rest of my code works in both cases.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    if data.empty:
        raise ValueError("No stock data was downloaded. Please check the symbol or internet connection.")
    return data.dropna().copy()


def create_features(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Create features using only information known by the end of each day."""
    df = data.copy()

    # Returns help the model understand short-term price momentum.
    df["return_1d"] = df["Close"].pct_change() * 100
    df["return_5d"] = df["Close"].pct_change(5) * 100
    df["return_10d"] = df["Close"].pct_change(10) * 100

    # Moving averages show whether the current price is above or below its
    # recent trend. I use ratios instead of raw prices where possible.
    df["ma_5"] = df["Close"].rolling(5).mean()
    df["ma_10"] = df["Close"].rolling(10).mean()
    df["ma_20"] = df["Close"].rolling(20).mean()
    df["close_vs_ma5"] = df["Close"] / df["ma_5"] - 1
    df["close_vs_ma10"] = df["Close"] / df["ma_10"] - 1
    df["close_vs_ma20"] = df["Close"] / df["ma_20"] - 1

    # Volatility and volume may indicate how uncertain or active the market was.
    df["volatility_10d"] = df["return_1d"].rolling(10).std()
    df["volume_change"] = df["Volume"].pct_change() * 100
    df["volume_ma5"] = df["Volume"].rolling(5).mean()
    df["volume_ratio"] = df["Volume"] / df["volume_ma5"]

    # Daily range and gap use data available after that day's market close.
    df["daily_range"] = (df["High"] - df["Low"]) / df["Close"] * 100
    df["opening_gap"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1) * 100

    # RSI is a common technical indicator. A 14-day rolling average is used.
    price_change = df["Close"].diff()
    average_gain = price_change.clip(lower=0).rolling(14).mean()
    average_loss = (-price_change.clip(upper=0)).rolling(14).mean()
    relative_strength = average_gain / average_loss
    df["rsi_14"] = 100 - (100 / (1 + relative_strength))

    # The target is tomorrow's *return*, not tomorrow's raw price. This is a
    # fairer forecasting task because stock prices naturally remain near today.
    df["tomorrow_return"] = df["Close"].pct_change().shift(-1) * 100
    df["tomorrow_direction"] = (df["tomorrow_return"] > 0).astype(int)

    feature_columns = [
        "return_1d", "return_5d", "return_10d", "close_vs_ma5",
        "close_vs_ma10", "close_vs_ma20", "volatility_10d",
        "volume_change", "volume_ratio", "daily_range", "opening_gap", "rsi_14",
    ]

    # A zero/invalid volume can create infinity in percentage calculations.
    # I turn those values into missing values before removing incomplete rows.
    df = df.replace([np.inf, -np.inf], np.nan)

    # dropna removes the first few rolling-window rows and the final row, for
    # which tomorrow's actual price is not yet available.
    return df.dropna(subset=feature_columns + ["tomorrow_return"]).copy(), feature_columns


def chronological_split(df: pd.DataFrame, features: list[str]):
    """Split old data for training and recent data for testing without shuffling."""
    split_index = int(len(df) * 0.80)
    train, test = df.iloc[:split_index], df.iloc[split_index:]
    print(f"Training period: {train.index[0].date()} to {train.index[-1].date()}")
    print(f"Testing period:  {test.index[0].date()} to {test.index[-1].date()}")
    return (
        train[features], test[features],
        train["tomorrow_direction"], test["tomorrow_direction"],
        train["tomorrow_return"], test["tomorrow_return"], test["Close"],
    )


def save_eda_chart(df: pd.DataFrame) -> None:
    """Save one simple chart that can be displayed in the GitHub README."""
    IMAGE_DIR.mkdir(exist_ok=True)
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df["Close"], label="Reliance closing price", linewidth=1.2)
    plt.plot(df.index, df["ma_20"], label="20-day moving average", linewidth=1)
    plt.title("Reliance Industries closing price and 20-day moving average")
    plt.xlabel("Date")
    plt.ylabel("Price (INR)")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "closing_price_and_ma.png", dpi=160)
    plt.close()


def main() -> None:
    data = download_stock_data(SYMBOL, START_DATE)
    df, features = create_features(data)
    print(f"\nUsable observations after feature engineering: {len(df)}")
    save_eda_chart(df)

    x_train, x_test, y_train_dir, y_test_dir, y_train_ret, y_test_ret, test_close = chronological_split(df, features)

    # StandardScaler is fitted only on training data. This avoids accidentally
    # using information from the test period while training the model.
    direction_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=RANDOM_STATE)),
    ])
    direction_model.fit(x_train, y_train_dir)
    direction_predictions = direction_model.predict(x_test)

    # The basic baseline predicts the most common direction from training data.
    # A useful ML model should beat this, not only report its own accuracy.
    baseline_direction = int(y_train_dir.mean() >= 0.5)
    baseline_accuracy = accuracy_score(y_test_dir, np.full(len(y_test_dir), baseline_direction))
    model_accuracy = accuracy_score(y_test_dir, direction_predictions)

    print("\nDIRECTION PREDICTION RESULTS")
    print(f"Baseline accuracy:     {baseline_accuracy:.2%}")
    print(f"Random Forest accuracy:{model_accuracy:.2%}")
    print(classification_report(y_test_dir, direction_predictions, target_names=["Down", "Up"], zero_division=0))

    # Ridge regression predicts percentage return. I convert it back to a price
    # only for reporting, so it is easy to understand the average error in INR.
    return_model = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=10.0))])
    return_model.fit(x_train, y_train_ret)
    predicted_return = return_model.predict(x_test)
    predicted_price = test_close.to_numpy() * (1 + predicted_return / 100)
    actual_price = test_close.to_numpy() * (1 + y_test_ret.to_numpy() / 100)

    # The honest baseline is "tomorrow's close equals today's close".
    baseline_mae = mean_absolute_error(actual_price, test_close)
    model_mae = mean_absolute_error(actual_price, predicted_price)
    model_rmse = mean_squared_error(actual_price, predicted_price) ** 0.5

    print("\nNEXT-DAY PRICE RESULTS")
    print(f"Naive baseline MAE: ₹{baseline_mae:.2f}")
    print(f"Ridge model MAE:    ₹{model_mae:.2f}")
    print(f"Ridge model RMSE:   ₹{model_rmse:.2f}")

    # This graph is useful in GitHub because it clearly shows the test period.
    plt.figure(figsize=(12, 5))
    plt.plot(y_test_ret.index, actual_price, label="Actual next-day close", linewidth=1.5)
    plt.plot(y_test_ret.index, predicted_price, label="Predicted next-day close", linewidth=1.1)
    plt.title("Test-period next-day closing-price forecast")
    plt.xlabel("Date")
    plt.ylabel("Price (INR)")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "predicted_vs_actual.png", dpi=160)
    plt.close()

    print("\nCharts saved in the images folder.")
    print("This project is for learning and is not trading advice.")


if __name__ == "__main__":
    main()
