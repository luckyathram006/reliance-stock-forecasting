# Reliance Stock Forecasting with Machine Learning

An educational machine-learning project that studies historical Reliance Industries (`RELIANCE.NS`) data and estimates the next trading day's return and direction.

> **Important:** This project is for learning and portfolio purposes only. It is not financial or investment advice.

## Project goal

I wanted to understand whether technical indicators calculated after a market closes can help forecast the next day's stock movement. The project covers the complete workflow: data collection, exploratory analysis, feature engineering, model training, evaluation, and visualisation.

## What I built

- Downloaded daily Reliance Industries OHLCV data using `yfinance`
- Created momentum, moving-average, volatility, volume, gap, and RSI features
- Used a chronological 80/20 train/test split, so future data is not used during training
- Trained a Random Forest model for next-day direction (up/down)
- Trained a Ridge regression model for next-day returns and converted them to price estimates
- Compared each model with a simple baseline rather than relying on model scores alone

## Results

Run the project to generate the results for the latest available data. I report two honest comparisons:

1. Direction-model accuracy versus a majority-class baseline
2. Price-model mean absolute error versus the naïve prediction that tomorrow's price will equal today's close

This is important because a low error for price levels does not automatically mean a useful trading strategy. Next-day direction is difficult to predict, and results can change across periods.

## Visualisations

After running the program, these charts are created:

- `images/closing_price_and_ma.png`
- `images/predicted_vs_actual.png`

## Tools used

Python, Pandas, NumPy, Matplotlib, scikit-learn, and yfinance.

## How to run

```bash
git clone https://github.com/YOUR-USERNAME/reliance-stock-forecasting.git
cd reliance-stock-forecasting
pip install -r requirements.txt
python stock_prediction.py
```

## Repository structure

```text
reliance-stock-forecasting/
├── stock_prediction.py       # Main project code with explanatory comments
├── requirements.txt
├── images/                   # Charts created when the program runs
└── README.md
```

## What I learned

- How to work with financial time-series data in Python
- Why chronological splitting matters for time-series projects
- How to avoid data leakage while scaling features
- Why ML results should be compared with a simple baseline
- How to communicate limitations instead of overstating model performance
