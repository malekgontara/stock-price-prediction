# Stock Price Prediction - LSTM (AAPL)

Predicts the next-day closing price of Apple (AAPL) from the previous 60 days of closing prices, using a stacked LSTM in Keras.

## Tech stack

- Python 3.10+
- NumPy, pandas, matplotlib
- scikit-learn (`MinMaxScaler`)
- Keras / TensorFlow
- yfinance

## Folder structure

```
stock-price-prediction/
├── notebooks/
│   └── stock_lstm.ipynb
├── src/
│   └── stock_lstm.py
├── requirements.txt
├── .gitignore
└── README.md
```

## How to run

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate      # macOS / Linux
pip install -r requirements.txt
jupyter notebook notebooks/stock_lstm.ipynb
# or: python src/stock_lstm.py
```

Both fetch AAPL prices live from Yahoo Finance.

## What the code does

Downloads Apple's daily closing prices (2012-2020) via `yfinance`, scales them to `[0, 1]` with `MinMaxScaler`, and reshapes into rolling 60-day windows. The model is `Sequential([LSTM(50, return_sequences=True), LSTM(50), Dense(25), Dense(1)])`, compiled with Adam + MSE and trained with `EarlyStopping(patience=100)` for up to 1000 epochs. After training it predicts on the held-out 20%, inverts the scaling, computes RMSE, and plots train / actual-validation / predicted curves.
