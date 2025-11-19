import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide TensorFlow warnings

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from datetime import timedelta

# -------------------------------------------------------------
# 🌆 Streamlit Page Setup
# -------------------------------------------------------------
st.set_page_config(page_title="🌍 Global Stock Predictor with Market Type", page_icon="📈", layout="wide")



# -------------------------------------------------------------
# 🌍 Title
# -------------------------------------------------------------
st.title("🌍 Global Stock & Crypto Price Predictor")
st.markdown("Predict **future prices** for any stock, crypto, or index using **LSTM deep learning** — now with automatic **Market Type Detection**!")

# -------------------------------------------------------------
# 🧠 Market Type Detector
# -------------------------------------------------------------
def detect_market_type(symbol: str) -> str:
    symbol = symbol.upper()
    if "-USD" in symbol or "-INR" in symbol:
        return "💰 Cryptocurrency"
    elif symbol.startswith("^"):
        return "📈 Market Index"
    elif symbol.endswith(".NS"):
        return "🇮🇳 Indian Stock (NSE)"
    elif symbol.endswith(".BO"):
        return "🇮🇳 Indian Stock (BSE)"
    elif symbol.endswith(".L"):
        return "🇬🇧 UK Stock (LSE)"
    elif symbol.endswith(".DE"):
        return "🇩🇪 German Stock (XETRA)"
    elif symbol.endswith(".PA"):
        return "🇫🇷 French Stock (Euronext)"
    elif symbol.endswith(".T"):
        return "🇯🇵 Japanese Stock (Tokyo)"
    elif symbol.endswith(".HK"):
        return "🇭🇰 Hong Kong Stock (HKEX)"
    elif symbol.endswith(".TO"):
        return "🇨🇦 Canadian Stock (TSX)"
    elif symbol.endswith("=X"):
        return "💱 Forex Pair"
    elif symbol.isalpha() and len(symbol) <= 5:
        return "🇺🇸 US Stock (NASDAQ/NYSE)"
    else:
        return "🌎 Global/Other Market"

# -------------------------------------------------------------
# 🎯 User Inputs
# -------------------------------------------------------------
ticker = st.text_input("🔍 Enter stock/crypto/index symbol (e.g., AAPL, TSLA, ZOMATO.NS, BTC-USD):", "AAPL").strip().upper()
n_days = st.slider("📆 Predict next how many days?", 1, 7, 3)

if ticker:
    market_type = detect_market_type(ticker)
    st.info(f"🧭 Detected Market Type: **{market_type}**")

# -------------------------------------------------------------
# 📊 Fetch Data and Predict
# -------------------------------------------------------------
if st.button("🚀 Predict"):
    with st.spinner("Fetching data and training LSTM model..."):
        try:
            data = yf.download(ticker, period="5y")

            if data.empty:
                st.error("⚠️ No data found for this ticker.")
            else:
                st.success(f"✅ Data fetched successfully for {ticker}")
                st.dataframe(data.tail())

                # Preprocess closing prices
                close_data = data["Close"].values.reshape(-1, 1)
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(close_data)

                # Create training dataset
                def create_dataset(dataset, time_step=60):
                    X, y = [], []
                    for i in range(len(dataset) - time_step - 1):
                        X.append(dataset[i:(i + time_step), 0])
                        y.append(dataset[i + time_step, 0])
                    return np.array(X), np.array(y)

                X, y = create_dataset(scaled_data)
                X = X.reshape(X.shape[0], X.shape[1], 1)

                # Build LSTM model
                model = Sequential([
                    LSTM(50, return_sequences=True, input_shape=(60, 1)),
                    LSTM(50),
                    Dense(1)
                ])
                model.compile(optimizer="adam", loss="mse")
                model.fit(X, y, epochs=8, batch_size=32, verbose=0)

                # Predict for test portion (last 20%)
                train_size = int(len(scaled_data) * 0.8)
                test_data = scaled_data[train_size - 60:]
                X_test, y_test = create_dataset(test_data)
                X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

                y_pred = model.predict(X_test, verbose=0)
                y_pred = scaler.inverse_transform(y_pred)
                y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

                # Predict future N days
                last_60 = scaled_data[-60:]
                future_predictions = []
                seq = last_60.copy()

                for _ in range(n_days):
                    pred = model.predict(seq.reshape(1, 60, 1), verbose=0)
                    future_predictions.append(pred[0][0])
                    seq = np.append(seq[1:], pred[0][0])

                future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
                future_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=n_days)

                # Create comparison DataFrame
                result_df = pd.DataFrame({
                    "Actual Price": y_test.flatten(),
                    "Predicted Price": y_pred.flatten()
                })
                result_df.index = data.index[-len(result_df):]

                # Display last few actual vs predicted values
                st.subheader(f"📊 Actual vs Predicted Prices for {ticker}")
                st.dataframe(result_df.tail(10))

                # Future predicted prices
                st.subheader(f"📈 Predicted Prices for Next {n_days} Days")
                future_df = pd.DataFrame({
                    "Date": future_dates.strftime("%Y-%m-%d"),
                    "Predicted Close Price": future_predictions.flatten()
                })
                st.dataframe(future_df)

                # -------------------------------------------------------------
                # 📉 Plot Actual vs Predicted
                # -------------------------------------------------------------
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(result_df.index, result_df["Actual Price"], label="Actual Price", linewidth=2)
                ax.plot(result_df.index, result_df["Predicted Price"], label="Predicted Price", linestyle="--", linewidth=2)
                ax.set_title(f"📊 Actual vs Predicted Prices for {ticker}", fontsize=14)
                ax.set_xlabel("Date")
                ax.set_ylabel("Price")
                ax.legend()
                st.pyplot(fig)

                # -------------------------------------------------------------
                # 📈 Append Future Prediction to Graph
                # -------------------------------------------------------------
                fig2, ax2 = plt.subplots(figsize=(10, 5))
                ax2.plot(data.index[-60:], data["Close"].tail(60), label="Past Actual Prices", color="blue")
                ax2.plot(future_dates, future_predictions, label="Future Predicted Prices", color="orange", linestyle="--")
                ax2.set_title(f"📈 Future {n_days}-Day Forecast for {ticker}", fontsize=14)
                ax2.set_xlabel("Date")
                ax2.set_ylabel("Price")
                ax2.legend()
                st.pyplot(fig2)

                # -------------------------------------------------------------
                # 🕒 Last Updated
                # -------------------------------------------------------------
                st.caption(f"🗓️ Data last updated till: {data.index[-1].strftime('%Y-%m-%d')} (includes today's data if available)")

        except Exception as e:
            st.error(f"❌ Error: {e}")
