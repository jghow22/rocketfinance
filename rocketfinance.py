import os
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Input
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS

# Disable GPU usage to prevent TensorFlow errors
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Apply a dark theme to Matplotlib
plt.style.use('dark_background')

# Flask application setup
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Fetch News Articles
def fetch_news(symbol):
    try:
        API_KEY = os.getenv("NEWS_API_KEY", "your_api_key_here")  # Replace with your News API key
        url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={API_KEY}&pageSize=5"
        response = requests.get(url)
        response.raise_for_status()
        articles = response.json().get('articles', [])
        return articles
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []

# ARIMA Prediction
def arima_prediction(data):
    try:
        data.index.freq = data.index.inferred_freq
        model = ARIMA(data['Close'], order=(5, 1, 0))
        model_fit = model.fit()
        return model_fit.forecast(steps=5).tolist()
    except Exception as e:
        print(f"ARIMA error: {e}")
        return []

# LSTM Prediction
def lstm_prediction(data):
    try:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
        
        model = Sequential([
            Input(shape=(scaled_data.shape[1], 1)),
            LSTM(units=50, return_sequences=True),
            LSTM(units=50),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(scaled_data, scaled_data, epochs=1, batch_size=1, verbose=0)
        
        prediction = model.predict(scaled_data[-1].reshape(1, 1, 1))
        return scaler.inverse_transform(prediction).flatten().tolist()
    except Exception as e:
        print(f"LSTM error: {e}")
        return []

# Generate and Save Chart
def generate_chart(symbol, is_crypto=False):
    try:
        data = yf.download(f"{symbol}-USD" if is_crypto else symbol, period="1y", interval="1d")
        if data.empty:
            print(f"No data found for {symbol}.")
            return None
        
        arima_pred = arima_prediction(data)
        lstm_pred = lstm_prediction(data)

        # Create chart
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data.index, data['Close'], label='Close Price', color='blue')
        if arima_pred:
            ax.plot(pd.date_range(start=data.index[-1], periods=5), arima_pred, label='ARIMA Forecast', color='cyan')
        if lstm_pred:
            ax.scatter(pd.date_range(start=data.index[-1], periods=1), lstm_pred, label='LSTM Prediction', color='red')
        ax.set_title(f"{symbol.upper()} {'Crypto' if is_crypto else 'Stock'} Chart")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()

        # Save chart
        file_name = f"{symbol}_chart.png"
        plt.savefig(file_name)
        plt.close(fig)
        return file_name
    except Exception as e:
        print(f"Error generating chart for {symbol}: {e}")
        return None

# Process Request
def process_request(symbol, is_crypto=False):
    chart_path = generate_chart(symbol, is_crypto)
    news = fetch_news(symbol)
    return {
        "chart_path": chart_path,
        "news": news
    }

# Flask Routes
@app.route("/")
def home():
    return "Rocket Finance Backend is Running!"

@app.route("/process", methods=["GET"])
def process():
    symbol = request.args.get("symbol", "AAPL")
    result = process_request(symbol, is_crypto=False)
    return jsonify(result)

# Run Flask App
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
