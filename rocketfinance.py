import os
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
from statsmodels.tsa.arima.model import ARIMA
import joblib
import matplotlib.pyplot as plt
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS

# Suppress TensorFlow logs and force CPU usage
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Apply a dark theme to Matplotlib
plt.style.use('dark_background')

# Flask application setup
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Paths to pre-trained models
LSTM_MODEL_PATH = "models/lstm_model.h5"  # Adjust path if models are in a subdirectory
ARIMA_MODEL_PATH = "models/arima_model.pkl"

# Debugging: Print current working directory and list files
print("Current working directory:", os.getcwd())
print("Files in the directory:", os.listdir(os.getcwd()))
if os.path.exists("models"):
    print("Files in 'models' directory:", os.listdir("models"))

# Load pre-trained models
try:
    lstm_model = load_model(LSTM_MODEL_PATH)
    print(f"LSTM model loaded from {LSTM_MODEL_PATH}")
except Exception as e:
    print(f"Error loading LSTM model: {e}")

try:
    arima_model = joblib.load(ARIMA_MODEL_PATH)
    print(f"ARIMA model loaded from {ARIMA_MODEL_PATH}")
except Exception as e:
    print(f"Error loading ARIMA model: {e}")

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

# ARIMA Prediction using the pre-trained model
def arima_prediction():
    try:
        # Forecast using the loaded ARIMA model
        return arima_model.forecast(steps=5).tolist()
    except Exception as e:
        print(f"ARIMA error: {e}")
        return []

# LSTM Prediction using the pre-trained model
def lstm_prediction(data):
    try:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

        # Predict using the pre-trained LSTM model
        prediction = lstm_model.predict(scaled_data[-1].reshape(1, 1, 1))
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
        
        arima_pred = arima_prediction()
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

# Main Entry Point
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
