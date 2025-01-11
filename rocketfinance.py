import os
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from flask import Flask, jsonify, request
from flask_cors import CORS
from modelhandler import create_lstm_model, create_arima_model, lstm_prediction, arima_prediction

# Flask application setup
app = Flask(__name__)
CORS(app)

# Recreate models
lstm_model = create_lstm_model()

def fetch_data(symbol):
    """Fetch historical data for a stock symbol."""
    data = yf.download(symbol, period="1y", interval="1d")
    if data.empty:
        raise ValueError(f"No data found for symbol: {symbol}")
    return data

@app.route("/")
def home():
    return "Rocket Finance Backend is Running!"

@app.route("/process", methods=["GET"])
def process():
    symbol = request.args.get("symbol", "AAPL")
    try:
        data = fetch_data(symbol)
        
        # Create ARIMA model on the fly
        arima_model = create_arima_model(data)
        
        # Generate predictions
        lstm_pred = lstm_prediction(lstm_model, data)
        arima_pred = arima_prediction(arima_model)

        return jsonify({
            "lstm_prediction": lstm_pred,
            "arima_prediction": arima_pred
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
