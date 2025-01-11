import os
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from flask import Flask, jsonify, request
from flask_cors import CORS
from modelhandler import create_lstm_model, create_arima_model, lstm_prediction, arima_prediction

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

lstm_model = create_lstm_model()

def fetch_data(symbol):
    """Fetch historical data for a stock symbol."""
    try:
        data = yf.download(symbol, period="3mo", interval="1d")
        if data.empty:
            raise ValueError(f"No data found for symbol: {symbol}")
        return data
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        raise

@app.route("/")
def home():
    return "Rocket Finance Backend is Running!"

@app.route("/process", methods=["GET"])
def process():
    symbol = request.args.get("symbol", "AAPL")
    print(f"Received request for symbol: {symbol}")
    try:
        data = fetch_data(symbol)
        arima_model = create_arima_model(data)
        lstm_pred = lstm_prediction(lstm_model, data)
        arima_pred = arima_prediction(arima_model)
        response = {"lstm_prediction": lstm_pred, "arima_prediction": arima_pred}
        print("Response:", response)
        return jsonify(response)
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
