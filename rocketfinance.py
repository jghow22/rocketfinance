import os
import openai
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, request
from flask_cors import CORS
from modelhandler import create_lstm_model, create_arima_model, lstm_prediction, arima_prediction

# Initialize Flask App
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# OpenAI API Key (Set this securely in your environment variables)
openai.api_key = os.getenv("OPENAI_API_KEY")  

# Load LSTM Model
lstm_model = create_lstm_model()
cache = {}

def fetch_data(symbol):
    """Fetch historical data for a stock symbol."""
    try:
        data = yf.download(symbol, period="1mo", interval="1d")  
        if data.empty:
            raise ValueError(f"No data found for symbol: {symbol}")
        return data
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        raise

def refine_predictions_with_openai(symbol, lstm_pred, arima_pred, history):
    """Enhance stock predictions using OpenAI's API."""
    prompt = f"""
    Given the following stock data for {symbol}, analyze trends and refine the LSTM and ARIMA predictions.
    
    - Historical Closing Prices: {history['Close'].tolist()}
    - LSTM Prediction: {lstm_pred}
    - ARIMA Prediction: {arima_pred}

    Provide a more accurate stock movement prediction along with confidence levels.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[{"role": "system", "content": "You are a stock market AI assistant."},
                      {"role": "user", "content": prompt}]
        )
        refined_prediction = response["choices"][0]["message"]["content"]
        return refined_prediction
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return "Error in refining prediction."

@app.route("/process", methods=["GET"])
def process():
    symbol = request.args.get("symbol", "AAPL")
    print(f"Received request for symbol: {symbol}")

    cached_result = cache.get(symbol)
    if cached_result:
        print("Returning cached result.")
        return jsonify(cached_result)

    try:
        data = fetch_data(symbol)
        arima_model = create_arima_model(data)
        lstm_pred = lstm_prediction(lstm_model, data)
        arima_pred = arima_prediction(arima_model)

        # Use OpenAI to refine predictions
        refined_prediction = refine_predictions_with_openai(symbol, lstm_pred, arima_pred, data)

        response = {
            "lstm_prediction": lstm_pred,
            "arima_prediction": arima_pred,
            "openai_refined_prediction": refined_prediction
        }
        cache[symbol] = response
        return jsonify(response)
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
