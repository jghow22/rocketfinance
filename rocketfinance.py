import os
import openai
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, request
from flask_cors import CORS

# Import necessary libraries for the models
from keras.models import Sequential
from keras.layers import Dense, LSTM, Input
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler

# Initialize Flask App
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# OpenAI API Key (set this securely in your environment variables)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Model Handler Functions (Integrated from modelhandler.py)

def create_lstm_model():
    """Recreate the LSTM model programmatically."""
    model = Sequential([
        Input(shape=(10, 1)),  # Adjust the input shape based on your data
        LSTM(units=50, return_sequences=True),
        LSTM(units=50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    print("LSTM model recreated successfully.")
    return model

def create_arima_model(data):
    """Recreate and fit the ARIMA model."""
    try:
        data = data.asfreq("D").fillna(method="ffill")  # Fill missing values
        model = ARIMA(data['Close'], order=(0, 1, 0))  # Simplified order
        model_fit = model.fit()
        print("ARIMA model recreated and trained successfully.")
        return model_fit
    except Exception as e:
        print(f"Error in ARIMA model creation: {e}")
        raise

def lstm_prediction(model, data):
    """Make predictions using the LSTM model."""
    try:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
        # Adjust the reshape parameters as needed for your model
        prediction = model.predict(scaled_data[-1].reshape(1, 1, 1))
        return scaler.inverse_transform(prediction).flatten().tolist()
    except Exception as e:
        print(f"LSTM prediction error: {e}")
        return []

def arima_prediction(model):
    """Make predictions using the ARIMA model."""
    try:
        return model.forecast(steps=5).tolist()
    except Exception as e:
        print(f"ARIMA prediction error: {e}")
        return []

# Create the LSTM model once and reuse it
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
            messages=[
                {"role": "system", "content": "You are a stock market AI assistant."},
                {"role": "user", "content": prompt}
            ]
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

    # Check if the result is cached
    cached_result = cache.get(symbol)
    if cached_result:
        print("Returning cached result.")
        return jsonify(cached_result)

    try:
        data = fetch_data(symbol)
        arima_model_obj = create_arima_model(data)
        lstm_pred = lstm_prediction(lstm_model, data)
        arima_pred = arima_prediction(arima_model_obj)

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
