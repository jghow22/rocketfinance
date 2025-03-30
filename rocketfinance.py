import os
import openai
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, request
from flask_cors import CORS
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import numpy as np

# Suppress TensorFlow INFO and WARNING logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Set a modern user agent for yfinance requests
os.environ["YAHOO_USER_AGENT"] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/108.0.0.0 Safari/537.36"
)

# Initialize Flask App with static folder (Flask serves static files automatically)
app = Flask(__name__, static_folder="static")
CORS(app, resources={r"/*": {"origins": "*"}})

# Set OpenAI API Key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

# ---------------------------
# Model Handler Functions
# ---------------------------
def create_lstm_model():
    """Recreate the LSTM model programmatically."""
    model = Sequential([
        Input(shape=(10, 1)),  # Expecting a sequence of 10 time steps
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
        data = data.asfreq("D").ffill()
        model = ARIMA(data['Close'], order=(0, 1, 0))
        model_fit = model.fit()
        print("ARIMA model recreated and trained successfully.")
        return model_fit
    except Exception as e:
        print(f"Error in ARIMA model creation: {e}")
        return None

def lstm_prediction(model, data):
    """Make predictions using the LSTM model."""
    try:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
        if len(scaled_data) < 10:
            print("Not enough data for LSTM prediction")
            return []
        # Use the last 10 data points as input for the LSTM model
        input_sequence = scaled_data[-10:]
        input_sequence = input_sequence.reshape(1, 10, 1)
        prediction = model.predict(input_sequence)
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

lstm_model = create_lstm_model()
cache = {}

# ---------------------------
# Helper Functions
# ---------------------------
def fetch_data(symbol, timeframe):
    """
    Fetch historical data for a stock symbol.
    First try using yf.download with a period parameter.
    If that returns empty, try explicit start/end dates.
    If still empty, generate dummy data.
    """
    period_mapping = {"1mo": "1mo", "3mo": "3mo", "1yr": "1y"}
    period = period_mapping.get(timeframe, "1mo")
    now = datetime.now()
    data = None

    # Try period method
    try:
        print(f"Attempting to fetch data for {symbol} with period={period}")
        data = yf.download(symbol, period=period, interval="1d", progress=False)
        if not data.empty:
            if data.index.tz is None:
                data.index = data.index.tz_localize("UTC")
            print(f"Fetched {len(data)} rows using period method.")
            return data
        else:
            print("Period method returned empty data.")
    except Exception as e:
        print(f"Error using period method for {symbol}: {e}")

    # Try explicit start/end dates
    try:
        if timeframe == "1mo":
            start = now - timedelta(days=30)
        elif timeframe == "3mo":
            start = now - timedelta(days=90)
        elif timeframe == "1yr":
            start = now - timedelta(days=365)
        else:
            start = now - timedelta(days=30)
        print(f"Fetching data for {symbol} from {start.strftime('%Y-%m-%d')} to {now.strftime('%Y-%m-%d')}")
        data = yf.download(symbol,
                           start=start.strftime("%Y-%m-%d"),
                           end=now.strftime("%Y-%m-%d"),
                           interval="1d",
                           progress=False)
        if not data.empty:
            if data.index.tz is None:
                data.index = data.index.tz_localize("UTC")
            print(f"Fetched {len(data)} rows using explicit dates.")
            return data
        else:
            print("Explicit dates method returned empty data.")
    except Exception as e:
        print(f"Error using explicit dates for {symbol}: {e}")

    # Fallback: generate dummy data
    print(f"Using dummy data fallback for {symbol}")
    dates = pd.date_range(end=now, periods=22, freq='B')
    dummy_close = np.linspace(150, 160, num=len(dates))
    dummy_data = pd.DataFrame({'Close': dummy_close}, index=dates)
    dummy_data.index = dummy_data.index.tz_localize("UTC")
    print(f"Generated {len(dummy_data)} rows of dummy data for {symbol}")
    return dummy_data

def generate_chart(data, symbol):
    """Generate a chart of the closing prices and save it in the static folder."""
    os.makedirs("static", exist_ok=True)
    filename = f"chart_{symbol.upper()}.png"
    filepath = os.path.join("static", filename)
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data['Close'])
    plt.title(f"{symbol.upper()} Closing Prices")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.grid(True)
    plt.savefig(filepath)
    plt.close()
    return filename

def fetch_news(symbol):
    """Return dummy news articles for the stock symbol."""
    news = [
        {"title": f"{symbol.upper()} surges amid market optimism", "source": {"name": "Reuters"}},
        {"title": f"{symbol.upper()} announces new product line", "source": {"name": "Bloomberg"}}
    ]
    return news

def refine_predictions_with_openai(symbol, lstm_pred, arima_pred, history):
    """
    (Temporary dummy implementation for testing)
    Instead of calling the OpenAI API, return a dummy prediction.
    """
    print(f"Skipping real OpenAI call for {symbol}; returning dummy prediction.")
    return "Stock looks good. Confidence: 80%."

# ---------------------------
# Flask Routes
# ---------------------------
@app.route("/")
def index():
    return "Red Tape Trading API is running."

@app.route("/process", methods=["GET"])
def process():
    symbol = request.args.get("symbol", "AAPL")
    timeframe = request.args.get("timeframe", "1mo")
    print(f"Received request for symbol: {symbol} with timeframe: {timeframe}")

    cache_key = f"{symbol.upper()}_{timeframe}"
    if cache_key in cache:
        print("Returning cached result.")
        return jsonify(cache[cache_key])
    try:
        data = fetch_data(symbol, timeframe)
        arima_model_obj = create_arima_model(data)
        lstm_pred = lstm_prediction(lstm_model, data)
        arima_pred = arima_prediction(arima_model_obj) if arima_model_obj is not None else []
        refined_prediction = refine_predictions_with_openai(symbol, lstm_pred, arima_pred, data)
        chart_filename = generate_chart(data, symbol)
        news = fetch_news(symbol)
        response = {
            "lstm_prediction": lstm_pred,
            "arima_prediction": arima_pred,
            "openai_refined_prediction": refined_prediction,
            "chart_path": chart_filename,
            "news": news
        }
        cache[cache_key] = response
        return jsonify(response)
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
