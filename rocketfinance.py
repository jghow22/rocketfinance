import os
import requests
import openai
import pandas as pd
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

# Initialize Flask App with static folder (Flask serves static files automatically)
app = Flask(__name__, static_folder="static")
CORS(app, resources={r"/*": {"origins": "*"}})

# Set OpenAI API Key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

# ---------------------------
# Data Fetching using Alpha Vantage
# ---------------------------
def fetch_data(symbol, timeframe):
    """
    Fetch historical daily adjusted stock data for a symbol from Alpha Vantage.
    Uses outputsize "compact" (last 100 days) for 1mo/3mo and "full" for 1yr.
    Filters the data to the requested timeframe.
    """
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise ValueError("Alpha Vantage API key not set in environment variable ALPHAVANTAGE_API_KEY")
        
    outputsize = "compact"  # compact returns ~100 data points
    if timeframe == "1yr":
        outputsize = "full"  # full returns up to 20 years of daily data
    
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": symbol,
        "apikey": api_key,
        "outputsize": outputsize,
        "datatype": "json"
    }
    print(f"Fetching data for {symbol} from Alpha Vantage with outputsize {outputsize}")
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise ValueError(f"Alpha Vantage API request failed with status code {response.status_code}")
    
    data_json = response.json()
    if "Error Message" in data_json:
        raise ValueError(f"Alpha Vantage API error: {data_json['Error Message']}")
    if "Time Series (Daily)" not in data_json:
        raise ValueError("Alpha Vantage API response missing 'Time Series (Daily)'")
    
    ts_data = data_json["Time Series (Daily)"]
    df = pd.DataFrame.from_dict(ts_data, orient="index")
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    # Rename the adjusted close column to "Close"
    df = df.rename(columns={"5. adjusted close": "Close"})
    df["Close"] = df["Close"].astype(float)
    
    # Filter the DataFrame to the requested timeframe
    now = datetime.now()
    if timeframe == "1mo":
        start_date = now - timedelta(days=30)
    elif timeframe == "3mo":
        start_date = now - timedelta(days=90)
    elif timeframe == "1yr":
        start_date = now - timedelta(days=365)
    else:
        start_date = now - timedelta(days=30)
    df = df[df.index >= start_date]
    if df.empty:
        raise ValueError(f"No data found for symbol: {symbol} in the specified timeframe")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    
    print(f"Fetched {len(df)} rows of data for {symbol} from Alpha Vantage")
    return df

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
# Other Helper Functions
# ---------------------------
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
