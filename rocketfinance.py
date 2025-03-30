import os
import requests
import openai
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import numpy as np

# For testing purposes, we'll continue using dummy predictions.
TEST_MODE = True  # Set to False when you want to enable real model predictions.

# Suppress TensorFlow logs (if using TensorFlow models later)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Set a modern user agent (for Alpha Vantage, if needed)
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
# Data Fetching using Alpha Vantage
# ---------------------------
def fetch_data(symbol, timeframe):
    """
    Fetch historical daily adjusted stock data for a symbol from Alpha Vantage.
    Uses "compact" outputsize for 1mo/3mo and "full" for 1yr.
    If the expected key is missing in the response, logs the full response and falls back to dummy data.
    """
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise ValueError("Alpha Vantage API key not set in environment variable ALPHAVANTAGE_API_KEY")
        
    outputsize = "compact"
    if timeframe == "1yr":
        outputsize = "full"
    
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
    if "Time Series (Daily)" not in data_json:
        print("Alpha Vantage API response:", data_json)
        print("Expected key 'Time Series (Daily)' not found. Falling back to dummy data.")
        return generate_dummy_data(symbol)
    
    ts_data = data_json["Time Series (Daily)"]
    df = pd.DataFrame.from_dict(ts_data, orient="index")
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    df = df.rename(columns={"5. adjusted close": "Close"})
    df["Close"] = df["Close"].astype(float)
    
    # Filter the DataFrame to the requested timeframe.
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
        print(f"No data found for {symbol} in the specified timeframe. Falling back to dummy data.")
        return generate_dummy_data(symbol)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    
    print(f"Fetched {len(df)} rows of data for {symbol} from Alpha Vantage")
    return df

def generate_dummy_data(symbol):
    """Generate dummy data for testing purposes."""
    now = datetime.now()
    dates = pd.date_range(end=now, periods=22, freq='B')
    dummy_close = np.linspace(150, 160, num=len(dates))
    dummy_data = pd.DataFrame({'Close': dummy_close}, index=dates)
    dummy_data.index = dummy_data.index.tz_localize("UTC")
    print(f"Generated {len(dummy_data)} rows of dummy data for {symbol}")
    return dummy_data

# ---------------------------
# (Disabled) Model Handler Functions
# ---------------------------
if not TEST_MODE:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Input

    def create_lstm_model():
        model = Sequential([
            Input(shape=(10, 1)),
            LSTM(units=50, return_sequences=True),
            LSTM(units=50),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        print("LSTM model recreated successfully.")
        return model

    def create_arima_model(data):
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
        try:
            return model.forecast(steps=5).tolist()
        except Exception as e:
            print(f"ARIMA prediction error: {e}")
            return []
    
    lstm_model = create_lstm_model()
else:
    # In test mode, we return dummy predictions.
    def create_arima_model(data):
        print("Skipping ARIMA model creation (test mode).")
        return None
    def lstm_prediction(model, data):
        print("Skipping LSTM prediction (test mode).")
        return [155]  # Dummy value
    def arima_prediction(model):
        print("Skipping ARIMA prediction (test mode).")
        return [156, 157, 158, 159, 160]  # Dummy forecast
    lstm_model = None

cache = {}

# ---------------------------
# Other Helper Functions
# ---------------------------
def generate_chart(data, symbol, forecast=None):
    """
    Generate a chart of the historical closing prices and, if provided,
    overlay the forecast as a line.
    """
    os.makedirs("static", exist_ok=True)
    filename = f"chart_{symbol.upper()}.png"
    filepath = os.path.join("static", filename)
    
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data['Close'], label="Historical")
    
    # If forecast values are provided, generate forecast dates and plot them.
    if forecast and len(forecast) > 0:
        last_date = data.index[-1]
        # Create forecast dates: assume next consecutive days (or business days if desired)
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=len(forecast), freq='B')
        plt.plot(forecast_dates, forecast, label="Forecast", linestyle="--", marker="o")
    
    plt.title(f"{symbol.upper()} Closing Prices")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.grid(True)
    plt.legend()
    plt.savefig(filepath)
    plt.close()
    return filename

def fetch_news(symbol):
    """
    Return dummy news articles for the stock symbol,
    each with a title, source, and a summary.
    """
    news = [
        {
            "title": f"{symbol.upper()} surges amid market optimism",
            "source": {"name": "Reuters"},
            "summary": "The stock experienced a significant surge today amid positive market sentiment and favorable earnings reports."
        },
        {
            "title": f"{symbol.upper()} announces new product line",
            "source": {"name": "Bloomberg"},
            "summary": "In a recent press release, the company unveiled its latest product line, which is expected to boost revenue in the upcoming quarters."
        }
    ]
    return news

def refine_predictions_with_openai(symbol, lstm_pred, arima_pred, history):
    """
    (Temporary dummy implementation for testing)
    Instead of calling the OpenAI API, return a detailed dummy prediction.
    """
    print(f"Skipping real OpenAI call for {symbol}; returning dummy prediction.")
    suggestion = (
        f"After analyzing the historical trends and recent market data, our models indicate a positive outlook for {symbol.upper()}. "
        f"The LSTM model forecasts a near-term price around {lstm_pred[0] if lstm_pred else 'N/A'}, while the ARIMA model suggests a gradual increase with forecasts of "
        f"{', '.join(str(x) for x in arima_pred)}. We recommend a cautious buy strategy with a target price in the upper forecast range."
    )
    return suggestion

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
        # Pass the ARIMA forecast to the chart function so it is overlaid.
        chart_filename = generate_chart(data, symbol, forecast=arima_pred)
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
