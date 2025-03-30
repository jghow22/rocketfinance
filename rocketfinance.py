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

# Global cache for responses
cache = {}

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Set a modern user agent (for consistency)
os.environ["YAHOO_USER_AGENT"] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/108.0.0.0 Safari/537.36"
)

# Initialize Flask App with static folder
app = Flask(__name__, static_folder="static")
CORS(app, resources={r"/*": {"origins": "*"}})

# Set OpenAI API Key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

# ---------------------------
# Data Fetching using Alpha Vantage
# ---------------------------
def fetch_data(symbol, timeframe):
    """
    Fetch historical daily stock data for a symbol from Alpha Vantage.
    Uses the free TIME_SERIES_DAILY endpoint.
    Raises an error if the expected data is missing.
    """
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise ValueError("Alpha Vantage API key not set in environment variable ALPHAVANTAGE_API_KEY")
        
    outputsize = "compact"  # ~100 data points
    if timeframe == "1yr":
        outputsize = "full"
    
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
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
        raise ValueError("Alpha Vantage API response missing 'Time Series (Daily)'")
    
    ts_data = data_json["Time Series (Daily)"]
    df = pd.DataFrame.from_dict(ts_data, orient="index")
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    # Use the closing price from "4. close"
    df = df.rename(columns={"4. close": "Close"})
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
    """Create and compile the LSTM model."""
    model = Sequential([
        Input(shape=(10, 1)),
        LSTM(units=50, return_sequences=True),
        LSTM(units=50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    print("LSTM model created successfully.")
    return model

def create_arima_model(data):
    """Create and fit the ARIMA model."""
    try:
        data = data.asfreq("D").ffill()
        model = ARIMA(data["Close"], order=(0, 1, 0))
        model_fit = model.fit()
        print("ARIMA model created and fitted successfully.")
        return model_fit
    except Exception as e:
        print(f"Error in ARIMA model creation: {e}")
        raise

def lstm_prediction(model, data):
    """Make predictions using the LSTM model."""
    try:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data["Close"].values.reshape(-1, 1))
        if len(scaled_data) < 10:
            raise ValueError("Not enough data for LSTM prediction")
        input_sequence = scaled_data[-10:]
        input_sequence = input_sequence.reshape(1, 10, 1)
        prediction = model.predict(input_sequence)
        return scaler.inverse_transform(prediction).flatten().tolist()
    except Exception as e:
        print(f"LSTM prediction error: {e}")
        raise

def arima_prediction(model):
    """Make predictions using the ARIMA model."""
    try:
        forecast = model.forecast(steps=5).tolist()
        return forecast
    except Exception as e:
        print(f"ARIMA prediction error: {e}")
        raise

lstm_model = create_lstm_model()

# ---------------------------
# Other Helper Functions
# ---------------------------
def generate_chart(data, symbol, forecast=None):
    """
    Generate a chart of historical closing prices.
    If forecast is provided, overlay it as a red dashed line with markers.
    """
    os.makedirs("static", exist_ok=True)
    filename = f"chart_{symbol.upper()}.png"
    filepath = os.path.join("static", filename)
    
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data["Close"], label="Historical", color="blue")
    
    if forecast and len(forecast) > 0:
        last_date = data.index[-1]
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=len(forecast), freq="B")
        plt.plot(forecast_dates, forecast, label="Forecast", linestyle="--", marker="o", color="red", linewidth=2)
    
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
    Return news articles for the symbol, each with a title, source, and summary.
    """
    news = [
        {
            "title": f"{symbol.upper()} surges amid market optimism",
            "source": {"name": "Reuters"},
            "summary": "The stock experienced a significant surge today as investors reacted to strong earnings and optimistic market trends."
        },
        {
            "title": f"{symbol.upper()} announces new product line",
            "source": {"name": "Bloomberg"},
            "summary": "In a recent press release, the company unveiled its latest product innovations, expected to drive growth in the upcoming quarters."
        }
    ]
    return news

def refine_predictions_with_openai(symbol, lstm_pred, arima_pred, history):
    """
    Call the OpenAI API to provide a detailed analysis of the stock.
    Returns an analysis including future trends and a confidence level.
    """
    history_tail = history["Close"].tail(30).tolist()
    prompt = f"""
    Analyze the following stock data for {symbol.upper()}:
    
    - Historical Closing Prices (last 30 days): {history_tail}
    - LSTM Prediction: {lstm_pred}
    - ARIMA Forecast: {arima_pred}
    
    Provide a detailed analysis of the stock's potential future trends and a confidence level in your forecast.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a stock market expert."},
                {"role": "user", "content": prompt}
            ],
            timeout=10
        )
        refined_prediction = response["choices"][0]["message"]["content"]
        return refined_prediction
    except Exception as e:
        print(f"OpenAI API error: {e}")
        raise

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
        arima_pred = arima_prediction(arima_model_obj)
        refined_prediction = refine_predictions_with_openai(symbol, lstm_pred, arima_pred, data)
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
