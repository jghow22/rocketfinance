import os
import requests
import openai
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta
import numpy as np

# Global cache for responses
cache = {}

# Initialize Flask App with static folder
app = Flask(__name__, static_folder="static")
CORS(app, resources={r"/*": {"origins": "*"}})

# Set API keys from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

# ---------------------------
# Data Fetching using Alpha Vantage
# ---------------------------
def fetch_data(symbol, timeframe):
    """
    Fetch stock data from Alpha Vantage.
    
    For intraday timeframes ("5min", "30min", "2h", "4h"), use TIME_SERIES_INTRADAY.
    - "5min" and "30min": fetch using those intervals.
    - "2h" and "4h": fetch using "60min" data then resample.
    
    For daily timeframes ("1day", "7day", "1mo", "3mo", "1yr"), use TIME_SERIES_DAILY and filter by date.
    """
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise ValueError("Alpha Vantage API key not set in environment variable ALPHAVANTAGE_API_KEY")
    
    # Define which timeframes are intraday
    intraday_timeframes = ["5min", "30min", "2h", "4h"]
    
    if timeframe in intraday_timeframes:
        # For "2h" and "4h", we fetch 60min data and then resample.
        if timeframe in ["2h", "4h"]:
            base_interval = "60min"
        else:
            base_interval = timeframe  # "5min" or "30min"
        function = "TIME_SERIES_INTRADAY"
        params = {
            "function": function,
            "symbol": symbol,
            "apikey": api_key,
            "interval": base_interval,
            "outputsize": "compact",
            "datatype": "json"
        }
        expected_key = f"Time Series ({base_interval})"
        print(f"Fetching intraday data for {symbol} with interval {base_interval}")
        response = requests.get("https://www.alphavantage.co/query", params=params)
        if response.status_code != 200:
            raise ValueError(f"Alpha Vantage API request failed with status code {response.status_code}")
        data_json = response.json()
        if expected_key not in data_json:
            print("Alpha Vantage API response:", data_json)
            raise ValueError(f"Alpha Vantage API response missing expected key: {expected_key}")
        ts_data = data_json[expected_key]
        df = pd.DataFrame.from_dict(ts_data, orient="index")
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        # Use "4. close" as the closing price for intraday data too
        df = df.rename(columns={"4. close": "Close"})
        df["Close"] = df["Close"].astype(float)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        # For "2h" and "4h", resample the data
        if timeframe in ["2h", "4h"]:
            freq = "2H" if timeframe == "2h" else "4H"
            df = df["Close"].resample(freq).mean().dropna().to_frame()
            print(f"Resampled data to {freq} frequency, resulting in {len(df)} rows.")
        return df
    else:
        # Daily data
        function = "TIME_SERIES_DAILY"
        outputsize = "compact"
        if timeframe == "1yr":
            outputsize = "full"
        params = {
            "function": function,
            "symbol": symbol,
            "apikey": api_key,
            "outputsize": outputsize,
            "datatype": "json"
        }
        expected_key = "Time Series (Daily)"
        print(f"Fetching daily data for {symbol} with outputsize {outputsize}")
        response = requests.get("https://www.alphavantage.co/query", params=params)
        if response.status_code != 200:
            raise ValueError(f"Alpha Vantage API request failed with status code {response.status_code}")
        data_json = response.json()
        if expected_key not in data_json:
            print("Alpha Vantage API response:", data_json)
            raise ValueError("Alpha Vantage API response missing 'Time Series (Daily)'")
        ts_data = data_json[expected_key]
        df = pd.DataFrame.from_dict(ts_data, orient="index")
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        df = df.rename(columns={"4. close": "Close"})
        df["Close"] = df["Close"].astype(float)
        
        # Define date ranges based on the timeframe
        now = datetime.now()
        if timeframe == "1day":
            start_date = now - timedelta(days=1)
        elif timeframe == "7day":
            start_date = now - timedelta(days=7)
        elif timeframe == "1mo":
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
        return df

# ---------------------------
# Model Handler Functions (ARIMA)
# ---------------------------
def create_arima_model(data):
    """
    Create and fit an ARIMA model on the 'Close' price.
    Using ARIMA(0,1,1) with a constant (trend='c') to capture drift.
    """
    try:
        data = data.asfreq("D").ffill()
        model = ARIMA(data["Close"], order=(0, 1, 1), trend="c")
        model_fit = model.fit()
        print("ARIMA model created and fitted successfully.")
        print(f"Model parameters: {model_fit.params}")
        return model_fit
    except Exception as e:
        print(f"Error in ARIMA model creation: {e}")
        raise

def arima_prediction(model):
    """
    Forecast the next 5 days using the ARIMA model.
    """
    try:
        forecast = model.forecast(steps=5).tolist()
        print(f"ARIMA forecast: {forecast}")
        return forecast
    except Exception as e:
        print(f"ARIMA prediction error: {e}")
        raise

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
    plt.gcf().autofmt_xdate()
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
            "summary": ("The stock experienced a significant surge today as investors reacted to strong earnings "
                        "and positive market sentiment. Analysts note that while volatility remains, there are signs of a potential rebound.")
        },
        {
            "title": f"{symbol.upper()} announces new product line",
            "source": {"name": "Bloomberg"},
            "summary": ("In a recent press release, the company unveiled its latest product innovations, which are expected "
                        "to drive future growth. Experts advise monitoring the market for sustained trends.")
        }
    ]
    return news

def refine_predictions_with_openai(symbol, lstm_pred, arima_pred, history):
    """
    Call the OpenAI API to provide a detailed analysis of the stock.
    The analysis includes historical performance, evaluation of the ARIMA forecast, a confidence level, and market recommendations.
    """
    history_tail = history["Close"].tail(30).tolist()
    prompt = f"""
    Analyze the following stock data for {symbol.upper()}:

    Historical Closing Prices (last 30 days): {history_tail}
    ARIMA Forecast (next 5 days): {arima_pred}

    Provide a detailed analysis that includes:
    - Key observations on historical performance (e.g., highs, lows, volatility, trends).
    - An evaluation of the ARIMA forecast, including any drift or trend changes.
    - A confidence level in the forecast.
    - Specific market recommendations, including risk management strategies.

    Format your response with clear headings and bullet points.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a stock market expert."},
                {"role": "user", "content": prompt}
            ],
            timeout=15
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
        arima_pred = arima_prediction(arima_model_obj)
        refined_prediction = refine_predictions_with_openai(symbol, "N/A", arima_pred, data)
        chart_filename = generate_chart(data, symbol, forecast=arima_pred)
        news = fetch_news(symbol)
        
        response = {
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
