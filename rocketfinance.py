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
# Model Handler Functions (ARIMA with drift)
# ---------------------------
def create_arima_model(data):
    """
    Create and fit an ARIMA model on the 'Close' price.
    Using ARIMA(0, 1, 1) with a linear trend (trend='t') to allow a drift in the integrated model.
    """
    try:
        data = data.asfreq("D").ffill()
        model = ARIMA(data["Close"], order=(0, 1, 1), trend='t')
        model_fit = model.fit()
        print("ARIMA model created and fitted successfully.")
        print("Model parameters:", model_fit.params)
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
            "summary": ("In a recent press release, the company unveiled its latest product innovations. Experts believe these developments "
                        "may drive growth in upcoming quarters, although market conditions require careful monitoring.")
        }
    ]
    return news

def refine_predictions_with_openai(symbol, lstm_pred, arima_pred, history):
    """
    Call the OpenAI API to provide a detailed analysis of the stock.
    The analysis includes historical performance, evaluation of the ARIMA forecast, confidence level, and market recommendations.
    """
    history_tail = history["Close"].tail(30).tolist()
    prompt = f"""
    Analyze the following stock data for {symbol.upper()}:

    Historical Closing Prices (last 30 days): {history_tail}
    ARIMA Forecast (next 5 days): {arima_pred}

    Provide a detailed analysis that includes:
    - Key observations on historical performance (e.g., highs, lows, volatility, trends).
    - An evaluation of the ARIMA forecast. Explain any observed drift or change in trend.
    - A confidence level in the forecast.
    - Specific market recommendations, including risk management strategies.

    Format your analysis with clear headings and bullet points.
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
