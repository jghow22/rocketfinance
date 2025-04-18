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

# Global cache for responses (currently unused)
cache = {}

# Initialize Flask App with static folder
app = Flask(__name__, static_folder="static")
CORS(app, resources={r"/*": {"origins": "*"}})

# Set API keys from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

# ---------------------------
# Helper: Create Dark Style for mplfinance
# ---------------------------
def create_dark_style():
    """Create a custom dark style for mplfinance."""
    import mplfinance as mpf
    mc = mpf.make_marketcolors(
        up='lime', down='red',
        edge='white', wick='white', volume='in'
    )
    style = mpf.make_mpf_style(
        base_mpf_style='nightclouds',
        marketcolors=mc,
        facecolor='black',
        edgecolor='black',
        gridcolor='dimgray'
    )
    return style

# ---------------------------
# Data Fetching from Alpha Vantage
# ---------------------------
def fetch_data(symbol, timeframe):
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise ValueError("Alpha Vantage API key not set in environment variable ALPHAVANTAGE_API_KEY")
    
    intraday_options = ["5min", "30min", "2h", "4h"]
    if timeframe in intraday_options:
        base_interval = "60min" if timeframe in ["2h", "4h"] else timeframe
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
        df = df.rename(columns={"4. close": "Close"})
        df["Close"] = df["Close"].astype(float)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        if timeframe in ["2h", "4h"]:
            freq = "2H" if timeframe == "2h" else "4H"
            df = df["Close"].resample(freq).mean().dropna().to_frame()
            print(f"Resampled intraday data to {freq} frequency, resulting in {len(df)} rows.")
        return df
    else:
        function = "TIME_SERIES_DAILY"
        outputsize = "compact" if timeframe != "1yr" else "full"
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
        df = df.rename(columns={
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close"
        })
        for col in ["Open", "High", "Low", "Close"]:
            df[col] = df[col].astype(float)
        now = datetime.now()
        if timeframe == "1day":
            start_date = now - timedelta(days=7)
        elif timeframe == "7day":
            start_date = now - timedelta(days=15)
        elif timeframe == "1mo":
            start_date = now - timedelta(days=45)
        elif timeframe == "3mo":
            start_date = now - timedelta(days=100)
        elif timeframe == "1yr":
            start_date = now - timedelta(days=400)
        else:
            start_date = now - timedelta(days=30)
        df = df[df.index >= start_date]
        if df.empty:
            raise ValueError(f"No data found for symbol: {symbol} in the specified timeframe")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        return df

# ---------------------------
# Intraday Forecast using Polynomial Regression (Degree 2)
# ---------------------------
def linear_regression_forecast(data, periods=5, degree=2):
    x = np.arange(len(data))
    y = data["Close"].values
    coeffs = np.polyfit(x, y, degree)
    poly = np.poly1d(coeffs)
    x_future = np.arange(len(data), len(data) + periods)
    forecast = poly(x_future)
    forecast[0] = y[-1]
    forecast = forecast.tolist()
    print(f"Polynomial regression forecast (degree={degree}): {forecast}")
    return forecast

# ---------------------------
# ARIMA Model for Daily Data
# ---------------------------
def create_arima_model(data):
    try:
        data = data.asfreq("D").ffill()
        model = ARIMA(data["Close"], order=(0, 1, 1), trend="t")
        model_fit = model.fit()
        print("ARIMA model created and fitted successfully.")
        print(f"Model parameters: {model_fit.params}")
        return model_fit
    except Exception as e:
        print(f"Error in ARIMA model creation: {e}")
        raise

def arima_prediction(model):
    try:
        forecast = model.forecast(steps=5).tolist()
        print(f"ARIMA forecast: {forecast}")
        return forecast
    except Exception as e:
        print(f"ARIMA prediction error: {e}")
        raise

# ---------------------------
# Build Raw Chart Data for Front-End
# ---------------------------
def get_chart_data(data, forecast, timeframe):
    historical_dates = data.index.strftime("%Y-%m-%dT%H:%M:%SZ").tolist()
    historical_values = data["Close"].tolist()
    last_date = data.index[-1]
    if timeframe.endswith("min"):
        minutes = int(timeframe.replace("min", ""))
        delta = timedelta(minutes=minutes)
        forecast_dates = [(last_date + delta * (i + 1)).strftime("%Y-%m-%dT%H:%M:%SZ") for i in range(len(forecast))]
    elif timeframe.endswith("h"):
        hours = int(timeframe.replace("h", ""))
        delta = timedelta(hours=hours)
        forecast_dates = [(last_date + delta * (i + 1)).strftime("%Y-%m-%dT%H:%M:%SZ") for i in range(len(forecast))]
    else:
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=len(forecast),
            freq="B"
        ).strftime("%Y-%m-%dT%H:%M:%SZ").tolist()
    return {
        "historicalDates": historical_dates,
        "historicalValues": historical_values,
        "forecastDates": forecast_dates,
        "forecastValues": forecast
    }

# ---------------------------
# Chart Generation with Candlestick + Continuous Forecast
# ---------------------------
def generate_chart(data, symbol, forecast=None, timeframe="1mo"):
    os.makedirs("static", exist_ok=True)
    filename = f"chart_{symbol.upper()}.png"
    filepath = os.path.join("static", filename)
    
    # If we have OHLC => candlestick
    if {"Open", "High", "Low", "Close"}.issubset(data.columns):
        try:
            import mplfinance as mpf
            data_filled = data.ffill()
            dark_style = create_dark_style()
            # Return figure & axes
            fig, axes = mpf.plot(
                data_filled,
                type='candle',
                style=dark_style,
                title=f"{symbol.upper()} Candlestick Chart",
                ylabel="Price",
                returnfig=True
            )
            ax = axes[0] if isinstance(axes, (list, tuple)) else axes
            fig.patch.set_facecolor("black")
            ax.set_facecolor("black")
            
            if forecast and len(forecast) > 0:
                chart_info = get_chart_data(data_filled, forecast, timeframe)
                f_dates = pd.to_datetime(chart_info["forecastDates"])
                last_close = data_filled["Close"].iloc[-1]
                # Draw flat connector
                ax.plot(
                    [data_filled.index[-1], f_dates[0]],
                    [last_close, last_close],
                    linestyle="--", color="yellow", linewidth=2, label="Connector"
                )
                # Draw forecast line
                ax.plot(
                    [data_filled.index[-1]] + list(f_dates),
                    [last_close] + forecast,
                    linestyle="--", marker="o", color="cyan", linewidth=2, label="Forecast"
                )
                ax.legend()
            
            fig.savefig(filepath, facecolor=fig.get_facecolor(), edgecolor='none')
            plt.close(fig)
        except Exception as e:
            print(f"Error using mplfinance for candlestick chart: {e}")
            # Fallback line
            plt.style.use("dark_background")
            plt.figure(figsize=(10, 5))
            plt.plot(data.index, data["Close"], label="Historical", color="white")
            if forecast and len(forecast) > 0:
                dates = pd.to_datetime(get_chart_data(data, forecast, timeframe)["forecastDates"])
                plt.plot(dates, forecast, "--o", color="cyan", label="Forecast")
            plt.title(f"{symbol.upper()} Prices")
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend()
            plt.grid(color="dimgray")
            plt.savefig(filepath)
            plt.close()
    else:
        # Intraday fallback
        plt.style.use("dark_background")
        plt.figure(figsize=(10, 5))
        plt.plot(data.index, data["Close"], label="Historical", color="white")
        if forecast and len(forecast) > 0:
            dates = pd.to_datetime(get_chart_data(data, forecast, timeframe)["forecastDates"])
            plt.plot(dates, forecast, "--o", color="cyan", label="Forecast")
        plt.title(f"{symbol.upper()} Prices")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(color="dimgray")
        plt.savefig(filepath)
        plt.close()

    print("Chart saved to", filepath)
    return filename

# ---------------------------
# News and OpenAI Analysis Functions
# ---------------------------
def fetch_news(symbol):
    return [
        {
            "title": f"{symbol.upper()} surges amid market optimism",
            "source": {"name": "Reuters"},
            "summary": ("The stock experienced a significant surge today as investors reacted to strong earnings "
                        "and positive market sentiment.")
        },
        {
            "title": f"{symbol.upper()} announces new product line",
            "source": {"name": "Bloomberg"},
            "summary": ("In a recent press release, the company unveiled its latest product innovations, which are expected "
                        "to drive future growth.")
        }
    ]

def refine_predictions_with_openai(symbol, lstm_pred, forecast, history):
    history_tail = history["Close"].tail(30).tolist()
    prompt = f"""
Analyze the following stock data for {symbol.upper()}:

Historical Closing Prices (last 30 days): {history_tail}
Forecast (next 5 periods): {forecast}

Provide a detailed analysis that includes:
- Key observations on historical performance
- An evaluation of the forecast and confidence level
- Specific market recommendations with risk management
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
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return "OpenAI analysis unavailable."

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
    try:
        data = fetch_data(symbol, timeframe)
        intraday = ["5min", "30min", "2h", "4h"]
        if timeframe in intraday:
            forecast = linear_regression_forecast(data, periods=5, degree=2)
        else:
            arima_model_obj = create_arima_model(data)
            forecast = arima_prediction(arima_model_obj)
        refined = refine_predictions_with_openai(symbol, None, forecast, data)
        chart_filename = generate_chart(data, symbol, forecast=forecast, timeframe=timeframe)
        chart_data = get_chart_data(data, forecast, timeframe)
        news = fetch_news(symbol)
        response = {
            "forecast": forecast,
            "openai_refined_prediction": refined,
            "chart_path": chart_filename,
            "chartData": {"symbol": symbol.upper(), **chart_data},
            "news": news
        }
        cache.pop(f"{symbol.upper()}_{timeframe}", None)
        return jsonify(response)
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
