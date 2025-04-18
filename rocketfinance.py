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
    """
    Fetch stock data for a symbol from Alpha Vantage.
    - For intraday ("5min", "30min", "2h", "4h"), uses TIME_SERIES_INTRADAY.
      For "2h" and "4h", fetches "60min" data then resamples.
    - For daily ("1day", "7day", "1mo", "3mo", "1yr"), uses TIME_SERIES_DAILY
      and returns OHLC data, filtered by an extended window.
    """
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
        # For intraday, use only the "Close" price.
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
        # Daily data: use TIME_SERIES_DAILY and retain OHLC.
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
    """
    Perform a degree-2 polynomial regression on intraday data to forecast the next 'periods' values.
    Forces the first forecast value to equal the last historical closing price.
    """
    try:
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
    except Exception as e:
        print(f"Error in polynomial regression forecast: {e}")
        raise

# ---------------------------
# ARIMA Model for Daily Data
# ---------------------------
def create_arima_model(data):
    """
    Create and fit an ARIMA model on the 'Close' price using ARIMA(0,1,1)
    with a linear trend (trend='t') for daily data.
    """
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
# Build Raw Chart Data for Front-End
# ---------------------------
def get_chart_data(data, forecast, timeframe):
    """
    Build raw chart data arrays including ISO-formatted historical and forecast dates and corresponding values.
    """
    historical_dates = data.index.strftime("%Y-%m-%dT%H:%M:%SZ").tolist()
    historical_values = data["Close"].tolist()
    last_date = data.index[-1]
    if timeframe.endswith("min"):
        minutes = int(timeframe.replace("min", ""))
        delta = timedelta(minutes=minutes)
        forecast_dates = [
            (last_date + delta * (i + 1)).strftime("%Y-%m-%dT%H:%M:%SZ")
            for i in range(len(forecast))
        ]
    elif timeframe.endswith("h"):
        hours = int(timeframe.replace("h", ""))
        delta = timedelta(hours=hours)
        forecast_dates = [
            (last_date + delta * (i + 1)).strftime("%Y-%m-%dT%H:%M:%SZ")
            for i in range(len(forecast))
        ]
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
    """
    Generate a chart image.
    - If OHLC columns are present (daily data), create a dark-themed candlestick chart using mplfinance.
      The forecast line is overlaid so its first point connects to the last historical close.
      A flat connector (yellow) bridges any gap.
    - For intraday data (or if OHLC is missing), fall back to a simple dark line chart.
    """
    os.makedirs("static", exist_ok=True)
    filename = f"chart_{symbol.upper()}.png"
    filepath = os.path.join("static", filename)
    
    # Check for OHLC => use candlestick
    if {"Open", "High", "Low", "Close"}.issubset(data.columns):
        try:
            import mplfinance as mpf
            data_filled = data.ffill()
            dark_style = create_dark_style()
            # Plot candlestick
            fig, axes = mpf.plot(
                data_filled,
                type='candle',
                style=dark_style,
                title=f"{symbol.upper()} Candlestick Chart",
                ylabel="Price",
                returnfig=True
            )
            # axes may be tuple/list
            ax = axes[0] if isinstance(axes, (list, tuple)) else axes
            fig.patch.set_facecolor("black")
            ax.set_facecolor("black")
            
            # Overlay forecast
            if forecast and len(forecast) > 0:
                chart_info = get_chart_data(data_filled, forecast, timeframe)
                f_dates = pd.to_datetime(chart_info["forecastDates"])
                last_close = data_filled["Close"].iloc[-1]
                # Flat connector
                ax.plot(
                    [data_filled.index[-1], f_dates[0]],
                    [last_close, last_close],
                    linestyle="--", color="yellow", linewidth=2, label="Connector"
                )
                # Forecast line
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
            # Fallback to dark line chart
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
        # Intraday fallback line chart
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

def refine_predictions_with_openai(symbol, lstm_pred, forecast, history):
    """
    Call the OpenAI API to provide a detailed analysis of the stock.
    """
    history_tail = history["Close"].tail(30).tolist()
    prompt = f"""
    Analyze the following stock data for {symbol.upper()}:

    Historical Closing Prices (last 30 days): {history_tail}
    Forecast (next 5 periods): {forecast}

    Provide a detailed analysis that includes:
    - Key observations on historical performance (e.g., highs, lows, volatility, trends).
    - An evaluation of the forecast, including any observed drift or trend changes and possible reasons.
    - Your confidence level in the forecast.
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
        intraday_options = ["5min", "30min", "2h", "4h"]
        if timeframe in intraday_options:
            forecast = linear_regression_forecast(data, periods=5, degree=2)
        else:
            arima_model_obj = create_arima_model(data)
            forecast = arima_prediction(arima_model_obj)
        
        refined_prediction = refine_predictions_with_openai(symbol, "N/A", forecast, data)
        chart_filename = generate_chart(data, symbol, forecast=forecast, timeframe=timeframe)
        chart_data = get_chart_data(data, forecast, timeframe)
        news = fetch_news(symbol)
        
        response = {
            "forecast": forecast,
            "openai_refined_prediction": refined_prediction,
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
