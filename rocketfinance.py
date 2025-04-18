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
    Modified to ensure OHLC data for all timeframes including intraday.
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
        
        # Rename columns to maintain consistent OHLC structure
        df = df.rename(columns={
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
        })
        for col in ["Open", "High", "Low", "Close"]:
            df[col] = df[col].astype(float)
            
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
            
        if timeframe in ["2h", "4h"]:
            freq = "2H" if timeframe == "2h" else "4H"
            # Resample with proper OHLC aggregation
            df = df.resample(freq).agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last'
            }).dropna()
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
# Generate OHLC data for forecast points
# ---------------------------
def generate_forecast_ohlc(data, forecast):
    """
    Generate OHLC values for forecast points using a combination of the 
    forecasted close values and average volatility patterns from historical data.
    """
    # Get average daily volatility metrics from historical data
    avg_range = (data["High"] - data["Low"]).mean()
    avg_open_close_diff = abs(data["Open"] - data["Close"]).mean()
    
    # Calculate the direction of each forecast day (up or down)
    forecast_ohlc = []
    last_close = data["Close"].iloc[-1]
    
    for i, close in enumerate(forecast):
        if i == 0:
            prev_close = last_close
        else:
            prev_close = forecast[i-1]
            
        direction = 1 if close > prev_close else -1
        
        # Calculate open, high, low based on forecasted close and historical volatility
        open_price = prev_close  # Start from previous close
        
        # Determine high and low based on average range and direction
        if direction > 0:
            # Upward day
            high = close + (avg_range * 0.3)  # High is above close
            low = open_price - (avg_range * 0.2)  # Low is below open
        else:
            # Downward day
            high = open_price + (avg_range * 0.2)  # High is above open
            low = close - (avg_range * 0.3)  # Low is below close
        
        # Ensure high is always highest, low is always lowest
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        forecast_ohlc.append({
            "open": float(open_price),
            "high": float(high),
            "low": float(low),
            "close": float(close)
        })
    
    return forecast_ohlc

# ---------------------------
# Build Raw Chart Data for Front-End
# ---------------------------
def get_chart_data(data, forecast, timeframe):
    """
    Build raw chart data arrays including ISO-formatted historical and forecast dates and values.
    Now includes OHLC data for both historical and forecast points.
    """
    historical_dates = data.index.strftime("%Y-%m-%dT%H:%M:%SZ").tolist()
    historical_values = data["Close"].tolist()
    
    # Add OHLC data if available
    historical_ohlc = None
    if {"Open", "High", "Low", "Close"}.issubset(data.columns):
        historical_ohlc = [
            {
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"])
            }
            for _, row in data.iterrows()
        ]
    
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
    
    # Generate forecast OHLC data
    forecast_ohlc = generate_forecast_ohlc(data, forecast)
    
    result = {
        "historicalDates": historical_dates,
        "historicalValues": historical_values,
        "forecastDates": forecast_dates,
        "forecastValues": forecast,
        "timeframe": timeframe
    }
    
    # Include OHLC data if available
    if historical_ohlc:
        result["ohlc"] = historical_ohlc
        result["forecastOhlc"] = forecast_ohlc
    
    return result

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

def refine_predictions_with_openai(symbol, lstm_pred, forecast, history, timeframe):
    """
    Call the OpenAI API to provide a detailed analysis of the stock that's timeframe-specific.
    """
    history_tail = history["Close"].tail(min(30, len(history))).tolist()
    
    # Create a timeframe-specific prompt
    if timeframe in ["5min", "30min", "2h", "4h"]:
        time_context = f"You are analyzing intraday {timeframe} data. Focus on short-term trading strategies and intraday patterns."
        analysis_timeframe = "intraday"
    elif timeframe == "1day":
        time_context = "You are analyzing daily data with a focus on very short-term price action (1-5 days)."
        analysis_timeframe = "very short-term (1-5 days)"
    elif timeframe == "7day":
        time_context = "You are analyzing a week of daily data with a focus on short-term price action (1-2 weeks)."
        analysis_timeframe = "short-term (1-2 weeks)"
    elif timeframe == "1mo":
        time_context = "You are analyzing a month of daily data with a focus on intermediate-term price action (2-4 weeks)."
        analysis_timeframe = "intermediate-term (2-4 weeks)"
    elif timeframe == "3mo":
        time_context = "You are analyzing three months of daily data with a focus on medium-term price action (1-3 months)."
        analysis_timeframe = "medium-term (1-3 months)"
    else:  # 1yr
        time_context = "You are analyzing a year of daily data with a focus on longer-term price action (3-12 months)."
        analysis_timeframe = "longer-term (3-12 months)"
    
    prompt = f"""
    {time_context}
    
    Analyze the following stock data for {symbol.upper()} with a {timeframe} timeframe:
    
    Historical Closing Prices (last {len(history_tail)} periods): {history_tail}
    Forecast (next 5 periods): {forecast}
    
    Provide a detailed {analysis_timeframe} analysis that includes:
    - Key observations on historical performance within this {timeframe} timeframe (highs, lows, volatility, trends)
    - Technical indicators relevant to this {timeframe} timeframe
    - An evaluation of the {timeframe} forecast, including any observed patterns
    - Your confidence level in the {timeframe} forecast 
    - Specific trading/investment recommendations appropriate for this {timeframe} timeframe
    - Risk management strategies for trades at this {timeframe} timeframe
    
    Format your response with clear headings and bullet points.
    """
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": f"You are a stock market expert specializing in {analysis_timeframe} analysis."},
                {"role": "user", "content": prompt}
            ],
            timeout=15
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return f"OpenAI analysis unavailable for {timeframe} timeframe."

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

        # Pass the timeframe to the OpenAI analysis
        refined_prediction = refine_predictions_with_openai(symbol, "N/A", forecast, data, timeframe)
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
