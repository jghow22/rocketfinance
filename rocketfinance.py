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
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random
import re
import threading
import io
import base64
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from enum import Enum
import time  # For retry delays

# Try to import TensorFlow but don't fail if it's not available
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. LSTM forecasting will be disabled.")

# Global cache for responses
cache = {}

# Initialize Flask App with static folder
app = Flask(__name__, static_folder="static")
CORS(app, resources={r"/*": {"origins": "*"}})

# Set API keys from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

# Ensure NLTK resources are downloaded (for sentiment analysis)
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

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
def fetch_data(symbol, timeframe, include_extended_hours=True):
    """
    Fetch stock data for a symbol from Alpha Vantage.
    Extended hours data is included by default for intraday timeframes.
    
    Args:
        symbol (str): The stock ticker symbol
        timeframe (str): Time period to fetch data for (e.g., "5min", "1day")
        include_extended_hours (bool, optional): Whether to include extended hours, defaults to True
        
    Returns:
        pd.DataFrame: DataFrame with stock price data
    """
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise ValueError("Alpha Vantage API key not set in environment variable ALPHAVANTAGE_API_KEY")
    
    # Check cache first - now includes extended hours in the key
    cache_key = f"{symbol.upper()}:{timeframe}:{include_extended_hours}"
    if cache_key in cache:
        timestamp, data = cache[cache_key]
        age = (datetime.now() - timestamp).total_seconds()
        if age < 300:  # 5 minutes cache
            print(f"Using cached data for {symbol} {timeframe} (age: {age:.1f}s)")
            return data
    
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
            "datatype": "json",
            "extended_hours": "true" if include_extended_hours else "false"
        }
        
        expected_key = f"Time Series ({base_interval})"
        print(f"Fetching intraday data for {symbol} with interval {base_interval}, extended hours: {include_extended_hours}")
        
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
        rename_dict = {
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close"
        }
        
        # Add volume if it exists
        if "5. volume" in df.columns:
            rename_dict["5. volume"] = "Volume"
        
        df = df.rename(columns=rename_dict)
        
        for col in ["Open", "High", "Low", "Close"]:
            if col in df.columns:
                df[col] = df[col].astype(float)
        
        if "Volume" in df.columns:
            df["Volume"] = df["Volume"].astype(float)
        
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        
        if timeframe in ["2h", "4h"]:
            freq = "2H" if timeframe == "2h" else "4H"
            # Resample with proper OHLC aggregation
            agg_dict = {
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last'
            }
            
            if "Volume" in df.columns:
                agg_dict["Volume"] = 'sum'
            
            df = df.resample(freq).agg(agg_dict).dropna()
            print(f"Resampled intraday data to {freq} frequency, resulting in {len(df)} rows.")
        
        # Mark extended hours data if enabled
        if include_extended_hours:
            df = mark_extended_hours(df)
        
        # Store in cache
        cache[cache_key] = (datetime.now(), df)
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
        
        # Rename columns to maintain consistent OHLC structure
        rename_dict = {
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close"
        }
        
        # Add volume if it exists
        if "5. volume" in df.columns:
            rename_dict["5. volume"] = "Volume"
        
        df = df.rename(columns=rename_dict)
        
        for col in ["Open", "High", "Low", "Close"]:
            if col in df.columns:
                df[col] = df[col].astype(float)
        
        if "Volume" in df.columns:
            df["Volume"] = df["Volume"].astype(float)
        
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
        
        # Daily data doesn't include extended hours, so no session marking
        
        # Store in cache
        cache[cache_key] = (datetime.now(), df)
        return df

def mark_extended_hours(data):
    """
    Mark data points as regular hours, pre-market, or after-hours.
    
    Args:
        data (pd.DataFrame): DataFrame with datetime index
    
    Returns:
        pd.DataFrame: Same DataFrame with additional 'session' column
    """
    df = data.copy()
    df['session'] = 'regular'
    
    # Convert index to datetime if it's not already
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Ensure timezone info is present
    if df.index.tz is None:
        df.index = df.index.tz_localize('America/New_York')
    elif str(df.index.tz) != 'America/New_York':
        df.index = df.index.tz_convert('America/New_York')
    
    # Extract time info
    times = df.index.time
    
    # Define market hours
    pre_market_start = pd.to_datetime('04:00:00').time()
    market_open = pd.to_datetime('09:30:00').time()
    market_close = pd.to_datetime('16:00:00').time()
    after_hours_end = pd.to_datetime('20:00:00').time()
    
    # Mark pre-market (4:00 AM to 9:30 AM ET)
    pre_market_mask = [(t >= pre_market_start and t < market_open) for t in times]
    df.loc[pre_market_mask, 'session'] = 'pre-market'
    
    # Mark after-hours (4:00 PM to 8:00 PM ET)
    after_hours_mask = [(t >= market_close and t <= after_hours_end) for t in times]
    df.loc[after_hours_mask, 'session'] = 'after-hours'
    
    return df

# ---------------------------
# Technical Indicators Calculation
# ---------------------------
def calculate_technical_indicators(data):
    """Calculate key technical indicators for forecasting enhancement."""
    df = data.copy()
    
    # Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=min(20, len(df))).mean()
    df['SMA_50'] = df['Close'].rolling(window=min(50, len(df))).mean()
    df['EMA_12'] = df['Close'].ewm(span=min(12, len(df)), adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=min(26, len(df)), adjust=False).mean()
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=min(9, len(df)), adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=min(14, len(df))).mean()
    avg_loss = loss.rolling(window=min(14, len(df))).mean()
    
    # Handle division by zero
    avg_loss = avg_loss.replace(0, np.nan)
    rs = avg_gain / avg_loss
    rs = rs.replace(np.nan, 0)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    window = min(20, len(df))
    df['BB_Middle'] = df['Close'].rolling(window=window).mean()
    std_dev = df['Close'].rolling(window=window).std()
    df['BB_Upper'] = df['BB_Middle'] + (std_dev * 2)
    df['BB_Lower'] = df['BB_Middle'] - (std_dev * 2)
    
    # Volatility Indicators
    df['ATR'] = calculate_atr(df, min(14, len(df)))  # Average True Range
    
    # Volume Indicators (if volume data is available)
    if 'Volume' in df.columns:
        df['OBV'] = calculate_obv(df)  # On-Balance Volume
    
    return df

def calculate_atr(data, period=14):
    """Calculate Average True Range."""
    df = data.copy()
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    
    # Replace NaN with 0
    df['H-PC'] = df['H-PC'].fillna(0)
    df['L-PC'] = df['L-PC'].fillna(0)
    
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=min(period, len(df))).mean()
    
    return df['ATR']

def calculate_obv(data):
    """Calculate On-Balance Volume."""
    if 'Volume' not in data.columns:
        return pd.Series(0, index=data.index)
    
    obv = [0]
    for i in range(1, len(data)):
        if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
            obv.append(obv[-1] + data['Volume'].iloc[i])
        elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
            obv.append(obv[-1] - data['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    
    return pd.Series(obv, index=data.index)

# ---------------------------
# Market Regime Detection
# ---------------------------
def detect_market_regime(data, window=20):
    """
    Detect the current market regime (trending, mean-reverting, volatile).
    Returns: 'trending_up', 'trending_down', 'mean_reverting', 'volatile', or 'unknown'
    """
    # Adjust window if data is too short
    window = min(window, len(data) // 2)
    if window < 5:
        return 'unknown'  # Not enough data
    
    df = data.tail(window*2).copy()
    
    # Calculate key metrics
    df['returns'] = df['Close'].pct_change()
    
    # Calculate autocorrelation and trend
    if len(df) > 2:
        autocorr = df['returns'].dropna().autocorr(lag=1)
        trend = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]
        
        # Calculate volatility
        if len(df['returns'].dropna()) > 0:
            volatility = df['returns'].std() * np.sqrt(252)  # Annualized volatility
        else:
            volatility = 0
        
        # Calculate ADX (Average Directional Index) to measure trend strength
        df = calculate_adx(df)
        adx_value = df['ADX'].iloc[-1] if 'ADX' in df.columns and not pd.isna(df['ADX'].iloc[-1]) else 15
        
        # Determine regime
        if adx_value > 25:  # Strong trend
            if trend > 0:
                return 'trending_up'
            else:
                return 'trending_down'
        elif autocorr < -0.2:  # Negative autocorrelation suggests mean reversion
            return 'mean_reverting'
        elif volatility > 0.3:  # High volatility
            return 'volatile'
        else:
            return 'unknown'
    else:
        return 'unknown'

def calculate_adx(df, period=14):
    """Calculate Average Directional Index."""
    # Adjust period if data is too short
    period = min(period, len(df) // 3)
    if period < 2:
        df['ADX'] = pd.Series(np.nan, index=df.index)
        return df
    
    # Calculate +DM, -DM, +DI, -DI, and ADX
    try:
        df['tr1'] = abs(df['High'] - df['Low'])
        df['tr2'] = abs(df['High'] - df['Close'].shift(1))
        df['tr3'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        df['+DM'] = np.where((df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']),
                            np.maximum(df['High'] - df['High'].shift(1), 0), 0)
        df['-DM'] = np.where((df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)),
                            np.maximum(df['Low'].shift(1) - df['Low'], 0), 0)
        
        df['TR14'] = df['TR'].rolling(window=period).sum()
        df['+DM14'] = df['+DM'].rolling(window=period).sum()
        df['-DM14'] = df['-DM'].rolling(window=period).sum()
        
        # Handle division by zero
        df['TR14'] = df['TR14'].replace(0, np.nan)
        df['+DI14'] = 100 * df['+DM14'] / df['TR14']
        df['-DI14'] = 100 * df['-DM14'] / df['TR14']
        
        # Handle division by zero and NaN values
        sum_di = df['+DI14'] + df['-DI14']
        sum_di = sum_di.replace(0, np.nan)
        df['DX'] = 100 * abs(df['+DI14'] - df['-DI14']) / sum_di
        df['DX'] = df['DX'].fillna(0)  # Replace NaN with 0
        df['ADX'] = df['DX'].rolling(window=period).mean()
        
        return df
        
    except Exception as e:
        print(f"Error calculating ADX: {e}")
        df['ADX'] = np.nan
        return df
    
# ---------------------------
# Basic Forecasting Methods
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
        
        # Force first forecast point to equal last actual close
        forecast[0] = y[-1]
        forecast = forecast.tolist()
        
        print(f"Polynomial regression forecast (degree={degree}): {forecast}")
        return forecast
    except Exception as e:
        print(f"Error in polynomial regression forecast: {e}")
        # Return flat forecast if error
        return [data["Close"].iloc[-1]] * periods

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
# Mean Reversion Forecast
# ---------------------------
def mean_reversion_forecast(data, periods=5):
    """
    Generate a mean-reversion forecast based on historical data.
    """
    try:
        # Calculate recent mean and current deviation
        window = min(20, len(data))
        mean_price = data["Close"].rolling(window=window).mean().iloc[-1]
        current_price = data["Close"].iloc[-1]
        deviation = current_price - mean_price
        
        # Calculate the mean reversion strength based on how far price is from mean
        std_dev = data["Close"].rolling(window=window).std().iloc[-1]
        reversion_speed = min(0.3, abs(deviation) / (2 * std_dev)) if std_dev > 0 else 0.1
        
        # Generate a reverting forecast
        forecast = []
        last_price = current_price
        
        for i in range(periods):
            # Calculate reversion component
            reversion = (mean_price - last_price) * reversion_speed
            
            # Add some noise
            noise = std_dev * np.random.normal(0, 0.3)
            
            # Calculate new price
            new_price = last_price + reversion + noise
            
            # Update for next iteration
            forecast.append(float(new_price))
            last_price = new_price
        
        print(f"Mean reversion forecast: {forecast}")
        return forecast
    except Exception as e:
        print(f"Error in mean reversion forecast: {e}")
        # Fall back to a simple forecast
        return [data["Close"].iloc[-1]] * periods
    
# ---------------------------
# Enhanced Forecasting System with Extended Hours Support
# ---------------------------
def enhanced_forecast(data, periods=5, timeframe="1day"):
    """
    Generate enhanced forecasts with more realistic price movements showing ups and downs.
    This combines trend forecasting with historical volatility patterns.
    Now supports extended hours data.
    """
    try:
        # Extract the closing prices
        close_prices = data["Close"].values
        
        # Determine if this is intraday or daily data
        is_intraday = timeframe.endswith('min') or timeframe.endswith('h')
        
        # Check if we have extended hours data
        has_extended_hours = 'session' in data.columns
        
        # 1. Get base forecast (trend component)
        if is_intraday:
            # Use polynomial regression for intraday
            base_forecast = linear_regression_forecast(data, periods, degree=2)
        else:
            # Use ARIMA for daily data
            try:
                arima_model = create_arima_model(data)
                base_forecast = arima_prediction(arima_model)
            except:
                # Fall back to linear regression if ARIMA fails
                base_forecast = linear_regression_forecast(data, periods, degree=1)
        
        # 2. Calculate volatility metrics from historical data
        # - Recent volatility (standard deviation of returns)
        returns = np.diff(close_prices) / close_prices[:-1]
        recent_volatility = np.std(returns[-min(30, len(returns)):])
        
        # - Average daily price movement as percentage
        avg_daily_movement = np.mean(np.abs(returns[-min(30, len(returns)):]))
        
        # - Calculate how often prices change direction
        direction_changes = np.sum(np.diff(np.signbit(np.diff(close_prices))))
        direction_change_frequency = direction_changes / (len(close_prices) - 2) if len(close_prices) > 2 else 0.3
        
        # If we have extended hours data, adjust volatility based on extended hours patterns
        if has_extended_hours:
            try:
                # Get volatility by session type
                regular_data = data[data['session'] == 'regular']
                pre_market_data = data[data['session'] == 'pre-market']
                after_hours_data = data[data['session'] == 'after-hours']
                
                # Calculate volatility for each session type if we have enough data
                if len(regular_data) > 5:
                    regular_returns = regular_data['Close'].pct_change().dropna()
                    regular_volatility = np.std(regular_returns)
                else:
                    regular_volatility = recent_volatility
                
                if len(pre_market_data) > 5:
                    pre_market_returns = pre_market_data['Close'].pct_change().dropna()
                    pre_market_volatility = np.std(pre_market_returns)
                    # Adjust overall volatility based on pre-market activity
                    if pre_market_volatility > regular_volatility * 1.2:
                        # If pre-market is more volatile, slightly increase forecast volatility
                        recent_volatility *= 1.1
                
                if len(after_hours_data) > 5:
                    after_hours_returns = after_hours_data['Close'].pct_change().dropna()
                    after_hours_volatility = np.std(after_hours_returns)
                    # Adjust overall volatility based on after-hours activity
                    if after_hours_volatility > regular_volatility * 1.2:
                        # If after-hours is more volatile, slightly increase forecast volatility
                        recent_volatility *= 1.1
            except Exception as e:
                print(f"Error analyzing extended hours volatility: {e}")
        
        # 3. Add realistic movement patterns
        enhanced_forecast = []
        last_price = close_prices[-1]
        last_direction = 1  # Start with upward movement
        
        for i in range(periods):
            # Get the trend component
            trend = base_forecast[i]
            
            # Determine if we should change direction based on historical frequency
            if np.random.random() < direction_change_frequency:
                last_direction *= -1
            
            # Generate a random component based on historical volatility
            # More volatile stocks will have larger random components
            random_component = last_price * recent_volatility * np.random.normal(0, 1.5)
            
            # Combine trend and random component with direction bias
            if i == 0:
                # First forecast point should be closer to the last actual price
                weight_random = 0.3
            else:
                # Later points can deviate more
                weight_random = 0.6
            
            # Calculate the forecast with random component
            new_price = trend + (random_component * weight_random * last_direction)
            
            # Ensure some minimum movement
            min_movement = last_price * avg_daily_movement * 0.5
            if abs(new_price - last_price) < min_movement:
                new_price = last_price + (min_movement * last_direction)
            
            # Add some persistence to avoid unrealistic jumps
            if i > 0:
                # Pull slightly toward previous forecast point
                new_price = 0.7 * new_price + 0.3 * enhanced_forecast[-1]
            
            # Ensure the forecast doesn't go negative
            new_price = max(new_price, 0.01 * last_price)
            
            enhanced_forecast.append(float(new_price))
            last_price = new_price
        
        # 4. Ensure the forecast maintains overall trend direction from the base forecast
        trend_direction = 1 if base_forecast[-1] > base_forecast[0] else -1
        actual_direction = 1 if enhanced_forecast[-1] > enhanced_forecast[0] else -1
        
        if trend_direction != actual_direction:
            # Adjust the last point to maintain the overall trend direction
            enhanced_forecast[-1] = enhanced_forecast[0] + abs(enhanced_forecast[-1] - enhanced_forecast[0]) * trend_direction
        
        print(f"Enhanced forecast: {enhanced_forecast}")
        return enhanced_forecast
        
    except Exception as e:
        print(f"Error in enhanced forecast: {e}")
        # Fall back to a simple forecast
        return [data["Close"].iloc[-1]] * periods

# ---------------------------
# Advanced Forecasting Methods with Extended Hours Support
# ---------------------------

def create_extended_hours_features(df):
    """
    Create features based on extended hours data.
    
    Args:
        df (pd.DataFrame): DataFrame with price data and 'session' column
        
    Returns:
        pd.DataFrame: DataFrame with additional features
    """
    # Make a copy to avoid modifying the original
    result = df.copy()
    
    if 'session' not in result.columns:
        return result
    
    # Calculate pre-market to previous close change
    result['pre_market_change'] = 0.0
    result['after_hours_change'] = 0.0
    
    # Group by date to find pre-market and after-hours changes
    if isinstance(result.index, pd.DatetimeIndex):
        # Extract dates for grouping
        result['date'] = result.index.date
        unique_dates = sorted(result['date'].unique())
        
        for i, date in enumerate(unique_dates):
            date_mask = result['date'] == date
            day_data = result[date_mask]
            
            # Process pre-market data (compare to previous day close)
            pre_market_data = day_data[day_data['session'] == 'pre-market']
            if not pre_market_data.empty and i > 0:
                # Get previous day data
                prev_date = unique_dates[i-1]
                prev_date_mask = result['date'] == prev_date
                prev_day_data = result[prev_date_mask]
                
                if not prev_day_data.empty:
                    # Get last price from previous day (regular or after-hours)
                    prev_close = prev_day_data['Close'].iloc[-1]
                    first_pre_market = pre_market_data['Open'].iloc[0]
                    
                    # Calculate change
                    if prev_close > 0:
                        pre_market_change = (first_pre_market - prev_close) / prev_close
                        result.loc[date_mask, 'pre_market_change'] = pre_market_change
            
            # Process after-hours data (compare to regular hours close)
            regular_data = day_data[day_data['session'] == 'regular']
            after_hours_data = day_data[day_data['session'] == 'after-hours']
            
            if not regular_data.empty and not after_hours_data.empty:
                regular_close = regular_data['Close'].iloc[-1]
                after_hours_close = after_hours_data['Close'].iloc[-1]
                
                if regular_close > 0:
                    after_hours_change = (after_hours_close - regular_close) / regular_close
                    result.loc[date_mask, 'after_hours_change'] = after_hours_change
    
    # Create rolling averages for these features
    result['pre_market_change_5d'] = result['pre_market_change'].rolling(5).mean()
    result['after_hours_change_5d'] = result['after_hours_change'].rolling(5).mean()
    
    # Fill NA values with 0
    for col in ['pre_market_change', 'after_hours_change', 
                'pre_market_change_5d', 'after_hours_change_5d']:
        result[col] = result[col].fillna(0)
    
    # Remove the temporary date column
    if 'date' in result.columns:
        result.drop('date', axis=1, inplace=True)
    
    return result

def adjust_forecast_volatility(forecast, data):
    """
    Add realistic volatility while preserving the trend.
    Now considers extended hours patterns if available.
    
    Args:
        forecast (list): Base forecast values
        data (pd.DataFrame): Historical data with optional 'session' column
    
    Returns:
        list: Forecast with adjusted volatility
    """
    returns = np.diff(data["Close"].values) / data["Close"].values[:-1]
    recent_volatility = np.std(returns[-min(30, len(returns)):])
    
    # Check if we have extended hours data
    has_extended_hours = 'session' in data.columns
    session_volatilities = {}
    
    if has_extended_hours:
        try:
            # Calculate volatility by session type
            for session in ['regular', 'pre-market', 'after-hours']:
                session_data = data[data['session'] == session]
                if len(session_data) > 5:
                    session_returns = session_data['Close'].pct_change().dropna()
                    session_volatilities[session] = np.std(session_returns)
            
            # If we don't have enough data for a session, use the overall volatility
            for session in ['regular', 'pre-market', 'after-hours']:
                if session not in session_volatilities:
                    session_volatilities[session] = recent_volatility
        except Exception as e:
            print(f"Error calculating session volatilities: {e}")
            # Default to using overall volatility
            session_volatilities = {
                'regular': recent_volatility,
                'pre-market': recent_volatility,
                'after-hours': recent_volatility
            }
    
    adjusted_forecast = [forecast[0]]  # Keep first point unchanged
    for i in range(1, len(forecast)):
        # Add volatility that follows a similar pattern to historical data
        trend_component = forecast[i] - forecast[i-1]
        
        # If we have extended hours data, alternate volatility patterns
        if has_extended_hours and len(session_volatilities) > 0:
            # Cycle through sessions for more realistic variation
            sessions = list(session_volatilities.keys())
            session = sessions[i % len(sessions)]
            vol = session_volatilities.get(session, recent_volatility)
            
            # Pre-market often has larger moves
            if session == 'pre-market':
                vol *= 1.2
        else:
            vol = recent_volatility
        
        random_component = forecast[i-1] * vol * np.random.normal(0, 0.8)
        
        # Add autocorrelation - volatility tends to cluster
        if i > 1:
            # If previous point was up, slightly higher chance of being up again
            prev_direction = 1 if adjusted_forecast[i-1] > adjusted_forecast[i-2] else -1
            random_component = random_component * 0.8 + prev_direction * abs(random_component) * 0.2
        
        new_price = forecast[i-1] + trend_component + random_component
        adjusted_forecast.append(float(new_price))
    
    return adjusted_forecast

def improved_ensemble_forecast(data, periods=5, timeframe="1day"):
    """
    Enhanced ensemble forecast with dynamic model weighting based on recent performance.
    Now considers extended hours data when available.
    
    Args:
        data (pd.DataFrame): Historical price data
        periods (int): Number of periods to forecast
        timeframe (str): Timeframe of the data
        
    Returns:
        list: Forecast values
    """
    if len(data) < 30:
        return enhanced_forecast(data, periods, timeframe)
    
    try:
        # Check if we have session data (extended hours)
        has_extended_hours = 'session' in data.columns
        
        # Prepare data and indicators
        df = calculate_technical_indicators(data)
        df = df.dropna(subset=['Close'])
        
        # If we have extended hours data, create additional features
        if has_extended_hours:
            # Create extended hours features
            df = create_extended_hours_features(df)
        
        # Create features from lagged prices and indicators
        features = create_features(df)
        
        # Add extended hours features if available
        if has_extended_hours:
            for col in ['pre_market_change', 'after_hours_change', 
                       'pre_market_change_5d', 'after_hours_change_5d']:
                if col in df.columns:
                    features[col] = df[col]
        
        # Train multiple models
        models = {
            "linear": train_linear_model(features, df['Close']),
            "ridge": train_ridge_model(features, df['Close']),
            "gradient_boost": train_gb_model(features, df['Close'])
        }
        
        # Try SVM if we have enough data
        if len(df) >= 30:
            models["svm"] = train_svm_model(features, df['Close'])
        
        # Evaluate models to determine weights
        model_errors = {}
        for name, model in models.items():
            if model is not None:
                # Use simple in-sample error for weight calculation
                if isinstance(model, tuple) and len(model) == 2:
                    svm_model, scaler = model
                    features_scaled = scaler.transform(features)
                    preds = svm_model.predict(features_scaled)
                else:
                    preds = model.predict(features)
                
                actuals = df['Close'].values[:len(preds)]
                if len(actuals) > 0 and len(preds) > 0:
                    mape = np.mean(np.abs((actuals - preds) / actuals)) * 100
                    model_errors[name] = mape
                else:
                    model_errors[name] = 15  # Default error if can't calculate
        
        # Add error estimates for other methods
        if timeframe.endswith('min') or timeframe.endswith('h'):
            model_errors["poly_reg"] = 10
        else:
            model_errors["arima"] = 12
        
        # Enhanced forecast typically performs well
        model_errors["enhanced"] = 8
        
        # Convert errors to weights (lower error = higher weight)
        weights = {}
        total_error = sum(1/err for err in model_errors.values() if err > 0)
        for model_name, error in model_errors.items():
            if error > 0:
                weights[model_name] = (1/error) / total_error
            else:
                weights[model_name] = 0.1  # Default weight for zero error (unlikely)
        
        print(f"Model weights based on performance: {weights}")
        
        # Generate forecasts using each model
        predictions = {}
        for name, model in models.items():
            if model is not None:
                predictions[name] = generate_model_predictions(model, features, df, periods)
        
        # Add ARIMA/polynomial and enhanced forecasts
        if timeframe.endswith('min') or timeframe.endswith('h'):
            predictions["poly_reg"] = linear_regression_forecast(data, periods, degree=2)
        else:
            try:
                arima_model = create_arima_model(data)
                predictions["arima"] = arima_prediction(arima_model)
            except:
                predictions["poly_reg"] = linear_regression_forecast(data, periods, degree=1)
        
        # Enhanced forecast with extended hours awareness
        predictions["enhanced"] = enhanced_forecast(data, periods, timeframe)
        
        # Apply weights to create ensemble forecast
        ensemble_forecast = []
        for i in range(periods):
            weighted_sum = 0
            weight_total = 0
            for model_name, prediction in predictions.items():
                if model_name in weights and i < len(prediction):
                    weighted_sum += prediction[i] * weights.get(model_name, 0)
                    weight_total += weights.get(model_name, 0)
            
            ensemble_val = weighted_sum / weight_total if weight_total > 0 else data["Close"].iloc[-1]
            ensemble_forecast.append(float(ensemble_val))
        
        # Adjust for volatility - now session-aware if extended hours data is available
        return adjust_forecast_volatility(ensemble_forecast, data)
        
    except Exception as e:
        print(f"Error in improved ensemble forecast: {e}")
        return enhanced_forecast(data, periods, timeframe)

def market_aware_forecast(data, periods=5, timeframe="1day", symbol="AAPL"):
    """
    Forecast that incorporates market sentiment, sector performance, and now extended hours data.
    
    Args:
        data (pd.DataFrame): Historical price data
        periods (int): Number of periods to forecast
        timeframe (str): Time period for the data
        symbol (str): Stock symbol
        
    Returns:
        list: Forecast values
    """
    try:
        # Get baseline forecast with extended hours awareness
        regime = detect_market_regime(data)
        if regime in ["trending_up", "trending_down"]:
            baseline = improved_ensemble_forecast(data, periods, timeframe)
        elif regime == "mean_reverting":
            baseline = mean_reversion_forecast(data, periods)
        elif regime == "volatile":
            baseline = enhanced_forecast(data, periods, timeframe)
        else:
            baseline = improved_ensemble_forecast(data, periods, timeframe)
        
        # Check if we have extended hours data
        has_extended_hours = 'session' in data.columns
        
        # Adjust for extended hours sentiment if available
        if has_extended_hours and timeframe.endswith(('min', 'h')):
            try:
                # Analyze extended hours patterns
                regular_data = data[data['session'] == 'regular']
                pre_market_data = data[data['session'] == 'pre-market']
                after_hours_data = data[data['session'] == 'after-hours']
                
                if not pre_market_data.empty and not regular_data.empty:
                    # Calculate pre-market sentiment
                    pre_market_change = pre_market_data['Close'].pct_change().mean() * 100
                    
                    # Incorporate pre-market sentiment into forecast
                    if abs(pre_market_change) > 1.0:  # Only adjust if significant
                        adjustment_factor = 1.0 + (pre_market_change * 0.02)
                        baseline = [price * adjustment_factor for price in baseline]
                
                if not after_hours_data.empty and not regular_data.empty:
                    # Calculate after-hours sentiment
                    after_hours_change = after_hours_data['Close'].pct_change().mean() * 100
                    
                    # Incorporate after-hours sentiment into forecast
                    if abs(after_hours_change) > 1.0:  # Only adjust if significant
                        adjustment_factor = 1.0 + (after_hours_change * 0.015)
                        baseline = [price * adjustment_factor for price in baseline]
            except Exception as e:
                print(f"Error adjusting for extended hours sentiment: {e}")
        
        return baseline
    except Exception as e:
        print(f"Error in market-aware forecast: {e}")
        return improved_ensemble_forecast(data, periods, timeframe)
    
# ---------------------------
# Machine Learning Features and Models
# ---------------------------
def create_features(df, target_col='Close', window=10):
    """
    Create features for machine learning models.
    Now includes extended hours features if available.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        target_col (str): Target column for prediction
        window (int): Window size for rolling features
        
    Returns:
        pd.DataFrame: DataFrame with features
    """
    window = min(window, len(df) // 3)
    if window < 2:
        return pd.DataFrame(index=df.index)
    
    X = pd.DataFrame(index=df.index)
    
    # Lagged prices
    for i in range(1, window + 1):
        X[f'lag_{i}'] = df[target_col].shift(i)
    
    # Price changes
    X['return_1d'] = df[target_col].pct_change(1)
    X['return_5d'] = df[target_col].pct_change(window // 2)
    
    # Rolling statistics
    X['rolling_mean'] = df[target_col].rolling(window=window).mean()
    X['rolling_std'] = df[target_col].rolling(window=window).std()
    
    # Technical indicators if available
    for col in ['RSI', 'MACD', 'ATR', 'SMA_20', 'BB_Upper', 'BB_Lower']:
        if col in df.columns:
            X[col] = df[col]
    
    # Extended hours features if available
    for col in ['pre_market_change', 'after_hours_change', 
               'pre_market_change_5d', 'after_hours_change_5d']:
        if col in df.columns:
            X[col] = df[col]
    
    # Session indicator variables if available
    if 'session' in df.columns:
        X['is_pre_market'] = (df['session'] == 'pre-market').astype(int)
        X['is_after_hours'] = (df['session'] == 'after-hours').astype(int)
    
    # Fill NA values
    X = X.fillna(method='bfill').fillna(method='ffill').fillna(0)
    
    return X

def train_linear_model(X, y):
    """Train a linear regression model."""
    try:
        model = LinearRegression()
        # Drop rows with NaN values
        X_clean = X.dropna()
        y_clean = y.loc[X_clean.index]
        
        if len(X_clean) < 10:  # Not enough data
            return None
        
        model.fit(X_clean, y_clean)
        return model
    except Exception as e:
        print(f"Error training linear model: {e}")
        return None

def train_ridge_model(X, y):
    """Train a ridge regression model."""
    try:
        model = Ridge(alpha=1.0)
        # Drop rows with NaN values
        X_clean = X.dropna()
        y_clean = y.loc[X_clean.index]
        
        if len(X_clean) < 10:  # Not enough data
            return None
        
        model.fit(X_clean, y_clean)
        return model
    except Exception as e:
        print(f"Error training ridge model: {e}")
        return None

def train_gb_model(X, y):
    """Train a gradient boosting regressor model."""
    try:
        model = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)
        # Drop rows with NaN values
        X_clean = X.dropna()
        y_clean = y.loc[X_clean.index]
        
        if len(X_clean) < 10:  # Not enough data
            return None
        
        model.fit(X_clean, y_clean)
        return model
    except Exception as e:
        print(f"Error training gradient boosting model: {e}")
        return None

def train_svm_model(X, y):
    """Train an SVR model."""
    try:
        # Scale the data
        scaler = StandardScaler()
        # Drop rows with NaN values
        X_clean = X.dropna()
        y_clean = y.loc[X_clean.index]
        
        if len(X_clean) < 10:  # Not enough data
            return None
        
        X_scaled = scaler.fit_transform(X_clean)
        model = SVR(kernel='rbf', C=100, gamma=0.1)
        model.fit(X_scaled, y_clean)
        
        # Return model and scaler as a tuple
        return (model, scaler)
    except Exception as e:
        print(f"Error training SVM model: {e}")
        return None

def generate_model_predictions(model, features, data, periods):
    """Generate predictions from a trained model."""
    if model is None:
        return [data["Close"].iloc[-1]] * periods
    
    try:
        predictions = []
        current_features = features.iloc[-1:].copy()
        
        # For SVR, we need to scale the data
        if isinstance(model, tuple) and len(model) == 2:
            svm_model, scaler = model
            current_features_scaled = scaler.transform(current_features)
            last_pred = svm_model.predict(current_features_scaled)[0]
        else:
            last_pred = model.predict(current_features)[0]
        
        predictions.append(float(last_pred))
        
        # For simplicity, assume the features remain relatively constant
        # In a more advanced implementation, you would update the features based on each prediction
        for _ in range(1, periods):
            # Add some random variation
            variation = np.random.normal(0, 0.01) * predictions[-1]
            next_pred = predictions[-1] * (1 + variation)
            predictions.append(float(next_pred))
        
        return predictions
    except Exception as e:
        print(f"Error generating model predictions: {e}")
        return [data["Close"].iloc[-1]] * periods
    
# ---------------------------
# Generate OHLC data for forecast points
# ---------------------------
def generate_forecast_ohlc(data, forecast):
    """
    Generate OHLC values for forecast points with more realistic patterns.
    Now considers extended hours patterns if available.
    
    Args:
        data (pd.DataFrame): Historical data
        forecast (list): List of forecast prices
        
    Returns:
        list: List of OHLC dictionaries for forecast points
    """
    # Get average daily volatility metrics from historical data
    avg_range = (data["High"] - data["Low"]).mean()
    avg_body_size = abs(data["Open"] - data["Close"]).mean()
    
    # Check if we have extended hours data
    has_extended_hours = 'session' in data.columns
    session_volatilities = {}
    
    if has_extended_hours:
        try:
            # Calculate metrics by session type
            for session in ['regular', 'pre-market', 'after-hours']:
                session_data = data[data['session'] == session]
                if len(session_data) >= 5:
                    session_volatilities[session] = {
                        'range': (session_data["High"] - session_data["Low"]).mean(),
                        'body': abs(session_data["Open"] - session_data["Close"]).mean()
                    }
        except Exception as e:
            print(f"Error calculating session metrics: {e}")
    
    # Calculate the direction of each forecast day (up or down)
    forecast_ohlc = []
    last_close = data["Close"].iloc[-1]
    
    for i, close in enumerate(forecast):
        if i == 0:
            prev_close = last_close
        else:
            prev_close = forecast[i-1]
        
        direction = 1 if close > prev_close else -1
        
        # Determine which session metrics to use (cycle through sessions if available)
        if has_extended_hours and session_volatilities:
            sessions = list(session_volatilities.keys())
            session = sessions[i % len(sessions)]
            metrics = session_volatilities.get(session, {'range': avg_range, 'body': avg_body_size})
            
            # Store session for reference
            current_session = session
        else:
            metrics = {'range': avg_range, 'body': avg_body_size}
            current_session = 'regular'
        
        # Calculate volatility factor - more movement for later forecast days
        volatility_factor = 1.0 + (i * 0.1)  # Increases volatility for later days
        
        # Calculate open based on previous close with some random variation
        random_open_factor = np.random.normal(0, 0.3)
        open_deviation = metrics['body'] * random_open_factor * volatility_factor
        open_price = prev_close + open_deviation
        
        # Determine high and low based on average range, direction, and volatility
        day_range = metrics['range'] * volatility_factor * (0.8 + np.random.random() * 0.4)
        
        if direction > 0:
            # Upward day
            body_height = abs(close - open_price)
            upper_wick = day_range * (0.4 + np.random.random() * 0.3)  # Upper wick 40-70% of range
            lower_wick = day_range - body_height - upper_wick
            high = max(close, open_price) + upper_wick
            low = min(close, open_price) - lower_wick
        else:
            # Downward day
            body_height = abs(close - open_price)
            lower_wick = day_range * (0.4 + np.random.random() * 0.3)  # Lower wick 40-70% of range
            upper_wick = day_range - body_height - lower_wick
            high = max(close, open_price) + upper_wick
            low = min(close, open_price) - lower_wick
        
        # Make sure high is at least slightly higher than both open and close
        high = max(high, open_price * 1.001, close * 1.001)
        
        # Make sure low is at least slightly lower than both open and close
        low = min(low, open_price * 0.999, close * 0.999)
        
        ohlc_point = {
            "open": float(open_price),
            "high": float(high),
            "low": float(low),
            "close": float(close)
        }
        
        # Add session information if we're using extended hours data
        if has_extended_hours:
            ohlc_point["session"] = current_session
        
        forecast_ohlc.append(ohlc_point)
    
    return forecast_ohlc

# ---------------------------
# Build Raw Chart Data for Front-End
# ---------------------------
def get_chart_data(data, forecast, timeframe):
    """
    Build raw chart data arrays including ISO-formatted historical and forecast dates and values.
    Now includes OHLC data for both historical and forecast points and session markers.
    
    Args:
        data (pd.DataFrame): Historical price data
        forecast (list): Forecasted prices
        timeframe (str): Time period for the data
    
    Returns:
        dict: Chart data for frontend rendering
    """
    historical_dates = data.index.strftime("%Y-%m-%dT%H:%M:%SZ").tolist()
    historical_values = data["Close"].tolist()
    
    # Add OHLC data if available
    historical_ohlc = None
    if {"Open", "High", "Low", "Close"}.issubset(data.columns):
        historical_ohlc = []
        for i, (_, row) in enumerate(data.iterrows()):
            ohlc_point = {
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"])
            }
            
            # Add session marker if available
            if 'session' in row:
                ohlc_point["session"] = row["session"]
            
            historical_ohlc.append(ohlc_point)
    
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
        "timeframe": timeframe,
        "includesExtendedHours": 'session' in data.columns
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
    - If OHLC columns are present (daily data), create a dark-themed candlestick chart.
    - For intraday data (or if OHLC is missing), fall back to a simple dark line chart.
    Now includes extended hours visualization if available.
    
    Args:
        data (pd.DataFrame): Historical price data
        symbol (str): Stock symbol
        forecast (list, optional): Forecast prices
        timeframe (str): Time period for the data
        
    Returns:
        str: Chart filename
    """
    try:
        os.makedirs("static", exist_ok=True)
        filename = f"chart_{symbol.upper()}.png"
        filepath = os.path.join("static", filename)
        
        # Check if we have extended hours data
        has_extended_hours = 'session' in data.columns
        
        # Check for OHLC => use candlestick
        if {"Open", "High", "Low", "Close"}.issubset(data.columns):
            try:
                import mplfinance as mpf
                data_filled = data.ffill()
                dark_style = create_dark_style()
                
                # If we have extended hours data, potentially split by session
                if has_extended_hours:
                    try:
                        # Create plot with extended hours
                        fig, axes = plt.subplots(figsize=(10, 6), facecolor='black')
                        axes.set_facecolor('black')
                        
                        # Split data by session type
                        regular_data = data_filled[data_filled['session'] == 'regular'].copy()
                        pre_market_data = data_filled[data_filled['session'] == 'pre-market'].copy()
                        after_hours_data = data_filled[data_filled['session'] == 'after-hours'].copy()
                        
                        # Plot each session type with different colors
                        if not regular_data.empty:
                            mpf.plot(
                                regular_data,
                                type='candle',
                                style=dark_style,
                                ax=axes,
                                title=f"{symbol.upper()} with Extended Hours",
                                ylabel="Price",
                                datetime_format='%Y-%m-%d %H:%M',
                                show_nontrading=True
                            )
                        
                        # Add pre-market with different color
                        if not pre_market_data.empty:
                            # Create and apply lighter style for pre-market
                            pre_market_mc = mpf.make_marketcolors(
                                up='lightblue', down='purple',
                                edge='white', wick='white', volume='in'
                            )
                            pre_market_style = mpf.make_mpf_style(
                                base_mpf_style='nightclouds',
                                marketcolors=pre_market_mc
                            )
                            
                            mpf.plot(
                                pre_market_data,
                                type='candle',
                                style=pre_market_style,
                                ax=axes,
                                datetime_format='%Y-%m-%d %H:%M',
                                show_nontrading=True,
                                addplot=[mpf.make_addplot(pre_market_data['Close'], ax=axes, scatter=False, color='cyan', linestyle='--', width=0.7)]
                            )
                        
                        # Add after-hours with different color
                        if not after_hours_data.empty:
                            # Create and apply lighter style for after-hours
                            after_hours_mc = mpf.make_marketcolors(
                                up='lightgreen', down='orange',
                                edge='white', wick='white', volume='in'
                            )
                            after_hours_style = mpf.make_mpf_style(
                                base_mpf_style='nightclouds',
                                marketcolors=after_hours_mc
                            )
                            
                            mpf.plot(
                                after_hours_data,
                                type='candle',
                                style=after_hours_style,
                                ax=axes,
                                datetime_format='%Y-%m-%d %H:%M',
                                show_nontrading=True,
                                addplot=[mpf.make_addplot(after_hours_data['Close'], ax=axes, scatter=False, color='orange', linestyle='--', width=0.7)]
                            )
                        
                        # Add legend to distinguish sessions
                        from matplotlib.lines import Line2D
                        legend_elements = [
                            Line2D([0], [0], color='white', marker='s', markersize=10, label='Regular Hours', markerfacecolor='lime'),
                            Line2D([0], [0], color='white', marker='s', markersize=10, label='Pre-Market', markerfacecolor='lightblue'),
                            Line2D([0], [0], color='white', marker='s', markersize=10, label='After-Hours', markerfacecolor='orange')
                        ]
                        axes.legend(handles=legend_elements, loc='upper left')
                        
                        # Overlay forecast if available
                        if forecast and len(forecast) > 0:
                            chart_info = get_chart_data(data_filled, forecast, timeframe)
                            f_dates = pd.to_datetime(chart_info["forecastDates"])
                            last_close = data_filled["Close"].iloc[-1]
                            
                            # Flat connector
                            axes.plot(
                                [data_filled.index[-1], f_dates[0]],
                                [last_close, last_close],
                                linestyle="--", color="yellow", linewidth=2, label="Connector"
                            )
                            
                            # Forecast line
                            axes.plot(
                                [data_filled.index[-1]] + list(f_dates),
                                [last_close] + forecast,
                                linestyle="--", marker="o", color="cyan", linewidth=2, label="Forecast"
                            )
                        
                        fig.savefig(filepath, facecolor=fig.get_facecolor(), edgecolor='none')
                        plt.close(fig)
                        
                    except Exception as e:
                        print(f"Error plotting extended hours: {e}")
                        # Fall back to standard plot
                        has_extended_hours = False
                
                # Standard candlestick plot (if no extended hours or the extended hours plot failed)
                if not has_extended_hours:
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
                # Fall back to line chart
                plt.style.use("dark_background")
                plt.figure(figsize=(10, 5))
                
                if has_extended_hours:
                    # Different colors for different sessions
                    sessions = data['session'].unique()
                    for session in sessions:
                        session_data = data[data['session'] == session]
                        if not session_data.empty:
                            color = 'white' if session == 'regular' else 'cyan' if session == 'pre-market' else 'orange'
                            label = 'Regular Hours' if session == 'regular' else 'Pre-Market' if session == 'pre-market' else 'After-Hours'
                            plt.plot(session_data.index, session_data["Close"], label=label, color=color)
                else:
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
            
            if has_extended_hours:
                # Different colors for different sessions
                sessions = data['session'].unique()
                for session in sessions:
                    session_data = data[data['session'] == session]
                    if not session_data.empty:
                        color = 'white' if session == 'regular' else 'cyan' if session == 'pre-market' else 'orange'
                        label = 'Regular Hours' if session == 'regular' else 'Pre-Market' if session == 'pre-market' else 'After-Hours'
                        plt.plot(session_data.index, session_data["Close"], label=label, color=color)
            else:
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
    except Exception as e:
        print(f"Error generating chart: {e}")
        # Return a default filename if chart generation fails
        return "chart_error.png"

# Helper function to extract key indicators for response
def extract_key_indicators(data_with_indicators):
    """Extract key technical indicators for the response."""
    indicators = {}
    try:
        # Get the most recent values of key indicators
        for indicator in ['RSI', 'MACD', 'ATR', 'SMA_20', 'SMA_50', 'BB_Upper', 'BB_Lower']:
            if indicator in data_with_indicators.columns:
                indicators[indicator] = float(data_with_indicators[indicator].iloc[-1])
        
        # Add derived indicators
        if 'Close' in data_with_indicators.columns:
            last_close = data_with_indicators['Close'].iloc[-1]
            
            # Trend determination
            if 'SMA_20' in indicators and 'SMA_50' in indicators:
                indicators['trend'] = 'bullish' if indicators['SMA_20'] > indicators['SMA_50'] else 'bearish'
            
            # Bollinger Band position
            if 'BB_Upper' in indicators and 'BB_Lower' in indicators:
                bb_width = indicators['BB_Upper'] - indicators['BB_Lower']
                if bb_width > 0:
                    bb_position = (last_close - indicators['BB_Lower']) / bb_width
                    indicators['BB_position'] = float(bb_position)
        
        # Add extended hours indicators if available
        for eh_indicator in ['pre_market_change', 'after_hours_change']:
            if eh_indicator in data_with_indicators.columns:
                # Get the most recent value that's not zero
                values = data_with_indicators[eh_indicator].values
                non_zero_values = values[values != 0]
                if len(non_zero_values) > 0:
                    indicators[eh_indicator] = float(non_zero_values[-1])
        
        return indicators
    except Exception as e:
        print(f"Error extracting indicators: {e}")
        return {}
    
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
    news_count = int(request.args.get("news_count", "5"))
    risk_appetite = request.args.get("risk_appetite", "moderate")
    # Extended hours is always enabled
    include_extended_hours = True
    
    print(f"Received request for symbol: {symbol} with timeframe: {timeframe}, extended hours always included")
    
    try:
        # Use a timer to track execution time
        start_time = datetime.now()
        
        # Fetch data with extended hours always enabled
        data = fetch_data(symbol, timeframe, include_extended_hours)
        
        # Generate enhanced forecast with new improved methods
        try:
            # Use the new market-aware forecast with all improvements
            forecast = market_aware_forecast(data, periods=5, timeframe=timeframe, symbol=symbol)
        except Exception as e:
            print(f"Error in market-aware forecast, using fallback: {e}")
            # Fall back to improved ensemble
            try:
                forecast = improved_ensemble_forecast(data, periods=5, timeframe=timeframe)
            except Exception as e:
                print(f"Error in improved ensemble forecast, using fallback: {e}")
                # Fall back to regime-aware
                try:
                    forecast = regime_aware_forecast(data, periods=5, timeframe=timeframe)
                except Exception as e:
                    print(f"Error in regime-aware forecast, using fallback: {e}")
                    # Fall back to enhanced forecast
                    try:
                        forecast = enhanced_forecast(data, periods=5, timeframe=timeframe)
                    except Exception as e:
                        print(f"Error in enhanced forecast, using basic forecast: {e}")
                        # Fall back to basic forecast
                        if timeframe.endswith('min') or timeframe.endswith('h'):
                            forecast = linear_regression_forecast(data, periods=5, degree=2)
                        else:
                            try:
                                arima_model = create_arima_model(data)
                                forecast = arima_prediction(arima_model)
                            except:
                                forecast = linear_regression_forecast(data, periods=5, degree=1)
        
        # Prepare chart data - needed for the UI
        chart_data = get_chart_data(data, forecast, timeframe)
        
        # Add flag for extended hours
        has_extended_hours = 'session' in data.columns
        
        # Start with a basic response that will work even if other parts timeout
        response = {
            "forecast": forecast,
            "chartData": {"symbol": symbol.upper(), **chart_data},
            "news": [{"title": "Loading news...", "source": {"name": "Trading System"}, "summary": "News will be available on next refresh."}],
            "openai_refined_prediction": f"Analysis for {symbol}: Based on technical analysis, the forecast suggests a trend from ${data['Close'].iloc[-1]:.2f} to ${forecast[-1]:.2f} over the {timeframe} timeframe.",
            "includesExtendedHours": has_extended_hours
        }
        
        # Generate chart in the background
        try:
            chart_filename = generate_chart(data, symbol, forecast=forecast, timeframe=timeframe)
            response["chart_path"] = chart_filename
        except Exception as e:
            print(f"Error generating chart: {e}")
            response["chart_path"] = "chart_error.png"
        
        # Check time elapsed and prioritize remaining operations
        elapsed = (datetime.now() - start_time).total_seconds()
        if elapsed > 15:  # If we're already taking too long, return what we have
            print(f"Request taking too long ({elapsed:.2f}s), returning partial data")
            return jsonify(response)
        
        # Now process additional data in order of importance
        # 1. Technical indicators
        try:
            data_with_indicators = calculate_technical_indicators(data)
            key_indicators = extract_key_indicators(data_with_indicators)
            if key_indicators:
                response["key_indicators"] = key_indicators
        except Exception as e:
            print(f"Error calculating indicators: {e}")
            data_with_indicators = data.copy()
        
        # 2. News data
        elapsed = (datetime.now() - start_time).total_seconds()
        if elapsed < 18:  # Still have time
            try:
                news = fetch_news(symbol, max_items=news_count)
                if news:
                    response["news"] = news
            except Exception as e:
                print(f"Error fetching news: {e}")
        
        # 3. Market regime detection
        elapsed = (datetime.now() - start_time).total_seconds()
        regime = "unknown"
        if elapsed < 20:  # Still have time
            try:
                regime = detect_market_regime(data)
                response["market_regime"] = regime
                print(f"Detected market regime: {regime}")
            except Exception as e:
                print(f"Error detecting market regime: {e}")
        
        # 4. Generate AI analysis - ALWAYS try to use OpenAI
        elapsed = (datetime.now() - start_time).total_seconds()
        if elapsed < 25:  # Give more time for OpenAI API call
            try:
                # Check if OpenAI API key is available
                if not openai.api_key:
                    print("WARNING: OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
                    # Instead of falling back, raise an exception to be caught
                    raise ValueError("OpenAI API key not configured. Set OPENAI_API_KEY environment variable.")
                
                # Prepare data for OpenAI
                price_change = (forecast[-1] - data["Close"].iloc[-1]) / data["Close"].iloc[-1] * 100
                direction = "bullish" if price_change > 0 else "bearish"
                
                # Get key indicators if available
                indicators_text = ""
                if "key_indicators" in response:
                    indicators = response["key_indicators"]
                    if "RSI" in indicators:
                        indicators_text += f"RSI: {indicators['RSI']:.1f}, "
                    if "MACD" in indicators:
                        indicators_text += f"MACD: {indicators['MACD']:.4f}, "
                    if "trend" in indicators:
                        indicators_text += f"Trend: {indicators['trend']}, "
                
                # Create a detailed prompt for OpenAI
                openai_prompt = f"""
                Provide a detailed analysis for {symbol.upper()} in the {timeframe} timeframe.
                Current Information:
                - Current price: ${data['Close'].iloc[-1]:.2f}
                - Forecast end price: ${forecast[-1]:.2f}
                - Direction: {direction.upper()}
                - Projected change: {price_change:.2f}%
                - Market regime: {regime}
                - Technical indicators: {indicators_text}
                """
                
                # Add extended hours info if available
                if has_extended_hours:
                    # Get the most recent pre-market and after-hours data
                    try:
                        # Group by date and session
                        data['date'] = data.index.date
                        latest_date = max(data['date'])
                        today_data = data[data['date'] == latest_date]
                        
                        pre_market_data = today_data[today_data['session'] == 'pre-market']
                        regular_data = today_data[today_data['session'] == 'regular']
                        after_hours_data = today_data[today_data['session'] == 'after-hours']
                        
                        eh_info = "\nExtended Hours Info:\n"
                        
                        # Add pre-market info if available for today
                        if not pre_market_data.empty:
                            pre_open = pre_market_data['Open'].iloc[0]
                            pre_close = pre_market_data['Close'].iloc[-1]
                            prev_close = data['Close'][data['date'] < latest_date].iloc[-1] if any(data['date'] < latest_date) else None
                            
                            if prev_close is not None:
                                pre_market_gap = (pre_open - prev_close) / prev_close * 100
                                eh_info += f"- Pre-market gap: {pre_market_gap:.2f}%\n"
                            
                            pre_market_change = (pre_close - pre_open) / pre_open * 100
                            eh_info += f"- Pre-market session change: {pre_market_change:.2f}%\n"
                        
                        # Add after-hours info if available for today
                        if not after_hours_data.empty and not regular_data.empty:
                            reg_close = regular_data['Close'].iloc[-1]
                            after_close = after_hours_data['Close'].iloc[-1]
                            
                            after_hours_change = (after_close - reg_close) / reg_close * 100
                            eh_info += f"- After-hours session change: {after_hours_change:.2f}%\n"
                        
                        openai_prompt += eh_info
                    except Exception as e:
                        print(f"Error preparing extended hours info for OpenAI: {e}")
                
                openai_prompt += """
                Your analysis should include:
                1. Current market context for the stock
                2. Key technical levels and signals
                3. A specific trading recommendation
                4. Potential risks to watch for
                """
                
                if has_extended_hours:
                    openai_prompt += "5. How extended hours trading is affecting the stock\n"
                
                openai_prompt += """
                Keep your response concise but detailed, and format it with markdown headings.
                """
                
                # Call OpenAI with retry mechanism
                max_retries = 2
                for attempt in range(max_retries):
                    try:
                        openai_response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": "You are a professional financial analyst specializing in technical analysis and market prediction."},
                                {"role": "user", "content": openai_prompt}
                            ],
                            max_tokens=500,
                            temperature=0.7
                        )
                        ai_analysis = openai_response.choices[0].message.content
                        response["openai_refined_prediction"] = ai_analysis
                        print("Successfully generated OpenAI analysis")
                        break
                    except Exception as e:
                        print(f"OpenAI API error (attempt {attempt+1}/{max_retries}): {e}")
                        if attempt == max_retries - 1:
                            raise  # Re-raise on last attempt
                        time.sleep(1)  # Wait before retry
            except Exception as e:
                print(f"Error generating AI analysis: {e}")
                # Don't provide a fallback - make it clear there was an issue with OpenAI
                response["openai_refined_prediction"] = f"""
                # OpenAI Analysis Unavailable
                We're unable to provide an AI-powered analysis for {symbol.upper()} at this time.
                **Reason:** {str(e)}
                Please ensure your OPENAI_API_KEY environment variable is correctly set and try again.
                """
        
        # 5. Generate trading signals
        elapsed = (datetime.now() - start_time).total_seconds()
        if elapsed < 28:  # Still have time
            try:
                signals = generate_trading_signals(data_with_indicators, risk_appetite)
                response["trading_signals"] = signals
            except Exception as e:
                print(f"Error generating trading signals: {e}")
        
        # 6. Generate sentiment analysis if time permits
        elapsed = (datetime.now() - start_time).total_seconds()
        if elapsed < 30:  # Still have time
            try:
                sentiment = analyze_news_sentiment(symbol)
                if sentiment:
                    response["sentiment_analysis"] = sentiment
            except Exception as e:
                print(f"Error analyzing sentiment: {e}")
        
        # Remove temporary date column if it exists
        if 'date' in data.columns:
            data.drop('date', axis=1, inplace=True)
        
        # Return the response with whatever we managed to calculate
        print(f"Total processing time: {(datetime.now() - start_time).total_seconds():.2f}s")
        return jsonify(response)
    
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({
            "error": str(e),
            "openai_refined_prediction": f"Error analyzing {symbol}: {str(e)}. Please try again later."
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
