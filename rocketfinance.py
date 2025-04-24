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
from sklearn.preprocessing import StandardScaler
import random
import re
import threading
import io
import base64
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from enum import Enum

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
def fetch_data(symbol, timeframe):
    """
    Fetch stock data for a symbol from Alpha Vantage.
    Modified to ensure OHLC data for all timeframes including intraday.
    """
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise ValueError("Alpha Vantage API key not set in environment variable ALPHAVANTAGE_API_KEY")
    
    # Check cache first
    cache_key = f"{symbol.upper()}:{timeframe}"
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
        
        # Store in cache
        cache[cache_key] = (datetime.now(), df)
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
# Enhanced Forecasting System
# ---------------------------
def enhanced_forecast(data, periods=5, timeframe="1day"):
    """
    Generate enhanced forecasts with more realistic price movements showing ups and downs.
    This combines trend forecasting with historical volatility patterns.
    """
    try:
        # Extract the closing prices
        close_prices = data["Close"].values
        # Determine if this is intraday or daily data
        is_intraday = timeframe.endswith('min') or timeframe.endswith('h')
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
# Machine Learning Features and Models
# ---------------------------
def create_features(df, target_col='Close', window=10):
    """Create features for machine learning models."""
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
# Advanced Ensemble Forecast
# ---------------------------
def ml_ensemble_forecast(data, periods=5, timeframe="1day"):
    """
    Create an ensemble forecast using multiple machine learning models.
    """
    # Only use ML ensemble if we have enough data
    if len(data) < 30:
        return enhanced_forecast(data, periods, timeframe)
    try:
        # Prepare features from technical indicators
        df = calculate_technical_indicators(data)
        df = df.dropna(subset=['Close'])
        # Create features from lagged prices and indicators
        features = create_features(df)
        # Train different models
        models = {
            "linear": train_linear_model(features, df['Close']),
            "ridge": train_ridge_model(features, df['Close']),
            "gradient_boost": train_gb_model(features, df['Close']),
            "svm": train_svm_model(features, df['Close'])
        }
        # Generate predictions from each model
        predictions = {}
        for name, model in models.items():
            if model is not None:  # Only use models that were successfully trained
                predictions[name] = generate_model_predictions(model, features, df, periods)
        # Generate ARIMA and enhanced forecasts as well
        if timeframe.endswith('min') or timeframe.endswith('h'):
            predictions["poly_reg"] = linear_regression_forecast(data, periods, degree=2)
        else:
            try:
                arima_model = create_arima_model(data)
                predictions["arima"] = arima_prediction(arima_model)
            except:
                predictions["poly_reg"] = linear_regression_forecast(data, periods, degree=1)
        predictions["enhanced"] = enhanced_forecast(data, periods, timeframe)
        # Ensemble the predictions (weighted average)
        weights = {
            "linear": 0.1,
            "ridge": 0.1,
            "gradient_boost": 0.2,
            "svm": 0.1,
            "poly_reg" if timeframe.endswith('min') or timeframe.endswith('h') else "arima": 0.2,
            "enhanced": 0.3
        }
        # Calculate weighted ensemble
        ensemble_forecast = []
        for i in range(periods):
            weighted_sum = 0
            weight_total = 0
            for model_name, prediction in predictions.items():
                if model_name in weights and i < len(prediction):
                    weighted_sum += prediction[i] * weights.get(model_name, 0)
                    weight_total += weights.get(model_name, 0)
            # Add some realistic variation
            ensemble_val = weighted_sum / weight_total if weight_total > 0 else data["Close"].iloc[-1]
            ensemble_forecast.append(float(ensemble_val))
        # Add some realistic variation like in enhanced_forecast
        last_price = data["Close"].iloc[-1]
        returns = np.diff(data["Close"].values) / data["Close"].values[:-1]
        recent_volatility = np.std(returns[-min(30, len(returns)):])
        # Add controlled randomness to forecast points
        for i in range(1, len(ensemble_forecast)):
            # Add market noise based on historical volatility
            random_component = ensemble_forecast[i-1] * recent_volatility * np.random.normal(0, 0.7)
            ensemble_forecast[i] = ensemble_forecast[i] + random_component
        print(f"Ensemble ML forecast: {ensemble_forecast}")
        return ensemble_forecast
    except Exception as e:
        print(f"Error in ML ensemble forecast: {e}")
        return enhanced_forecast(data, periods, timeframe)
    
# ---------------------------
# News Fetching Function
# ---------------------------
def fetch_news(symbol, max_items=5):
    """
    Fetch actual news articles for the symbol from a news API.
    Args:
        symbol: The stock symbol to fetch news for
        max_items: Maximum number of news items to return
    Returns:
        A list of news articles, each with title, source, and summary
    """
    try:
        # Get API key from environment variables
        news_api_key = os.getenv("NEWSAPI_KEY")
        if not news_api_key:
            print("Warning: NewsAPI key not set, using placeholder news")
            return get_placeholder_news(symbol)
        # Prepare the query - search for both the symbol and company name if possible
        company_names = {
            "AAPL": "Apple",
            "MSFT": "Microsoft",
            "GOOGL": "Google",
            "AMZN": "Amazon",
            "META": "Meta Facebook",
            "TSLA": "Tesla",
            "NVDA": "NVIDIA",
            "JPM": "JPMorgan",
            "BAC": "Bank of America",
            "WMT": "Walmart",
            "DIS": "Disney",
            "NFLX": "Netflix",
            "XOM": "Exxon",
            "CVX": "Chevron",
            "PFE": "Pfizer",
            "JNJ": "Johnson & Johnson"
            # Add more common symbols and company names as needed
        }
        # Construct query using both symbol and company name if available
        query = symbol
        if symbol.upper() in company_names:
            query = f"{symbol} OR {company_names[symbol.upper()]}"
        # Build API request
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "apiKey": news_api_key,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": max_items
        }
        # Set timeout to avoid blocking
        timeout = 5  # seconds
        response = requests.get(url, params=params, timeout=timeout)
        if response.status_code != 200:
            print(f"NewsAPI error: Status {response.status_code}")
            return get_placeholder_news(symbol)
        data = response.json()
        if data.get("status") != "ok" or "articles" not in data:
            print(f"NewsAPI error: {data.get('message', 'Unknown error')}")
            return get_placeholder_news(symbol)
        articles = data["articles"]
        # Format the response
        news = []
        for article in articles:
            # Summarize the content if it's too long
            content = article.get("content", article.get("description", ""))
            if content and len(content) > 300:
                summary = content[:297] + "..."
            else:
                summary = content
            news.append({
                "title": article.get("title", ""),
                "source": {"name": article.get("source", {}).get("name", "Unknown Source")},
                "summary": summary,
                "url": article.get("url", ""),
                "publishedAt": article.get("publishedAt", "")
            })
        return news
    except Exception as e:
        print(f"Error fetching news: {e}")
        return get_placeholder_news(symbol)

def get_placeholder_news(symbol):
    """Return placeholder news when the API is unavailable."""
    current_date = datetime.now().strftime("%Y-%m-%d")
    news = [
        {
            "title": f"{symbol.upper()} stock shows market volatility",
            "source": {"name": "Market Insight"},
            "summary": f"Recent trading of {symbol.upper()} demonstrates ongoing market volatility as investors respond to broader economic indicators and company-specific developments.",
            "publishedAt": current_date
        },
        {
            "title": f"Analysts issue updated guidance on {symbol.upper()}",
            "source": {"name": "Financial Observer"},
            "summary": f"Investment analysts have issued new price targets for {symbol.upper()}, reflecting revised expectations based on recent performance and forward outlook.",
            "publishedAt": current_date
        },
        {
            "title": f"{symbol.upper()} in focus as market evaluates sector trends",
            "source": {"name": "Trading View"},
            "summary": f"Investors are closely watching {symbol.upper()} as a possible indicator of broader sector performance. Technical analysis suggests watching key support and resistance levels.",
            "publishedAt": current_date
        }
    ]
    return news

# ---------------------------
# Enhanced News Sentiment Analysis
# ---------------------------
class EnhancedNewsSentimentAnalyzer:
    """Enhanced news sentiment analysis using multiple NLP models."""
    
    def __init__(self):
        """Initialize the sentiment analyzer."""
        # Ensure NLTK resources are downloaded
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
        
        self.vader = SentimentIntensityAnalyzer()
        # Financial-specific words and their sentiment scores
        self.financial_lexicon = {
            'beat': 3.0, 'exceeded': 3.0, 'surpassed': 3.0, 'outperform': 2.5,
            'upgrade': 2.0, 'upgraded': 2.0, 'buy': 1.5, 'bullish': 2.0,
            'miss': -3.0, 'missed': -3.0, 'disappointing': -2.5, 'underperform': -2.5,
            'downgrade': -2.0, 'downgraded': -2.0, 'sell': -1.5, 'bearish': -2.0,
            'investigation': -1.5, 'lawsuit': -1.5, 'sec': -1.0, 'fine': -1.0,
            'earnings': 0.0  # Neutral unless qualified
        }
        # Add words to the VADER lexicon
        for word, score in self.financial_lexicon.items():
            self.vader.lexicon[word] = score
    
    def analyze_text(self, text):
        """Analyze the sentiment of a piece of text."""
        # Clean the text
        text = re.sub(r'[^\w\s]', '', text)
        
        # Use multiple sentiment analysis techniques and average them
        vader_scores = self.vader.polarity_scores(text)
        textblob_analysis = TextBlob(text)
        
        # Combine scores (give more weight to VADER as it's finance-tuned)
        compound_sentiment = vader_scores['compound'] * 0.7 + textblob_analysis.sentiment.polarity * 0.3
        
        # Convert to -1 to 1 scale
        normalized_score = max(-1.0, min(1.0, compound_sentiment))
        
        # Determine sentiment category
        if normalized_score > 0.25:
            category = "positive"
        elif normalized_score < -0.25:
            category = "negative"
        else:
            category = "neutral"
            
        return {
            "score": normalized_score,
            "category": category,
            "vader_score": vader_scores['compound'],
            "textblob_score": textblob_analysis.sentiment.polarity
        }
    
    def analyze_news_batch(self, news_items):
        """Analyze a batch of news items and return detailed sentiment analysis."""
        results = []
        
        for item in news_items:
            # Combine title and summary for analysis
            title = item.get('title', '')
            summary = item.get('summary', '')
            full_text = f"{title}. {summary}"
            
            # Get sentiment
            sentiment = self.analyze_text(full_text)
            
            # Add additional context
            result = {
                "title": title,
                "source": item.get('source', {}).get('name', 'Unknown'),
                "published_at": item.get('publishedAt', None),
                "sentiment_score": sentiment["score"],
                "sentiment_category": sentiment["category"],
                "url": item.get('url', '')
            }
            results.append(result)
            
        return results
    
    def get_overall_sentiment(self, analyzed_items):
        """Calculate overall sentiment from a list of analyzed news items."""
        if not analyzed_items:
            return {"score": 0, "category": "neutral", "confidence": 0}
            
        # Weight more recent news higher and consider source credibility
        total_score = 0
        weights = 0
        
        current_time = datetime.now()
        
        for item in analyzed_items:
            # Base weight
            weight = 1.0
            
            # Adjust weight based on recency if publish date is available
            if item.get('published_at'):
                try:
                    pub_date = datetime.fromisoformat(item['published_at'].replace('Z', '+00:00'))
                    hours_ago = (current_time - pub_date).total_seconds() / 3600
                    # Exponential decay - news from 24h ago has half the weight
                    recency_factor = 2 ** (-hours_ago / 24)
                    weight *= recency_factor
                except (ValueError, TypeError):
                    pass
            
            # Adjust weight based on source credibility
            if item.get('source'):
                credible_sources = ['Bloomberg', 'Reuters', 'Wall Street Journal', 'Financial Times', 'CNBC']
                if any(src.lower() in item['source'].lower() for src in credible_sources):
                    weight *= 1.5
            
            # Add weighted score
            total_score += item['sentiment_score'] * weight
            weights += weight
        
        if weights > 0:
            avg_score = total_score / weights
        else:
            avg_score = 0
            
        # Determine category and confidence
        if avg_score > 0.25:
            category = "positive"
        elif avg_score < -0.25:
            category = "negative"
        else:
            category = "neutral"
            
        # Calculate confidence based on agreement between sources
        scores = [item['sentiment_score'] for item in analyzed_items]
        if len(scores) > 1:
            # Standard deviation as a measure of disagreement
            std_dev = pd.Series(scores).std()
            # Higher standard deviation = lower confidence
            confidence = max(0, min(100, 100 * (1 - std_dev)))
        else:
            confidence = 50  # Default confidence for single source
            
        return {
            "score": avg_score,
            "category": category,
            "confidence": confidence,
            "sources_count": len(analyzed_items)
        }

def analyze_news_sentiment(symbol):
    """Analyze news sentiment for a given symbol with advanced analysis."""
    try:
        # Fetch news
        news = fetch_news(symbol, max_items=5)
        if not news:
            return {"score": 0, "category": "neutral", "confidence": 0}
        
        # Create analyzer instance
        analyzer = EnhancedNewsSentimentAnalyzer()
        
        # Analyze all news items
        analyzed_news = analyzer.analyze_news_batch(news)
        
        # Get overall sentiment
        overall_sentiment = analyzer.get_overall_sentiment(analyzed_news)
        
        # Add detailed items for reference
        overall_sentiment["items"] = analyzed_news
        
        return overall_sentiment
    except Exception as e:
        print(f"Error analyzing news sentiment: {e}")
        return {"score": 0, "category": "neutral", "confidence": 0}

# ---------------------------
# Social Media Sentiment Analysis (Placeholder)
# ---------------------------
def get_social_media_sentiment(symbol):
    """
    Get social media sentiment for a symbol.
    This is a placeholder function that returns simulated data.
    """
    # In a real implementation, you'd query Twitter, Reddit, StockTwits APIs
    sentiment_score = random.uniform(-0.6, 0.6)
    
    # Add some bias based on the symbol (for demo purposes)
    popular_symbols = {
        "AAPL": 0.2, "MSFT": 0.15, "TSLA": 0.3, "AMZN": 0.1, "NFLX": -0.1,
        "FB": -0.15, "GOOGL": 0.1, "NVDA": 0.25
    }
    
    if symbol.upper() in popular_symbols:
        sentiment_score += popular_symbols[symbol.upper()]
    
    # Ensure score is within -1 to 1 range
    sentiment_score = max(-1.0, min(1.0, sentiment_score))
    
    # Determine category
    if sentiment_score > 0.2:
        category = "positive"
    elif sentiment_score < -0.2:
        category = "negative"
    else:
        category = "neutral"
    
    return {
        "score": sentiment_score,
        "category": category,
        "sources": ["Twitter", "Reddit", "StockTwits"],
        "volume": random.randint(100, 5000),  # Simulated mention volume
        "trending": abs(sentiment_score) > 0.4  # Is it trending?
    }

# ---------------------------
# Sentiment Trend Analysis
# ---------------------------
def analyze_sentiment_trends(symbol, price_data, days=30):
    """
    Analyze sentiment trends and correlate with price action.
    This is a placeholder that returns simulated data.
    
    Args:
        symbol: Stock symbol
        price_data: DataFrame with price history
        days: Number of days to analyze
    """
    # Generate some simulated sentiment data with trends
    np.random.seed(42)  # For reproducibility
    
    # Create a date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create a base trend
    t = np.linspace(0, 1, len(date_range))
    base_trend = 0.2 * np.sin(2 * np.pi * t * 2)  # Sinusoidal trend
    
    # Add some noise
    noise = np.random.normal(0, 0.15, len(date_range))
    
    # Combine trend and noise
    news_sentiment = np.clip(base_trend + noise, -1, 1)
    
    # Social sentiment follows news with a slight delay and more volatility
    social_noise = np.random.normal(0, 0.25, len(date_range))
    social_sentiment = np.clip(np.roll(base_trend, 2) + social_noise, -1, 1)
    
    # Calculate current values
    current_news = news_sentiment[-1]
    current_social = social_sentiment[-1]
    
    # Simulate trend
    recent_news_trend = news_sentiment[-5:].mean() - news_sentiment[-10:-5].mean()
    if recent_news_trend > 0.1:
        trend = "strongly_improving"
    elif recent_news_trend > 0.05:
        trend = "improving"
    elif recent_news_trend < -0.1:
        trend = "strongly_deteriorating"
    elif recent_news_trend < -0.05:
        trend = "deteriorating"
    else:
        trend = "stable"
    
    # Create a sample analysis
    if current_news > 0.3 and current_social > 0.3:
        analysis = "Both news and social sentiment are strongly positive, suggesting bullish market perception."
    elif current_news < -0.3 and current_social < -0.3:
        analysis = "Both news and social sentiment are strongly negative, suggesting bearish market perception."
    elif current_news * current_social < 0:  # Different signs
        analysis = "There's a divergence between news and social sentiment, which might indicate a shifting market perception."
    else:
        analysis = "Sentiment is relatively neutral with no strong directional bias."
    
    if trend == "strongly_improving" or trend == "improving":
        analysis += " Sentiment has been improving recently, which could support further price increases."
    elif trend == "strongly_deteriorating" or trend == "deteriorating":
        analysis += " Sentiment has been deteriorating recently, which could lead to price weakness."
    
    return {
        "news_sentiment_current": float(current_news),
        "social_sentiment_current": float(current_social),
        "sentiment_trend": trend,
        "analysis": analysis
    }

# ---------------------------
# Automated Trading Signals
# ---------------------------
class SignalType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

class SignalStrength(Enum):
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"

class SignalGenerator:
    """Generate trading signals based on technical indicators and price action."""
    
    def __init__(self, risk_appetite="moderate"):
        """Initialize with risk appetite: conservative, moderate, or aggressive."""
        self.risk_appetite = risk_appetite
        
    def generate_signals(self, data, include_indicators=True):
        """Generate trading signals from the provided data."""
        # Ensure we have indicators
        if include_indicators:
            data = calculate_technical_indicators(data)
            
        # Generate signals from various methods
        trend_signal = self._generate_trend_signal(data)
        momentum_signal = self._generate_momentum_signal(data)
        reversal_signal = self._generate_reversal_signal(data)
        breakout_signal = self._generate_breakout_signal(data)
        volume_signal = self._generate_volume_signal(data)
        
        # Weight based on risk appetite
        if self.risk_appetite == "conservative":
            weights = {"trend": 0.4, "momentum": 0.2, "reversal": 0.1, "breakout": 0.1, "volume": 0.2}
        elif self.risk_appetite == "aggressive":
            weights = {"trend": 0.2, "momentum": 0.3, "reversal": 0.2, "breakout": 0.2, "volume": 0.1}
        else:  # moderate
            weights = {"trend": 0.3, "momentum": 0.25, "reversal": 0.15, "breakout": 0.15, "volume": 0.15}
            
        # Store individual signals
        signals = {
            "components": {
                "trend": trend_signal,
                "momentum": momentum_signal,
                "reversal": reversal_signal,
                "breakout": breakout_signal,
                "volume": volume_signal
            }
        }
        
        # Calculate signal score (-100 to +100)
        signal_score = 0
        for name, signal in signals["components"].items():
            signal_value = 0
            if signal["type"] == SignalType.BUY.value:
                signal_value = 100 if signal["strength"] == SignalStrength.STRONG.value else 60 if signal["strength"] == SignalStrength.MODERATE.value else 30
            elif signal["type"] == SignalType.SELL.value:
                signal_value = -100 if signal["strength"] == SignalStrength.STRONG.value else -60 if signal["strength"] == SignalStrength.MODERATE.value else -30
            signal_score += signal_value * weights[name]
            
        # Determine overall signal
        if signal_score > 50:
            overall_type = SignalType.BUY.value
            overall_strength = SignalStrength.STRONG.value if signal_score > 75 else SignalStrength.MODERATE.value
        elif signal_score < -50:
            overall_type = SignalType.SELL.value
            overall_strength = SignalStrength.STRONG.value if signal_score < -75 else SignalStrength.MODERATE.value
        else:
            overall_type = SignalType.HOLD.value
            overall_strength = SignalStrength.MODERATE.value if abs(signal_score) > 25 else SignalStrength.WEAK.value
                
        signals["overall"] = {
            "type": overall_type,
            "strength": overall_strength,
            "score": signal_score,
            "risk_appetite": self.risk_appetite
        }
        
        # Generate risk management recommendations
        if overall_type == SignalType.BUY.value:
            signals["risk_management"] = self._calculate_buy_risk_management(data)
        elif overall_type == SignalType.SELL.value:
            signals["risk_management"] = self._calculate_sell_risk_management(data)
        
        return signals
    
    # Helper method implementation stubs (these would be fully implemented)
    def _generate_trend_signal(self, data): pass
    def _generate_momentum_signal(self, data): pass
    def _generate_reversal_signal(self, data): pass
    def _generate_breakout_signal(self, data): pass
    def _generate_volume_signal(self, data): pass
    
    def _calculate_buy_risk_management(self, data):
        """Calculate risk management for buy signals."""
        current_price = data['Close'].iloc[-1]
        atr = data.get('ATR', pd.Series()).iloc[-1] or current_price * 0.02
        
        # Simple implementation - in full code would use support/resistance
        stop_loss = max(current_price - atr * 3, data['Low'].tail(10).min() * 0.98)
        take_profit_1 = current_price + atr * 2
        take_profit_2 = current_price + atr * 4
        
        return {
            "entry": float(current_price),
            "stop_loss": float(stop_loss),
            "take_profit_1": float(take_profit_1),
            "take_profit_2": float(take_profit_2),
            "risk_reward": float((take_profit_1 - current_price) / (current_price - stop_loss)) if current_price > stop_loss else 0,
            "position_size": f"{2}% account risk"  # Simplified
        }
    
    def _calculate_sell_risk_management(self, data):
        """Calculate risk management for sell signals."""
        # Similar to buy but reversed (implementation omitted for brevity)
        return {}

# Function to generate trading signals
def generate_trading_signals(data, risk_appetite="moderate"):
    """Generate trading signals for the given data."""
    generator = SignalGenerator(risk_appetite)
    return generator.generate_signals(data)

# ---------------------------
# Generate OHLC data for forecast points
# ---------------------------
def generate_forecast_ohlc(data, forecast):
    """
    Generate OHLC values for forecast points with more realistic patterns.
    """
    # Get average daily volatility metrics from historical data
    avg_range = (data["High"] - data["Low"]).mean()
    avg_body_size = abs(data["Open"] - data["Close"]).mean()
    # Calculate the direction of each forecast day (up or down)
    forecast_ohlc = []
    last_close = data["Close"].iloc[-1]
    for i, close in enumerate(forecast):
        if i == 0:
            prev_close = last_close
        else:
            prev_close = forecast[i-1]
        direction = 1 if close > prev_close else -1
        # Calculate volatility factor - more movement for later forecast days
        volatility_factor = 1.0 + (i * 0.1)  # Increases volatility for later days
        # Calculate open based on previous close with some random variation
        # More realistic than always starting at previous close
        random_open_factor = np.random.normal(0, 0.3)
        open_deviation = avg_body_size * random_open_factor * volatility_factor
        open_price = prev_close + open_deviation
        # Determine high and low based on average range, direction, and volatility
        day_range = avg_range * volatility_factor * (0.8 + np.random.random() * 0.4)
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
    - If OHLC columns are present (daily data), create a dark-themed candlestick chart.
    - For intraday data (or if OHLC is missing), fall back to a simple dark line chart.
    """
    try:
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
                # Fall back to line chart
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
    except Exception as e:
        print(f"Error generating chart: {e}")
        # Return a default filename if chart generation fails
        return "chart_error.png"
    
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
    print(f"Received request for symbol: {symbol} with timeframe: {timeframe}")
    try:
        # Use a timer to track execution time
        start_time = datetime.now()
        # Fetch data first - this is the most critical part
        data = fetch_data(symbol, timeframe)
        # Generate enhanced forecast - this is also essential
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
        # Start with a basic response that will work even if other parts timeout
        response = {
            "forecast": forecast,
            "chartData": {"symbol": symbol.upper(), **chart_data},
            "news": [{"title": "Loading news...", "source": {"name": "Trading System"}, "summary": "News will be available on next refresh."}]
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
        if elapsed < 20:  # Still have time
            try:
                regime = detect_market_regime(data)
                response["market_regime"] = regime
                print(f"Detected market regime: {regime}")
            except Exception as e:
                print(f"Error detecting market regime: {e}")
                regime = "unknown"
        else:
            regime = "unknown"
        # 4. Trading signals
        elapsed = (datetime.now() - start_time).total_seconds()
        if elapsed < 22:  # Still have time
            try:
                signals = generate_trading_signals(data_with_indicators, risk_appetite)
                response["trading_signals"] = signals
            except Exception as e:
                print(f"Error generating trading signals: {e}")
        # 5. Enhanced sentiment analysis
        elapsed = (datetime.now() - start_time).total_seconds()
        if elapsed < 25:  # Still have time
            try:
                sentiment = analyze_news_sentiment(symbol)
                if sentiment:
                    response["sentiment_analysis"] = sentiment
            except Exception as e:
                print(f"Error analyzing sentiment: {e}")
        # Return the response with whatever we managed to calculate
        print(f"Total processing time: {(datetime.now() - start_time).total_seconds():.2f}s")
        return jsonify(response)
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

# Helper to extract key indicators for response
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
        return indicators
    except Exception as e:
        print(f"Error extracting indicators: {e}")
        return {}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
