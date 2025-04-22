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
            "5. volume": "Volume" if "5. volume" in df.columns else None
        })
        # Remove None values from rename dictionary
        df = df.rename(columns={k: v for k, v in df.columns.items() if v is not None})
        
        for col in ["Open", "High", "Low", "Close"]:
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
            "4. close": "Close",
            "5. volume": "Volume" if "5. volume" in df.columns else None
        })
        # Remove None values from rename dictionary
        df = df.rename(columns={k: v for k, v in df.columns.items() if v is not None})
        
        for col in ["Open", "High", "Low", "Close"]:
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
        # Fall back to enhanced forecast
        return enhanced_forecast(data, periods)

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
        # Fall back to original forecasting methods
        if timeframe.endswith('min') or timeframe.endswith('h'):
            return linear_regression_forecast(data, periods, degree=2)
        else:
            try:
                arima_model = create_arima_model(data)
                return arima_prediction(arima_model)
            except:
                return linear_regression_forecast(data, periods, degree=1)
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
# News Sentiment Analysis
# ---------------------------
def analyze_news_sentiment(symbol):
    """Analyze news sentiment for a given symbol."""
    news = fetch_news(symbol, max_items=15)
    if not news:
        return 0  # Neutral sentiment
    
    # Use OpenAI to analyze sentiment of recent news
    try:
        news_texts = [f"Title: {item['title']}\nSummary: {item['summary']}" for item in news if item.get('title') and item.get('summary')]
        
        # Batch news into groups to avoid token limits
        batched_news = []
        current_batch = []
        for item in news_texts[:10]:  # Limit to 10 most recent articles
            current_batch.append(item)
            if len(current_batch) >= 3:
                batched_news.append(current_batch)
                current_batch = []
        if current_batch:
            batched_news.append(current_batch)
        
        # Process each batch
        sentiments = []
        for batch in batched_news:
            batch_text = "\n\n".join(batch)
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a financial news analyst. Analyze the sentiment of these news articles about a stock."},
                    {"role": "user", "content": f"Analyze these news articles about {symbol} and provide a sentiment score from -1.0 (extremely bearish) to +1.0 (extremely bullish):\n\n{batch_text}"}
                ]
            )
            
            # Extract sentiment score from the response
            response_text = response["choices"][0]["message"]["content"]
            
            # Try to find a numeric score in the response
            import re
            matches = re.search(r"([-+]?\d*\.\d+|\d+)", response_text)
            if matches:
                try:
                    sentiment = float(matches.group(0))
                    # Ensure sentiment is within -1 to 1 range
                    sentiment = max(-1.0, min(1.0, sentiment))
                    sentiments.append(sentiment)
                except ValueError:
                    pass
        
        # Average the sentiments
        if sentiments:
            avg_sentiment = sum(sentiments) / len(sentiments)
            print(f"News sentiment score for {symbol}: {avg_sentiment}")
            return avg_sentiment
        return 0
    except Exception as e:
        print(f"Error analyzing news sentiment: {e}")
        return 0

# ---------------------------
# Adaptive Forecasting
# ---------------------------
def adaptive_forecast(data, periods=5, timeframe="1day", symbol="AAPL"):
    """
    Adapt forecasting method based on detected market regime.
    """
    # Detect the current market regime
    regime = detect_market_regime(data)
    print(f"Detected market regime: {regime}")
    
    # Analyze news sentiment
    sentiment = analyze_news_sentiment(symbol)
    
    # Use appropriate forecasting method based on regime
    if regime == 'trending_up' or regime == 'trending_down':
        # Use trend-following model with strong trend component
        forecast = ml_ensemble_forecast(data, periods, timeframe)
        # Strengthen the trend direction
        trend_direction = 1 if regime == 'trending_up' else -1
        forecast = strengthen_trend(forecast, trend_direction, strength=0.02)
    elif regime == 'mean_reverting':
        # Use mean-reversion model
        forecast = mean_reversion_forecast(data, periods)
    elif regime == 'volatile':
        # Use high-volatility model
        forecast = ml_ensemble_forecast(data, periods, timeframe)
        # Increase volatility in the forecast
        forecast = adjust_forecast_volatility(forecast, data, multiplier=1.5)
    else:
        # Default to ensemble forecast
        forecast = ml_ensemble_forecast(data, periods, timeframe)
    
    # Apply sentiment adjustment
    forecast = adjust_forecast_with_sentiment(forecast, sentiment)
    
    return forecast

def strengthen_trend(forecast, trend_direction, strength=0.02):
    """Strengthen the trend direction in a forecast."""
    result = forecast.copy()
    for i in range(1, len(result)):
        # Add a small percentage increase/decrease in the trend direction
        result[i] = result[i] * (1 + (strength * i * trend_direction))
    return result

def adjust_forecast_volatility(forecast, data, multiplier=1.5):
    """Adjust the volatility of a forecast."""
    # Calculate historical volatility
    returns = np.diff(data["Close"].values) / data["Close"].values[:-1]
    volatility = np.std(returns[-min(30, len(returns)):])
    
    # Adjust forecast
    result = [forecast[0]]
    for i in range(1, len(forecast)):
        # Add randomness scaled by volatility
        random_component = forecast[i-1] * volatility * multiplier * np.random.normal(0, 1.0)
        result.append(forecast[i] + random_component)
    
    return result

def adjust_forecast_with_sentiment(forecast, sentiment, volatility_factor=1.0):
    """Adjust forecast based on market sentiment and conditions."""
    result = forecast.copy()
    
    # If sentiment is strongly positive, slightly increase later forecast points
    if sentiment > 0.5:
        for i in range(1, len(result)):
            adjustment = result[i] * (0.01 * (i+1) * sentiment)
            result[i] += adjustment
    # If sentiment is strongly negative, slightly decrease later forecast points
    elif sentiment < -0.5:
        for i in range(1, len(result)):
            adjustment = result[i] * (0.01 * (i+1) * abs(sentiment))
            result[i] -= adjustment
    
    # Adjust volatility based on overall market conditions
    for i in range(1, len(result)):
        current_diff = result[i] - result[i-1]
        # Scale the difference by volatility factor
        result[i] = result[i-1] + (current_diff * volatility_factor)
    
    return result
# ---------------------------
# Trade Recommendations
# ---------------------------
def identify_support_levels(data, window=20):
    """Identify key support levels."""
    # Find local minima
    window = min(window, len(data) // 4)
    if window < 2:
        return [data["Low"].min()]
        
    df = data.copy()
    df['min'] = df['Low'].rolling(window=window, center=True).min()
    local_mins = df[df['Low'] == df['min']]['Low'].tolist()
    
    # Group nearby levels
    if not local_mins:
        return [data["Low"].min()]
        
    levels = []
    for price in sorted(local_mins):
        # Check if this level is close to an existing one
        if not levels or min(abs(price - level) / level for level in levels) > 0.02:
            levels.append(price)
    
    # Sort by importance (frequency of touches)
    levels_with_importance = []
    for level in levels:
        # Count how many times price came within 1% of this level
        touches = sum(1 for low in data["Low"] if abs(low - level) / level < 0.01)
        levels_with_importance.append((level, touches))
    
    # Sort by number of touches, then by recency (prefer more recent levels)
    levels_with_importance.sort(key=lambda x: x[1], reverse=True)
    return [level for level, _ in levels_with_importance[:3]]

def identify_resistance_levels(data, window=20):
    """Identify key resistance levels."""
    # Find local maxima
    window = min(window, len(data) // 4)
    if window < 2:
        return [data["High"].max()]
        
    df = data.copy()
    df['max'] = df['High'].rolling(window=window, center=True).max()
    local_maxs = df[df['High'] == df['max']]['High'].tolist()
    
    # Group nearby levels
    if not local_maxs:
        return [data["High"].max()]
        
    levels = []
    for price in sorted(local_maxs):
        # Check if this level is close to an existing one
        if not levels or min(abs(price - level) / level for level in levels) > 0.02:
            levels.append(price)
    
    # Sort by importance (frequency of touches)
    levels_with_importance = []
    for level in levels:
        # Count how many times price came within 1% of this level
        touches = sum(1 for high in data["High"] if abs(high - level) / level < 0.01)
        levels_with_importance.append((level, touches))
    
    # Sort by number of touches, then by recency
    levels_with_importance.sort(key=lambda x: x[1], reverse=True)
    return [level for level, _ in levels_with_importance[:3]]

def calculate_risk_reward(current_price, forecast, support_levels, resistance_levels):
    """Calculate risk and reward based on support/resistance and forecast."""
    # Determine potential reward
    potential_high = max(forecast + [level for level in resistance_levels if level > current_price])
    reward = potential_high - current_price
    
    # Determine potential risk
    potential_low = min([level for level in support_levels if level < current_price] or [current_price * 0.95])
    risk = current_price - potential_low
    
    return max(risk, 0.001 * current_price), max(reward, 0.001 * current_price)

def calculate_forecast_confidence(data, forecast, timeframe):
    """Calculate confidence score for the forecast."""
    # Base confidence on historical volatility
    returns = data["Close"].pct_change().dropna()
    volatility = returns.std() * 100  # Convert to percentage
    
    # Determine timeframe factor
    if timeframe in ["5min", "30min", "2h", "4h"]:
        timeframe_factor = 0.7  # Lower confidence for intraday
    elif timeframe == "1day":
        timeframe_factor = 0.8
    elif timeframe in ["7day", "1mo"]:
        timeframe_factor = 0.85
    else:
        timeframe_factor = 0.9
    
    # Calculate trend strength
    trend_strength = 0.8
    if len(data) >= 20:
        sma_20 = data["Close"].rolling(window=20).mean().iloc[-1]
        sma_50 = data["Close"].rolling(window=min(50, len(data))).mean().iloc[-1]
        if sma_20 > sma_50:
            trend_direction = 1  # Uptrend
        else:
            trend_direction = -1  # Downtrend
        
        # Check if forecast follows trend
        forecast_direction = 1 if forecast[-1] > forecast[0] else -1
        if forecast_direction == trend_direction:
            trend_strength = 0.9
        else:
            trend_strength = 0.6
    
    # Base confidence calculation
    base_confidence = 90 - min(volatility * 5, 40)  # Lower confidence for higher volatility
    
    # Adjust by trend and timeframe factors
    confidence = base_confidence * trend_strength * timeframe_factor
    
    # Ensure confidence is within 0-100 range
    return min(max(confidence, 10), 100)

def recommend_position_size(data, confidence):
    """Recommend position size based on volatility and confidence."""
    # Calculate historical volatility
    returns = data["Close"].pct_change().dropna()
    volatility = returns.std() * 100  # as percentage
    
    # Base position size on volatility and confidence
    if volatility > 5:
        # High volatility
        base_size = "Small"
    elif volatility > 2:
        # Medium volatility
        base_size = "Medium"
    else:
        # Low volatility
        base_size = "Full"
    
    # Adjust based on confidence
    if confidence < 30:
        adjustment = "Very Small"
    elif confidence < 50:
        adjustment = "Small"
    elif confidence < 70:
        adjustment = "Medium"
    else:
        adjustment = "Full"
    
    # Combine base size and adjustment
    if base_size == "Small" or adjustment == "Very Small":
        return "Very Small (10-15% of normal position)"
    elif base_size == "Small" or adjustment == "Small":
        return "Small (25-33% of normal position)"
    elif base_size == "Medium" or adjustment == "Medium":
        return "Medium (50% of normal position)"
    else:
        return "Full Position"

def calculate_stop_loss(current_price, direction, support_levels):
    """Calculate stop loss level."""
    if direction == "bullish":
        # For bullish trades, set stop below nearest support
        support_below = [level for level in support_levels if level < current_price]
        if support_below:
            return max(support_below) * 0.99  # Just below support
        else:
            return current_price * 0.95  # Default 5% stop
    else:
        # For bearish trades, use a wider stop above nearest resistance
        return current_price * 1.05  # Default 5% stop

def calculate_take_profit(current_price, direction, resistance_levels, forecast):
    """Calculate take profit level."""
    if direction == "bullish":
        # For bullish trades, set target at or above resistance
        resistance_above = [level for level in resistance_levels if level > current_price]
        if resistance_above:
            return min(resistance_above) * 1.01  # Just above resistance
        else:
            # Use the highest forecast point
            max_forecast = max(forecast)
            if max_forecast > current_price:
                return max_forecast
            return current_price * 1.05  # Default 5% target
    else:
        # For bearish trades, set target below support
        return current_price * 0.95  # Default 5% target

def generate_recommendation_notes(direction, confidence, risk, reward, timeframe):
    """Generate recommendation notes."""
    if confidence > 70:
        confidence_text = "high confidence"
    elif confidence > 50:
        confidence_text = "moderate confidence"
    else:
        confidence_text = "speculative"
    
    risk_reward_ratio = reward / risk if risk > 0 else 0
    
    if risk_reward_ratio > 3:
        risk_reward_text = "very favorable"
    elif risk_reward_ratio > 2:
        risk_reward_text = "favorable"
    elif risk_reward_ratio > 1:
        risk_reward_text = "acceptable"
    else:
        risk_reward_text = "unfavorable"
    
    timeframe_text = "short-term" if timeframe in ["5min", "30min", "2h", "4h", "1day"] else "medium-term"
    
    return f"{confidence_text} {timeframe_text} {direction} trade with {risk_reward_text} risk/reward ratio"

def generate_trade_recommendations(data, forecast, symbol, timeframe):
    """
    Generate concrete trade recommendations based on forecasts and market analysis.
    """
    # Current price and forecast direction
    current_price = data['Close'].iloc[-1]
    forecast_direction = "bullish" if forecast[-1] > current_price else "bearish"
    
    # Calculate key levels
    support_levels = identify_support_levels(data)
    resistance_levels = identify_resistance_levels(data)
    
    # Determine risk/reward ratio
    risk, reward = calculate_risk_reward(current_price, forecast, support_levels, resistance_levels)
    
    # Calculate confidence score (0-100)
    confidence = calculate_forecast_confidence(data, forecast, timeframe)
    
    # Generate position size recommendation based on volatility and confidence
    position_size = recommend_position_size(data, confidence)
    
    # Generate stop loss and take profit levels
    stop_loss = calculate_stop_loss(current_price, forecast_direction, support_levels)
    take_profit = calculate_take_profit(current_price, forecast_direction, resistance_levels, forecast)
    
    recommendations = {
        "direction": forecast_direction,
        "confidence": confidence,
        "entry_price": current_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "risk_reward_ratio": reward / risk if risk > 0 else 0,
        "recommended_position_size": position_size,
        "key_support_levels": support_levels[:2],  # Top 2 support levels
        "key_resistance_levels": resistance_levels[:2],  # Top 2 resistance levels
        "timeframe": timeframe,
        "notes": generate_recommendation_notes(forecast_direction, confidence, risk, reward, timeframe)
    }
    
    return recommendations

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

def extract_key_indicators(data_with_indicators):
    """Extract key technical indicators for the response."""
    indicators = {}
    
    # Get the most recent values of key indicators
    for indicator in ['RSI', 'MACD', 'ATR', 'SMA_20', 'SMA_50', 'BB_Upper', 'BB_Lower']:
        if indicator in data_with_indicators.columns:
            indicators[indicator] = float(data_with_indicators[indicator].iloc[-1])
    
    # Calculate additional standard indicators
    if 'Close' in data_with_indicators.columns:
        last_close = data_with_indicators['Close'].iloc[-1]
        
        # Trend indicators
        if 'SMA_20' in indicators and 'SMA_50' in indicators:
            indicators['trend'] = 'bullish' if indicators['SMA_20'] > indicators['SMA_50'] else 'bearish'
        
        # Bollinger Band position
        if 'BB_Upper' in indicators and 'BB_Lower' in indicators:
            bb_position = (last_close - indicators['BB_Lower']) / (indicators['BB_Upper'] - indicators['BB_Lower'])
            indicators['BB_position'] = float(bb_position)
    
    return indicators
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
# OpenAI Analysis Functions
# ---------------------------
def refine_predictions_with_openai(symbol, regime, forecast, history, timeframe):
    """
    Call the OpenAI API to provide a detailed analysis of the stock that's timeframe-specific.
    Now includes market regime information.
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
    
    # Add market regime information
    regime_context = ""
    if regime == "trending_up":
        regime_context = "The stock is in an uptrend. Consider bullish strategies like buying on dips or breakouts."
    elif regime == "trending_down":
        regime_context = "The stock is in a downtrend. Consider bearish strategies or waiting for reversal signals."
    elif regime == "mean_reverting":
        regime_context = "The stock is showing mean-reverting behavior. Consider counter-trend strategies."
    elif regime == "volatile":
        regime_context = "The stock is showing high volatility. Consider volatility-based strategies and tighter risk management."
    
    prompt = f"""
    {time_context}
    
    Analyze the following stock data for {symbol.upper()} with a {timeframe} timeframe:
    
    Market Regime: {regime}
    {regime_context}
    
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
    # Added new parameter for number of news items
    news_count = int(request.args.get("news_count", "5"))
    risk_profile = request.args.get("risk_profile", "moderate")  # New parameter
    
    print(f"Received request for symbol: {symbol} with timeframe: {timeframe}")

    try:
        # Fetch data with additional fields if possible
        data = fetch_data(symbol, timeframe)
        
        # Add technical indicators
        data_with_indicators = calculate_technical_indicators(data)
        
        # Detect market regime
        regime = detect_market_regime(data)
        print(f"Detected market regime: {regime}")
        
        # Use adaptive forecasting based on market regime
        forecast = adaptive_forecast(data, periods=5, timeframe=timeframe, symbol=symbol)
        
        # Get news and analyze sentiment
        news = fetch_news(symbol, max_items=news_count)
        sentiment = analyze_news_sentiment(symbol)
        print(f"News sentiment: {sentiment}")
        
        # Generate OpenAI analysis with additional context
        refined_prediction = refine_predictions_with_openai(symbol, regime, forecast, data_with_indicators, timeframe)
        
        # Generate trade recommendations
        trade_recommendations = generate_trade_recommendations(data, forecast, symbol, timeframe)
        
        # Generate chart
        chart_filename = generate_chart(data, symbol, forecast=forecast, timeframe=timeframe)
        
        # Prepare chart data including indicators
        chart_data = get_chart_data(data, forecast, timeframe)
        
        # Add additional data to the response
        response = {
            "forecast": forecast,
            "openai_refined_prediction": refined_prediction,
            "chart_path": chart_filename,
            "chartData": {"symbol": symbol.upper(), **chart_data},
            "news": news,
            "market_regime": regime,
            "sentiment_score": sentiment,
            "trade_recommendations": trade_recommendations,
            "key_indicators": extract_key_indicators(data_with_indicators)
        }
        
        return jsonify(response)
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
