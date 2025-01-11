from keras.models import Sequential
from keras.layers import Dense, LSTM, Input
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler

def create_lstm_model():
    """Recreate the LSTM model programmatically."""
    model = Sequential([
        Input(shape=(10, 1)),  # Adjust the input shape based on your data
        LSTM(units=50, return_sequences=True),
        LSTM(units=50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    print("LSTM model recreated successfully.")
    return model

def create_arima_model(data):
    """Recreate and fit the ARIMA model."""
    try:
        data = data.asfreq("D").fillna(method="ffill")  # Fill missing values
        model = ARIMA(data['Close'], order=(0, 1, 0))  # Simplified order
        model_fit = model.fit()
        print("ARIMA model recreated and trained successfully.")
        return model_fit
    except Exception as e:
        print(f"Error in ARIMA model creation: {e}")
        raise

def lstm_prediction(model, data):
    """Make predictions using the LSTM model."""
    try:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
        prediction = model.predict(scaled_data[-1].reshape(1, 1, 1))
        return scaler.inverse_transform(prediction).flatten().tolist()
    except Exception as e:
        print(f"LSTM prediction error: {e}")
        return []

def arima_prediction(model):
    """Make predictions using the ARIMA model."""
    try:
        return model.forecast(steps=5).tolist()
    except Exception as e:
        print(f"ARIMA prediction error: {e}")
        return []
