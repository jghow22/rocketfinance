import os
from keras.models import load_model
import joblib
from sklearn.preprocessing import StandardScaler

LSTM_MODEL_PATH = "models/lstm_model.h5"  # Adjust if necessary
ARIMA_MODEL_PATH = "models/arima_model.pkl"

def load_lstm_model():
    """Load the pre-trained LSTM model."""
    if not os.path.exists(LSTM_MODEL_PATH):
        raise FileNotFoundError(f"LSTM model not found at {LSTM_MODEL_PATH}")
    return load_model(LSTM_MODEL_PATH)

def load_arima_model():
    """Load the pre-trained ARIMA model."""
    if not os.path.exists(ARIMA_MODEL_PATH):
        raise FileNotFoundError(f"ARIMA model not found at {ARIMA_MODEL_PATH}")
    return joblib.load(ARIMA_MODEL_PATH)

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
