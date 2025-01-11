import base64
import io
from keras.models import model_from_json
from statsmodels.tsa.arima.model import ARIMA
import joblib
from sklearn.preprocessing import StandardScaler

# Base64-encoded LSTM model JSON and weights
LSTM_MODEL_JSON = """<YOUR_BASE64_ENCODED_MODEL_JSON>"""
LSTM_MODEL_WEIGHTS = """<YOUR_BASE64_ENCODED_MODEL_WEIGHTS>"""

# Base64-encoded ARIMA model
ARIMA_MODEL = """<YOUR_BASE64_ENCODED_ARIMA_MODEL>"""

def load_lstm_model():
    """Load the LSTM model from embedded JSON and weights."""
    # Decode the JSON
    model_json = base64.b64decode(LSTM_MODEL_JSON).decode("utf-8")
    model = model_from_json(model_json)

    # Decode and load weights
    weights = base64.b64decode(LSTM_MODEL_WEIGHTS)
    with io.BytesIO(weights) as f:
        model.load_weights(f)

    return model

def load_arima_model():
    """Load the ARIMA model from embedded base64."""
    # Decode the model
    model_data = base64.b64decode(ARIMA_MODEL)
    with io.BytesIO(model_data) as f:
        return joblib.load(f)

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
