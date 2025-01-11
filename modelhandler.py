import yfinance as yf
import joblib
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Input
from statsmodels.tsa.arima.model import ARIMA

# Function to fetch data
def fetch_data(symbol, period="1y", interval="1d"):
    data = yf.download(symbol, period=period, interval=interval)
    if data.empty:
        raise ValueError(f"No data found for symbol: {symbol}")
    return data

# Train and save LSTM model
def train_lstm(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    model = Sequential([
        Input(shape=(scaled_data.shape[1], 1)),
        LSTM(units=50, return_sequences=True),
        LSTM(units=50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(scaled_data, scaled_data, epochs=10, batch_size=32, verbose=1)

    model.save("lstm_model.h5")
    print("LSTM model saved as 'lstm_model.h5'")

# Train and save ARIMA model
def train_arima(data):
    data = data.asfreq("D")
    model = ARIMA(data['Close'], order=(5, 1, 0))
    model_fit = model.fit()
    joblib.dump(model_fit, "arima_model.pkl")
    print("ARIMA model saved as 'arima_model.pkl'")

# Main process
if __name__ == "__main__":
    symbol = "AAPL"  # Replace with the desired stock symbol
    data = fetch_data(symbol)
    train_lstm(data)
    train_arima(data)
