import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost as xgb
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load the data
df = pd.read_csv('AAPL.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# RSI calculation
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# MACD calculation
def calculate_macd(prices, fast_period=12, slow_period=26):
    fast_ema = prices.ewm(span=fast_period, adjust=False).mean()
    slow_ema = prices.ewm(span=slow_period, adjust=False).mean()
    return fast_ema - slow_ema

# Function to add features
def add_features(df):
    df['MA10'] = df['Adj Close Price'].rolling(window=10).mean()
    df['MA30'] = df['Adj Close Price'].rolling(window=30).mean()
    df['RSI'] = calculate_rsi(df['Adj Close Price'])
    df['MACD'] = calculate_macd(df['Adj Close Price'])
    return df

# Add features
apple = add_features(df)

# Create lags
for i in range(1, 4):
    apple[f'Lag_{i}'] = apple['Adj Close Price'].shift(i)

apple = apple.dropna()

# Define features and target
features = ['Open Price', 'High Price', 'Low Price', 'Close Price', 'Volume', 'MA10', 'MA30', 'RSI', 'MACD', 'Lag_1', 'Lag_2', 'Lag_3']
target = 'Adj Close Price'

X = apple[features]
y = apple[target]

# Feature selection
selector = SelectKBest(score_func=f_regression, k=8)
X_selected = selector.fit_transform(X, y)
selected_features = [features[i] for i in selector.get_support(indices=True)]

# Split the data
train_size = int(len(X_selected) * 0.7)
X_train, X_test = X_selected[:train_size], X_selected[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Scale the features
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

# XGBoost model with increased regularization
xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.01,
    max_depth=3,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=0.5,
    reg_lambda=1.5
)

# Implement time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)
xgb_scores = []

for train_index, val_index in tscv.split(X_train_scaled):
    X_train_cv, X_val_cv = X_train_scaled[train_index], X_train_scaled[val_index]
    y_train_cv, y_val_cv = y_train_scaled[train_index], y_train_scaled[val_index]
    
    xgb_model.fit(X_train_cv, y_train_cv)
    xgb_pred_cv = xgb_model.predict(X_val_cv)
    xgb_scores.append(mean_squared_error(y_val_cv, xgb_pred_cv))

print(f"XGBoost CV MSE: {np.mean(xgb_scores):.4f} (+/- {np.std(xgb_scores):.4f})")

# Fit XGBoost on the entire training set
xgb_model.fit(X_train_scaled, y_train_scaled)

# LSTM model with reduced complexity and dropout
lstm_model = Sequential([
    LSTM(units=32, activation='relu', input_shape=(1, X_train_scaled.shape[1]), return_sequences=True),
    Dropout(0.2),
    LSTM(units=16, activation='relu'),
    Dropout(0.2),
    Dense(units=1)
])
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Implement early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

lstm_model.fit(
    X_train_lstm, y_train_scaled,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=0
)

# Make predictions
xgb_pred = scaler_y.inverse_transform(xgb_model.predict(X_test_scaled).reshape(-1, 1))
lstm_pred = scaler_y.inverse_transform(lstm_model.predict(X_test_lstm))

# Ensemble predictions with different weights
ensemble_weights = [0.5, 0.5]  # Equal weights for XGBoost and LSTM
ensemble_pred = ensemble_weights[0] * xgb_pred + ensemble_weights[1] * lstm_pred

# Evaluate models
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} - MSE: {mse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")

evaluate_model(y_test, xgb_pred, "XGBoost")
evaluate_model(y_test, lstm_pred, "LSTM")
evaluate_model(y_test, ensemble_pred, "Weighted Ensemble")

# Generate future dates
last_date = apple.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='B')

# Create future features dynamically
def create_future_features(last_known_data, future_dates, historical_volatility):
    future_data = pd.DataFrame(index=future_dates)
    for i, date in enumerate(future_dates):
        if i == 0:
            last_price = last_known_data['Adj Close Price']
        else:
            last_price = future_data.loc[future_dates[i-1], 'Adj Close Price']
        
        # Add some randomness based on historical volatility
        price_change = np.random.normal(0, historical_volatility)
        new_price = last_price * (1 + price_change)
        
        future_data.loc[date, 'Adj Close Price'] = new_price
        future_data.loc[date, 'Open Price'] = new_price * (1 + np.random.normal(0, historical_volatility/2))
        future_data.loc[date, 'High Price'] = new_price * (1 + abs(np.random.normal(0, historical_volatility)))
        future_data.loc[date, 'Low Price'] = new_price * (1 - abs(np.random.normal(0, historical_volatility)))
        future_data.loc[date, 'Close Price'] = new_price * (1 + np.random.normal(0, historical_volatility/2))
        future_data.loc[date, 'Volume'] = last_known_data['Volume'] * (1 + np.random.normal(0, 0.1))
    
    future_data['MA10'] = future_data['Adj Close Price'].rolling(window=10).mean()
    future_data['MA30'] = future_data['Adj Close Price'].rolling(window=30).mean()
    future_data['RSI'] = calculate_rsi(future_data['Adj Close Price'])
    future_data['MACD'] = calculate_macd(future_data['Adj Close Price'])
    
    for i in range(1, 4):
        future_data[f'Lag_{i}'] = future_data['Adj Close Price'].shift(i)
    
    future_data = future_data.ffill().bfill()
    
    return future_data[selected_features]

# Calculate historical volatility
historical_volatility = apple['Adj Close Price'].pct_change().std()

last_known_data = apple.iloc[-1].to_dict()
future_features = create_future_features(last_known_data, future_dates, historical_volatility)

# Scale future features
future_features_scaled = scaler_X.transform(future_features)

# Make predictions for future dates
future_xgb_pred = scaler_y.inverse_transform(xgb_model.predict(future_features_scaled).reshape(-1, 1))
future_lstm_pred = scaler_y.inverse_transform(lstm_model.predict(future_features_scaled.reshape(-1, 1, future_features_scaled.shape[1])))

# Ensemble future predictions
future_ensemble_pred = ensemble_weights[0] * future_xgb_pred + ensemble_weights[1] * future_lstm_pred

# Apply exponential smoothing to future predictions
def exponential_smoothing(series, alpha):
    result = [series[0]]
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return np.array(result)

future_ensemble_pred_smoothed = exponential_smoothing(future_ensemble_pred.flatten(), alpha=0.3)

# Add some randomness to future predictions
future_ensemble_pred_final = future_ensemble_pred_smoothed + np.random.normal(0, historical_volatility, size=len(future_ensemble_pred_smoothed))

# Combine actual data, known predictions, and future predictions
historical_dates = apple.index[train_size:]
historical_predictions = pd.Series(ensemble_pred.flatten(), index=historical_dates)
future_predictions = pd.Series(future_ensemble_pred_final, index=future_dates)

# Plot results
plt.figure(figsize=(15, 8))
plt.plot(apple.index, apple['Adj Close Price'], label='Actual Price', alpha=0.7)
plt.plot(historical_dates, historical_predictions, label='Historical Prediction', alpha=0.7)
plt.plot(future_dates, future_predictions, label='Future Prediction', alpha=0.7, linestyle='--')

plt.title('Apple Stock Price Prediction (Including Future)')
plt.xlabel('Date')
plt.ylabel('Adj Close Price USD($)')
plt.legend()

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gcf().autofmt_xdate()

plt.tight_layout()
plt.show()

# Print future predictions
print("\nFuture Price Predictions:")
for date, price in zip(future_dates, future_predictions):
    print(f"{date.date()}: ${price:.2f}")