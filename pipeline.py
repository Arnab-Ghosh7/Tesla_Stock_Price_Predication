import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import pickle

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_and_clean_data(file_path):
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    print(f"Dataset loaded. Shape: {df.shape}")
    
    # Check for missing values
    missing_vals = df.isnull().sum().sum()
    print(f"Total missing values: {missing_vals}")
    if missing_vals > 0:
        # Note: Stock price missing values are typically handled using forward-fill
        # or interpolation to preserve sequential time-series patterns.
        print("Handling missing values using forward-fill...")
        df = df.ffill()
        
    # Check for duplicates
    duplicates = df.duplicated().sum()
    print(f"Total duplicate rows: {duplicates}")
    if duplicates > 0:
        df = df.drop_duplicates()
        
    # Convert Date column to datetime and set it as index
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df = df.set_index('Date')
    return df

def prepare_sequences(data, lookback=60, forecast=10):
    """
    Creates sequences of lookback days to predict the next forecast days.
    """
    X, y = [], []
    for i in range(lookback, len(data) - forecast + 1):
        X.append(data[i - lookback : i])
        y.append(data[i : i + forecast])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape, forecast=10, units=64, dropout_rate=0.2, learning_rate=0.001):
    model = Sequential([
        LSTM(units, input_shape=input_shape, activation='tanh', return_sequences=False),
        Dropout(dropout_rate),
        Dense(forecast) # output shape matches forecast horizon (e.g. 10)
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model

def build_rnn_model(input_shape, forecast=10, units=64, dropout_rate=0.2, learning_rate=0.001):
    model = Sequential([
        SimpleRNN(units, input_shape=input_shape, activation='tanh', return_sequences=False),
        Dropout(dropout_rate),
        Dense(forecast) # output shape matches forecast horizon (e.g. 10)
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model

def custom_grid_search(X_train, y_train, input_shape, epochs=15):
    """
    Performs grid search hyperparameter tuning for LSTM.
    We split training data into 80% train / 20% validation chronologically to tune.
    """
    print("\nStarting Hyperparameter Tuning for LSTM...")
    
    # Chronological train-val split
    split_idx = int(len(X_train) * 0.8)
    X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
    y_tr, y_val = y_train[:split_idx], y_train[split_idx:]
    
    param_grid = {
        'units': [32, 64],
        'dropout_rate': [0.1, 0.2],
        'learning_rate': [0.001, 0.01]
    }
    
    best_loss = float('inf')
    best_params = {}
    
    results = []
    
    for units in param_grid['units']:
        for dropout in param_grid['dropout_rate']:
            for lr in param_grid['learning_rate']:
                print(f"Testing LSTM - Units: {units}, Dropout: {dropout}, LR: {lr}")
                model = build_lstm_model(input_shape, forecast=y_tr.shape[1], units=units, dropout_rate=dropout, learning_rate=lr)
                early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
                
                history = model.fit(
                    X_tr, y_tr,
                    epochs=epochs,
                    batch_size=32,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stop],
                    verbose=0
                )
                
                val_loss = min(history.history['val_loss'])
                print(f"-> Validation Loss (MSE): {val_loss:.6f}")
                
                results.append({
                    'units': units,
                    'dropout_rate': dropout,
                    'learning_rate': lr,
                    'val_loss': val_loss
                })
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_params = {
                        'units': units,
                        'dropout_rate': dropout,
                        'learning_rate': lr
                    }
                    
    results_df = pd.DataFrame(results)
    print("\nGrid Search Completed!")
    print(f"Best Parameters: {best_params} with Validation Loss: {best_loss:.6f}")
    return best_params, results_df

def calculate_metrics(y_true, y_pred, horizon_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'Horizon': horizon_name,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

def main():
    os.makedirs('models', exist_ok=True)
    
    # 1. Load Data
    df = load_and_clean_data("e:/Tesla Stock Price Predication/TSLA.csv")
    
    # Select target column
    target_data = df[['Adj Close']].values
    
    # 2. Train-Test Split (Chronological 80% / 20%)
    split_idx = int(len(target_data) * 0.8)
    train_data = target_data[:split_idx]
    test_data = target_data[split_idx:]
    
    test_dates = df.index[split_idx:]
    
    # 3. Fit scaler on training data and transform both
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_data)
 
    lookback = 60
    forecast = 10
    
    test_scaled_padded = np.concatenate([train_scaled[-lookback:], scaler.transform(test_data)], axis=0)
    
    # Save scaler
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # 4. Prepare Sequences
    X_train, y_train = prepare_sequences(train_scaled, lookback, forecast)
    X_test, y_test = prepare_sequences(test_scaled_padded, lookback, forecast)
    
    # We must squeeze targets to be 2D of shape (samples, forecast)
    y_train = y_train.squeeze(-1)
    y_test = y_test.squeeze(-1)
    
    input_shape = (X_train.shape[1], X_train.shape[2]) # (60, 1)
    
    # 5. LSTM Hyperparameter Tuning
    best_params, results_df = custom_grid_search(X_train, y_train, input_shape, epochs=15)
    results_df.to_csv('models/grid_search_results.csv', index=False)
    
    # 6. Train Final LSTM Model with best params
    print("\nTraining Final LSTM Model...")
    lstm_model = build_lstm_model(
        input_shape,
        forecast=forecast,
        units=best_params['units'],
        dropout_rate=best_params['dropout_rate'],
        learning_rate=best_params['learning_rate']
    )
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lstm_history = lstm_model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Save LSTM model
    lstm_model.save('models/lstm_model.h5')
    print("LSTM Model saved to models/lstm_model.h5")
    
    # 7. Train SimpleRNN Model
    print("\nTraining SimpleRNN Model...")
    # Use standard hyperparameters for comparison
    rnn_model = build_rnn_model(
        input_shape,
        forecast=forecast,
        units=64,
        dropout_rate=0.2,
        learning_rate=0.001
    )
    
    rnn_history = rnn_model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Save RNN model
    rnn_model.save('models/rnn_model.h5')
    print("SimpleRNN Model saved to models/rnn_model.h5")
    
    # 8. Predictions and Inverse Scaling
    lstm_preds_scaled = lstm_model.predict(X_test)
    rnn_preds_scaled = rnn_model.predict(X_test)
    

    lstm_preds = scaler.inverse_transform(lstm_preds_scaled)
    rnn_preds = scaler.inverse_transform(rnn_preds_scaled)
    
    # Save prediction metrics
    metrics = []
    
    horizons = [(1, 0, '1-Day Ahead'), (5, 4, '5-Day Ahead'), (10, 9, '10-Day Ahead')]
    
    for h, idx, label in horizons:
        # LSTM
        lstm_metrics = calculate_metrics(y_test_original[:, idx], lstm_preds[:, idx], f"LSTM_{label}")
        # RNN
        rnn_metrics = calculate_metrics(y_test_original[:, idx], rnn_preds[:, idx], f"RNN_{label}")
        
        metrics.append(lstm_metrics)
        metrics.append(rnn_metrics)
        
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv('models/evaluation_metrics.csv', index=False)
    print("\nEvaluation Metrics:")
    print(metrics_df)
   
    pred_dates = test_dates[:len(lstm_preds)]
    
    predictions_df = pd.DataFrame({
        'Date': pred_dates,
        'Actual_Close': test_data[:len(lstm_preds)].flatten(),
        'LSTM_1d': lstm_preds[:, 0],
        'LSTM_5d': lstm_preds[:, 4],
        'LSTM_10d': lstm_preds[:, 9],
        'RNN_1d': rnn_preds[:, 0],
        'RNN_5d': rnn_preds[:, 4],
        'RNN_10d': rnn_preds[:, 9],
    })
    predictions_df.to_csv('models/predictions.csv', index=False)
    print("Predictions saved to models/predictions.csv")
    
    # 10. Save training histories
    history_df = pd.DataFrame({
        'epoch': range(1, len(lstm_history.history['loss']) + 1),
        'lstm_loss': lstm_history.history['loss'],
        'lstm_val_loss': lstm_history.history['val_loss']
    })
    history_df.to_csv('models/lstm_history.csv', index=False)
    
    rnn_hist_df = pd.DataFrame({
        'epoch': range(1, len(rnn_history.history['loss']) + 1),
        'rnn_loss': rnn_history.history['loss'],
        'rnn_val_loss': rnn_history.history['val_loss']
    })
    rnn_hist_df.to_csv('models/rnn_history.csv', index=False)
    print("Training histories saved.")

if __name__ == "__main__":
    main()
