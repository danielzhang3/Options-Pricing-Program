import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import shap
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid GUI issues
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from api.models import TestingOptionData
from api.stockutils import ticker_symbols, fallback_map, get_risk_free_rate
from api.multiple_linear_regression import bulk_fetch_price_history, get_price_from_cache, get_vol_from_cache


def load_and_preprocess_data():
    """
    Load and preprocess option data for neural network training.
    Returns:
        tuple: (X, y) where X is feature matrix and y is target prices
    """
    print("Loading option data from database...")
    df = pd.DataFrame(list(TestingOptionData.objects.all().values()))
    
    if df.empty:
        raise ValueError("No data found in TestingOptionData")
    
    print(f"Initial data shape: {df.shape}")
    
    # Filter out invalid trade prices
    df = df[df['trade_price'] > 0]
    print(f"After filtering trade_price > 0: {df.shape}")

    # Parse dates
    df['trade_datetime'] = pd.to_datetime(df['trade_datetime']).dt.tz_localize(None)
    df['expiration_date'] = pd.to_datetime(df['expiration_date']).dt.tz_localize(None)
    df['days_to_expiry'] = (df['expiration_date'] - df['trade_datetime']).dt.days
    
    # Filter out expired options
    df = df[df['days_to_expiry'] >= 0]
    print(f"After filtering days_to_expiry > 0: {df.shape}")

    # Feature engineering
    df['option_type_encoded'] = df['option_type'].map({'C': 1, 'P': 0, 'call': 1, 'put': 0})
    
    # Fetch market data for missing underlying prices and volatility
    print("Fetching market data for missing prices and volatility...")
    df = fetch_market_data(df)
    
    # Additional feature engineering
    df['risk_free_rate'] = df.apply(lambda row: get_risk_free_rate(row['trade_datetime']), axis=1)
    df['moneyness'] = df['underlying_price'] / df['strike_price']
    df['volatility_sq'] = df['volatility'] ** 2
    df['days_to_expiry_sq'] = df['days_to_expiry'] ** 2
    
    # Add more sophisticated features
    df['time_to_expiry_years'] = df['days_to_expiry'] / 365.0
    df['log_moneyness'] = np.log(df['moneyness'])
    df['volatility_time'] = df['volatility'] * np.sqrt(df['time_to_expiry_years'])
    
    # Define features for the model
    features = [
        'strike_price', 'underlying_price', 'option_type_encoded',
        'days_to_expiry', 'volatility', 'moneyness', 'risk_free_rate',
        'volatility_sq', 'days_to_expiry_sq', 'time_to_expiry_years',
        'log_moneyness', 'volatility_time'
    ]
    
    # Remove rows with missing data
    df = df.dropna(subset=features + ['trade_price'])
    print(f"After removing missing data: {df.shape}")
    
    # Remove outliers (optional - can be commented out)
    df = remove_outliers(df, 'trade_price', threshold=3)
    
    X = df[features]
    y = df['trade_price']
    
    print(f"Final feature matrix shape: {X.shape}")
    print(f"Feature names: {features}")
    print(f"Target price range: ${y.min():.2f} - ${y.max():.2f}")
    
    return X, y


def fetch_market_data(df):
    """
    Fetch underlying prices and volatility for options data.
    """
    # Get unique underlyings
    unique_underlyings = set(df['underlying'].unique())
    all_tickers = set()
    
    for underlying in unique_underlyings:
        base_symbol = underlying.upper().replace('$', '')
        primary = ticker_symbols.get(base_symbol, base_symbol)
        fallback = fallback_map.get(primary)
        all_tickers.add(primary)
        if fallback:
            all_tickers.add(fallback)
    
    print(f"Fetching price history for {len(all_tickers)} tickers...")
    price_history = bulk_fetch_price_history(all_tickers, period_days=180)
    
    def lookup_price(row):
        base_symbol = row['underlying'].upper().replace('$', '')
        primary = ticker_symbols.get(base_symbol, base_symbol)
        fallback = fallback_map.get(primary)
        for ticker in [primary, fallback]:
            if ticker:
                price = get_price_from_cache(price_history, ticker, row['trade_datetime'])
                if price is not None:
                    return price
        return None
    
    def lookup_vol(row):
        base_symbol = row['underlying'].upper().replace('$', '')
        primary = ticker_symbols.get(base_symbol, base_symbol)
        fallback = fallback_map.get(primary)
        for ticker in [primary, fallback]:
            if ticker:
                vol = get_vol_from_cache(price_history, ticker, row['trade_datetime'])
                if vol is not None:
                    return vol
        return None
    
    # Fill missing underlying prices and volatility
    df['underlying_price'] = df.apply(lookup_price, axis=1)
    df['volatility'] = df.apply(lookup_vol, axis=1)
    
    return df


def remove_outliers(df, column, threshold=3):
    """
    Remove outliers using z-score method.
    """
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    return df[z_scores < threshold]


def build_model(input_dim, learning_rate=0.001):
    """
    Build a neural network model for options pricing.
    
    Args:
        input_dim (int): Number of input features
        learning_rate (float): Learning rate for optimizer
    
    Returns:
        tf.keras.Model: Compiled neural network model
    """
    model = Sequential([
        # Input layer
        Dense(128, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Hidden layers
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        
        # Output layer
        Dense(1, activation='linear')  # Linear activation for regression
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    return model


def train_neural_network(epochs=100, batch_size=32, validation_split=0.2):
    """
    Train a neural network model for options pricing.
    
    Args:
        epochs (int): Maximum number of training epochs
        batch_size (int): Batch size for training
        validation_split (float): Fraction of data to use for validation
    
    Returns:
        tuple: (summary, model, scaler) containing evaluation metrics, trained model, and feature scaler
    """
    print("Starting neural network training...")
    
    # Load and preprocess data
    X, y = load_and_preprocess_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Build model
    model = build_model(X_train_scaled.shape[1])
    print(f"Model architecture:")
    model.summary()
    
    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=15, 
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train model
    print("Training model...")
    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test_scaled, verbose=0).flatten()
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    residuals = y_test - y_pred
    
    # Calculate percentage errors with safe MAPE to prevent distortion from small values
    safe_y_test = y_test.copy()
    safe_y_test[safe_y_test < 1] = 1  # prevent exploding percentage errors
    mape = np.mean(np.abs((y_test - y_pred) / safe_y_test)) * 100
    
    # Summary statistics
    summary = {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'r2': float(r2),
        'mape': float(mape),
        'residual_std': float(np.std(residuals)),
        'max_overprediction': float(min(residuals)),
        'max_underprediction': float(max(residuals)),
        'n_test_samples': len(y_test),
        'training_epochs': len(history.history['loss']),
        'final_training_loss': float(history.history['loss'][-1]),
        'final_validation_loss': float(history.history['val_loss'][-1])
    }
    
    print("\n" + "="*50)
    print("NEURAL NETWORK TRAINING RESULTS")
    print("="*50)
    print(f"Mean Absolute Error: ${summary['mae']:.4f}")
    print(f"Root Mean Square Error: ${summary['rmse']:.4f}")
    print(f"R² Score: {summary['r2']:.4f}")
    print(f"Mean Absolute Percentage Error: {summary['mape']:.2f}%")
    print(f"Residual Standard Deviation: ${summary['residual_std']:.4f}")
    print(f"Test Samples: {summary['n_test_samples']}")
    print(f"Training Epochs: {summary['training_epochs']}")
    print("="*50)
    
    # Add SHAP analysis for sample predictions
    print("\n" + "="*50)
    print("SHAP FEATURE IMPORTANCE ANALYSIS")
    print("="*50)
    
    # Analyze a few sample predictions
    sample_indices = [0, 1, 2]  # Analyze first 3 samples
    for i, sample_idx in enumerate(sample_indices):
        if sample_idx < len(X_test):
            print(f"\nSample {i+1} Analysis:")
            sample_input = X_test.iloc[sample_idx]
            sample_true = y_test.iloc[sample_idx]
            sample_dict = sample_input.to_dict()
            sample_pred = predict_option_price(model, scaler, sample_dict)
            
            print(f"  True price: ${sample_true:.2f}")
            print(f"  Predicted price: ${sample_pred:.2f}")
            print(f"  Error: ${abs(sample_true - sample_pred):.2f}")
            print(f"  Percentage Error: {abs(sample_true - sample_pred) / sample_true * 100:.2f}%")
            
            # Generate SHAP explanation for this sample
            try:
                explain_prediction_with_shap_kernel(model, scaler, X_test, sample_idx)
                print(f"  SHAP explanation generated for sample {sample_idx}")
            except Exception as e:
                print(f"  SHAP explanation failed for sample {sample_idx}: {str(e)}")
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    
    return summary, model, scaler


def predict_option_price(model, scaler, option_data):
    """
    Predict option price using trained neural network.
    
    Args:
        model: Trained neural network model
        scaler: Fitted StandardScaler
        option_data (dict): Dictionary containing option features
    
    Returns:
        float: Predicted option price
    """
    # Prepare features in the same order as training
    features = [
        'strike_price', 'underlying_price', 'option_type_encoded',
        'days_to_expiry', 'volatility', 'moneyness', 'risk_free_rate',
        'volatility_sq', 'days_to_expiry_sq', 'time_to_expiry_years',
        'log_moneyness', 'volatility_time'
    ]
    
    # Create feature vector
    X = np.array([[option_data[feature] for feature in features]])
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make prediction
    prediction = model.predict(X_scaled, verbose=0)[0][0]
    
    return float(prediction)


def explain_prediction_with_shap_kernel(model, scaler, X_test_df, sample_idx=0):
    """
    Explain a prediction using SHAP KernelExplainer (more stable for TF 2.4+)
    
    Args:
        model: Trained Keras Sequential model
        scaler: Fitted StandardScaler
        X_test_df (pd.DataFrame): Test features (unscaled)
        sample_idx (int): Index of the test sample to explain
    """
    # Select background and sample
    X_background = X_test_df.sample(100, random_state=42)
    X_explain = X_test_df.iloc[[sample_idx]]

    # Scale inputs
    X_background_scaled = scaler.transform(X_background)
    X_explain_scaled = scaler.transform(X_explain)

    # Define prediction function (wrapper around model)
    def model_predict(X):
        return model.predict(X).flatten()

    # KernelExplainer works with any black-box model
    explainer = shap.KernelExplainer(model_predict, X_background_scaled)

    # Compute SHAP values
    shap_values = explainer.shap_values(X_explain_scaled, nsamples=100)

    # Create and save waterfall plot
    fig = shap.plots.waterfall(shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X_explain_scaled[0],
        feature_names=X_test_df.columns.tolist()
    ))
    
    # Save the plot to a file
    plt.savefig(f'shap_explanation_sample_{sample_idx}.png', bbox_inches='tight', dpi=300)
    plt.close()  # Close the figure to free memory
    print(f"  SHAP plot saved as 'shap_explanation_sample_{sample_idx}.png'")


def analyze_sample_prediction(model, scaler, X_test, y_test, sample_idx=0):
    """
    Analyze a specific sample prediction with SHAP explanation.
    
    Args:
        model: Trained Keras Sequential model
        scaler: Fitted StandardScaler
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test targets
        sample_idx (int): Index of the test sample to analyze
    """
    # Get sample data
    sample_input = X_test.iloc[sample_idx]
    sample_true = y_test.iloc[sample_idx]
    
    # Convert sample_input to dictionary format expected by predict_option_price
    sample_dict = sample_input.to_dict()
    
    # Make prediction
    sample_pred = predict_option_price(model, scaler, sample_dict)

    print(f"Sample {sample_idx} Analysis:")
    print(f"True price: ${sample_true:.2f}")
    print(f"Predicted price: ${sample_pred:.2f}")
    print(f"Error: ${abs(sample_true - sample_pred):.2f}")
    print(f"Percentage Error: {abs(sample_true - sample_pred) / sample_true * 100:.2f}%")
    print("-" * 50)

    # SHAP explanation
    explain_prediction_with_shap_kernel(model, scaler, X_test, sample_idx)

