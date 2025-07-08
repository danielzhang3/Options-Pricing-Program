import pandas as pd 
import numpy as np 
from datetime import datetime, timedelta
from api.stockutils import ticker_symbols, fallback_map, get_risk_free_rate
from api.models import TestingOptionData
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import yfinance as yf

def load_option_data():
    """
    Fetch all option data from the database and return as a pandas DataFrame.
    Returns:
        pd.DataFrame: DataFrame containing all fields from TestingOptionData.
    """
    queryset = TestingOptionData.objects.all().values()
    return pd.DataFrame(list(queryset))

def bulk_fetch_price_history(tickers, period_days=90):
    price_history = {}
    for ticker in tickers:
        try:
            hist = yf.Ticker(ticker).history(period=f'{period_days}d')
            price_history[ticker] = hist
        except Exception as e:
            print(f"Failed to fetch history for {ticker}: {e}")
            price_history[ticker] = pd.DataFrame()
    return price_history

def get_price_from_cache(price_history, ticker, date):
    hist = price_history.get(ticker)
    if hist is not None and not hist.empty:
        # Make both index and date tz-naive, and reindex DataFrame
        if hist.index.tz is not None:
            hist = hist.copy()
            hist.index = hist.index.tz_localize(None)
        target_date = pd.Timestamp(date).tz_localize(None) if pd.Timestamp(date).tzinfo is not None else pd.Timestamp(date)
        prev_dates = hist.index[hist.index <= target_date]
        if len(prev_dates) > 0:
            return hist.loc[prev_dates[-1], 'Close']
    return None

def get_vol_from_cache(price_history, ticker, date):
    hist = price_history.get(ticker)
    if hist is not None and not hist.empty:
        if hist.index.tz is not None:
            hist = hist.copy()
            hist.index = hist.index.tz_localize(None)
        target_date = pd.Timestamp(date).tz_localize(None) if pd.Timestamp(date).tzinfo is not None else pd.Timestamp(date)
        prev_dates = hist.index[hist.index <= target_date]
        if len(prev_dates) >= 30:
            last_30 = prev_dates[-30:]
            returns = hist.loc[last_30, 'Close'].pct_change()
            return returns.std() * np.sqrt(252)
    return None

def preprocess(df):
    """
    Preprocess the option data DataFrame with feature engineering.
    Args:
        df (pd.DataFrame): Raw option data DataFrame.
    Returns:
        pd.DataFrame: Preprocessed DataFrame with engineered features.
    """
    # Drop rows with missing critical data
    df = df.dropna(subset=['trade_price', 'strike_price', 'trade_datetime', 'expiration_date', 'underlying'])
    # Filter out rows where trade_price <= 0
    n_before = len(df)
    df = df[df['trade_price'] > 0]
    n_after = len(df)
    print(f"Filtered out {n_before - n_after} rows with trade_price <= 0.")
    
    # Convert datetime columns and remove timezone info
    df['trade_datetime'] = pd.to_datetime(df['trade_datetime']).dt.tz_localize(None)
    df['expiration_date'] = pd.to_datetime(df['expiration_date']).dt.tz_localize(None)
    
    # Calculate days to expiry
    df['days_to_expiry'] = (df['expiration_date'] - df['trade_datetime']).dt.days
    
    # Encode option type (C=1, P=0)
    df['option_type_encoded'] = df['option_type'].map({'C': 1, 'P': 0})
    
    # Identify all unique underlyings and map to tickers (including fallbacks)
    unique_underlyings = set(df['underlying'].unique())
    all_tickers = set()
    for underlying in unique_underlyings:
        base_symbol = underlying.upper().replace('$', '')
        primary = ticker_symbols.get(base_symbol, base_symbol)
        fallback = fallback_map.get(primary)
        all_tickers.add(primary)
        if fallback:
            all_tickers.add(fallback)
    print(f"Bulk fetching price history for: {all_tickers}")
    price_history = bulk_fetch_price_history(all_tickers, period_days=180)
    
    # Fetch underlying prices and volatility using the pre-fetched data
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
    
    print("Looking up underlying prices...")
    df['underlying_price'] = df.apply(lookup_price, axis=1)
    print("Looking up volatility data...")
    df['volatility'] = df.apply(lookup_vol, axis=1)
    print("Adding risk-free rate...")
    df['risk_free_rate'] = df.apply(
        lambda row: get_risk_free_rate(row['trade_datetime']), 
        axis=1
    )
    print("Sample risk_free_rate values after assignment:", df['risk_free_rate'].head(10))
    print("Number of NaNs in risk_free_rate after assignment:", df['risk_free_rate'].isna().sum())
    # Calculate moneyness if we have underlying price
    df['moneyness'] = df['underlying_price'] / df['strike_price']
    # Add new engineered features
    df['volatility_sq'] = df['volatility'] ** 2
    df['days_to_expiry_sq'] = df['days_to_expiry'] ** 2
    # Drop rows with missing external features
    df = df.dropna(subset=['underlying_price', 'volatility', 'risk_free_rate'])
    return df

def scale_features(X):
    """
    Scale features using z-score normalization (standardization).
    Args:
        X (pd.DataFrame): Features to scale.
    Returns:
        tuple: (X_scaled, mu, sigma) where mu and sigma are the scaling parameters.
    """
    mu = X.mean()
    sigma = X.std()
    X_scaled = (X - mu) / sigma
    return X_scaled, mu, sigma

def gradient_descent(X, y, alpha=0.01, lambda_=0.1, num_iters=1000):
    """
    Implement gradient descent with L2 regularization (Ridge regression).
    Args:
        X (np.array): Feature matrix (scaled).
        y (np.array): Target variable.
        alpha (float): Learning rate.
        lambda_ (float): Regularization parameter.
        num_iters (int): Number of iterations.
    Returns:
        np.array: Optimized parameters theta.
    """
    m, n = X.shape
    X = np.c_[np.ones(m), X]  # add intercept term
    theta = np.zeros(n + 1)

    for i in range(num_iters):
        predictions = X @ theta
        errors = predictions - y
        gradient = (1/m) * (X.T @ errors)
        
        # Add regularization (excluding intercept)
        reg = (lambda_/m) * np.r_[0, theta[1:]]
        theta -= alpha * (gradient + reg)

        if i % 100 == 0:
            cost = (1/(2*m)) * np.sum(errors**2) + (lambda_/(2*m)) * np.sum(theta[1:]**2)
            print(f"Iteration {i}: Cost = {cost:.4f}")
    
    return theta

def predict_with_gradient_descent(X, theta, mu, sigma):
    """
    Make predictions using the gradient descent model.
    Args:
        X (pd.DataFrame): Features to predict on.
        theta (np.array): Model parameters.
        mu, sigma: Scaling parameters.
    Returns:
        np.array: Predictions.
    """
    X_scaled = (X - mu) / sigma
    X_with_intercept = np.c_[np.ones(X_scaled.shape[0]), X_scaled]
    return X_with_intercept @ theta

def generate_prediction_dataframe(df, theta, mu, sigma):
    """
    Return DataFrame with actual vs. predicted prices for visualization.
    Args:
        df (pd.DataFrame): Preprocessed DataFrame.
        theta (np.array): Model parameters.
        mu, sigma: Scaling parameters.
    Returns:
        pd.DataFrame: DataFrame with actual vs predicted prices and residuals.
    """
    features = ['strike_price', 'underlying_price', 'option_type_encoded', 'days_to_expiry', 'volatility', 'moneyness', 'volatility_sq', 'days_to_expiry_sq']
    X = df[features]
    y_actual = df['trade_price'].values
    y_predicted = predict_with_gradient_descent(X, theta, mu, sigma)
    
    result_df = df.copy()
    result_df['predicted_price'] = y_predicted
    result_df['residual'] = y_actual - y_predicted
    
    # Calculate percentage errors with safe MAPE to prevent distortion from small values
    safe_y_actual = y_actual.copy()
    safe_y_actual[safe_y_actual < 1] = 1  # prevent exploding percentage errors
    result_df['residual_pct'] = (result_df['residual'] / safe_y_actual) * 100

    # Replace inf, -inf, and NaN with 0 for safe JSON serialization
    result_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    result_df.fillna(0, inplace=True)
    
    return result_df[['trade_datetime', 'underlying', 'option_type', 'strike_price', 'days_to_expiry', 'trade_price', 'predicted_price', 'residual', 'residual_pct']]

def run_regression(df):
    """
    Train a multiple linear regression model on the preprocessed option data.
    Args:
        df (pd.DataFrame): Preprocessed DataFrame with engineered features.
    Returns:
        tuple: (model, scaling_params) where scaling_params contains mu and sigma.
    """
    X = df[['strike_price', 'underlying_price', 'option_type_encoded', 'days_to_expiry', 'volatility']]
    y = df['trade_price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    print("Scaling features...")
    X_train_scaled, mu, sigma = scale_features(X_train)
    X_test_scaled = (X_test - mu) / sigma

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)
    print("R² Score (trade_price):", r2_score(y_test, preds))
    print("MSE (trade_price):", mean_squared_error(y_test, preds))

    coeff_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_
    })
    print("\nFeature Coefficients:")
    print(coeff_df)

    scaling_params = {'mu': mu, 'sigma': sigma}
    return model, scaling_params

def run_gradient_descent_regression(df, alpha=0.01, lambda_=0.1, num_iters=1000):
    """
    Train a multiple linear regression model using gradient descent with regularization.
    Args:
        df (pd.DataFrame): Preprocessed DataFrame with engineered features.
        alpha (float): Learning rate.
        lambda_ (float): Regularization parameter.
        num_iters (int): Number of iterations.
    Returns:
        tuple: (theta, scaling_params, prediction_df) where prediction_df contains actual vs predicted prices.
    """
    features = ['strike_price', 'underlying_price', 'option_type_encoded', 'days_to_expiry', 'volatility', 'moneyness', 'volatility_sq', 'days_to_expiry_sq']
    # Drop any rows with NaN in features or target
    df = df.dropna(subset=features + ['trade_price'])

    X = df[features]
    y = df['trade_price'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to DataFrames for easier NaN dropping
    X_train = pd.DataFrame(X_train, columns=features)
    X_test = pd.DataFrame(X_test, columns=features)
    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)

    # Drop any rows with NaNs in train/test sets
    train_mask = ~(X_train.isna().any(axis=1) | y_train.isna())
    test_mask = ~(X_test.isna().any(axis=1) | y_test.isna())
    X_train = X_train[train_mask]
    y_train = y_train[train_mask].values
    X_test = X_test[test_mask]
    y_test = y_test[test_mask].values

    # Scale features
    print("Scaling features...")
    X_train_scaled, mu, sigma = scale_features(X_train)
    X_test_scaled = (X_test - mu) / sigma

    # NaN diagnostics after scaling
    print("NaNs per column in X_train_scaled:\n", X_train_scaled.isna().sum())
    print("NaNs per column in X_test_scaled:\n", X_test_scaled.isna().sum())

    # Drop rows with any NaNs in features or target, using index alignment
    y_train_series = pd.Series(y_train, index=X_train_scaled.index)
    train_mask = ~(X_train_scaled.isna().any(axis=1) | y_train_series.isna())
    X_train_scaled_clean = X_train_scaled[train_mask]
    y_train_clean = y_train_series[train_mask]

    y_test_series = pd.Series(y_test, index=X_test_scaled.index)
    test_mask = ~(X_test_scaled.isna().any(axis=1) | y_test_series.isna())
    X_test_scaled_clean = X_test_scaled[test_mask]
    y_test_clean = y_test_series[test_mask]

    print(f"After NaN removal: X_train_scaled: {X_train_scaled_clean.shape}, y_train: {y_train_clean.shape}, X_test_scaled: {X_test_scaled_clean.shape}, y_test: {y_test_clean.shape}")

    # Check for empty sets
    if X_train_scaled_clean.empty or X_test_scaled_clean.empty:
        print("ERROR: No data left after NaN removal. Check your data and feature engineering steps.")
        print("Sample of original X_train_scaled:", X_train_scaled.head())
        raise ValueError("No data left after NaN removal. Cannot train model.")

    # Train model using gradient descent
    print(f"Training with gradient descent (α={alpha}, λ={lambda_}, iterations={num_iters})...")
    theta = gradient_descent(X_train_scaled_clean.values, y_train_clean.values, alpha, lambda_, num_iters)
    
    # Make predictions
    preds = predict_with_gradient_descent(X_test_scaled_clean, theta, mu, sigma)
    y_test_exp = y_test_clean.values
    print("R² Score (trade_price):", r2_score(y_test_exp, preds))
    print("MSE (trade_price):", mean_squared_error(y_test_exp, preds))

    # Create coefficient DataFrame
    feature_names = ['intercept'] + list(X.columns)
    coeff_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': theta
    })
    print("\nFeature Coefficients:")
    print(coeff_df)

    # Generate prediction comparison DataFrame
    print("\nGenerating prediction comparison...")
    prediction_df = generate_prediction_dataframe(df, theta, mu, sigma)
    
    # Print summary statistics
    print("\nPrediction Summary Statistics:")
    print(f"Mean Absolute Error: ${prediction_df['residual'].abs().mean():.2f}")
    print(f"Mean Absolute Percentage Error: {prediction_df['residual_pct'].abs().mean():.1f}%")
    print(f"Standard Deviation of Residuals: ${prediction_df['residual'].std():.2f}")
    print(f"Max Overprediction: ${prediction_df['residual'].min():.2f}")
    print(f"Max Underprediction: ${prediction_df['residual'].max():.2f}")
    
    print("\nSample Predictions (first 10 rows):")
    print(prediction_df.head(10))

    scaling_params = {'mu': mu, 'sigma': sigma}
    return theta, scaling_params, prediction_df

def train_multiple_linear_regression(alpha=0.01, lambda_=0.1, num_iters=1000):
    """
    API entry point: Train multiple linear regression using gradient descent and return results for frontend.
    Args:
        alpha (float): Learning rate.
        lambda_ (float): Regularization parameter.
        num_iters (int): Number of iterations.
    Returns:
        dict: Results including metrics, coefficients, predictions, and summary stats.
    """
    # Load and preprocess data
    df = load_option_data()
    if df.empty:
        return {"error": "No data available for regression."}
    df_processed = preprocess(df)
    if df_processed.empty:
        return {"error": "No data after preprocessing (missing features)."}

    # Train model
    theta, scaling_params, prediction_df = run_gradient_descent_regression(
        df_processed, alpha=alpha, lambda_=lambda_, num_iters=num_iters
    )

    # Calculate metrics
    y_true = prediction_df['trade_price'].values
    y_pred = prediction_df['predicted_price'].values
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

    # Coefficients
    feature_names = ['intercept', 'strike_price', 'underlying_price', 'option_type_encoded', 'days_to_expiry', 'volatility', 'moneyness', 'volatility_sq', 'days_to_expiry_sq']
    coefficients = [
        {"feature": name, "coefficient": float(coef)}
        for name, coef in zip(feature_names, theta)
    ]

    # Summary statistics
    summary_stats = {
        "mean_absolute_error": float(np.abs(prediction_df['residual']).mean()),
        "mean_absolute_percentage_error": float(np.abs(prediction_df['residual_pct']).mean()),
        "std_residuals": float(prediction_df['residual'].std()),
        "max_overprediction": float(prediction_df['residual'].min()),
        "max_underprediction": float(prediction_df['residual'].max()),
    }

    # Sample predictions (limit to 100 for frontend)
    predictions = prediction_df.head(100).to_dict(orient="records")

    return {
        "r2_score": r2,
        "mse": mse,
        "coefficients": coefficients,
        "predictions": predictions,
        "summary_stats": summary_stats,
    }

if __name__ == "__main__":
    # Load and preprocess data
    df = load_option_data()
    print(f"Loaded {len(df)} records from TestingOptionData.")
    n_total = len(df)
    n_missing_price = df['close_price'].isna().sum() if 'close_price' in df.columns else 0
    pct_missing_price = 100 * n_missing_price / n_total if n_total else 0
    print(f"Rows missing close_price: {n_missing_price} ({pct_missing_price:.1f}%)")
    print("Note: Volatility is not a DB field; it is computed in preprocessing and may be missing due to missing price history.")
    # If you want to check missing volatility after preprocessing:
    df_processed = preprocess(df)
    n_missing_vol = df_processed['volatility'].isna().sum() if 'volatility' in df_processed.columns else 0
    n_proc = len(df_processed)
    pct_missing_vol = 100 * n_missing_vol / n_proc if n_proc else 0
    print(f"Rows missing volatility after preprocessing: {n_missing_vol} ({pct_missing_vol:.1f}%) out of {n_proc} preprocessed rows.")
    
    # Train the model with gradient descent
    print("\nTraining regression model with gradient descent...")
    theta, scaling_params, prediction_df = run_gradient_descent_regression(df_processed)