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
    queryset = TestingOptionData.objects.all().values()
    return pd.DataFrame(list(queryset))

def bulk_fetch_price_history(tickers, period_days=365):
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
    df = df.dropna(subset=['trade_price', 'strike_price', 'trade_datetime', 'expiration_date', 'underlying'])
    n_before = len(df)
    df = df[df['trade_price'] > 0]
    n_after = len(df)
    print(f"Filtered out {n_before - n_after} rows with trade_price <= 0.")
    
    df['trade_datetime'] = pd.to_datetime(df['trade_datetime']).dt.tz_localize(None)
    df['expiration_date'] = pd.to_datetime(df['expiration_date']).dt.tz_localize(None)
    
    df['days_to_expiry'] = (df['expiration_date'] - df['trade_datetime']).dt.days
    
    df['option_type_encoded'] = df['option_type'].map({'C': 1, 'P': 0})
    
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
    price_history = bulk_fetch_price_history(all_tickers, period_days=365)
    
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
    df['moneyness'] = df['underlying_price'] / df['strike_price']
    df['volatility_sq'] = df['volatility'] ** 2
    df['days_to_expiry_sq'] = df['days_to_expiry'] ** 2
    df = df.dropna(subset=['underlying_price', 'volatility', 'risk_free_rate'])
    return df

def scale_features(X):
    mu = X.mean()
    sigma = X.std()
    X_scaled = (X - mu) / sigma
    return X_scaled, mu, sigma

def gradient_descent(X, y, alpha=0.01, lambda_=0.1, num_iters=1000):
    m, n = X.shape
    X = np.c_[np.ones(m), X]
    theta = np.zeros(n + 1)

    for i in range(num_iters):
        predictions = X @ theta
        errors = predictions - y
        gradient = (1/m) * (X.T @ errors)
        
        reg = (lambda_/m) * np.r_[0, theta[1:]]
        theta -= alpha * (gradient + reg)

        if i % 100 == 0:
            cost = (1/(2*m)) * np.sum(errors**2) + (lambda_/(2*m)) * np.sum(theta[1:]**2)
            print(f"Iteration {i}: Cost = {cost:.4f}")
    
    return theta

def predict_with_gradient_descent(X, theta, mu, sigma):
    X_scaled = (X - mu) / sigma
    X_with_intercept = np.c_[np.ones(X_scaled.shape[0]), X_scaled]
    return X_with_intercept @ theta

def generate_prediction_dataframe(df, theta, mu, sigma):
    features = ['strike_price', 'underlying_price', 'option_type_encoded', 'days_to_expiry', 'volatility', 'moneyness', 'volatility_sq', 'days_to_expiry_sq']
    X = df[features]
    y_actual = df['trade_price'].values
    y_predicted = predict_with_gradient_descent(X, theta, mu, sigma)
    
    result_df = df.copy()
    result_df['predicted_price'] = y_predicted
    result_df['residual'] = y_actual - y_predicted
    
    safe_y_actual = y_actual.copy()
    safe_y_actual[safe_y_actual < 1] = 1
    result_df['residual_pct'] = (result_df['residual'] / safe_y_actual) * 100

    result_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    result_df.fillna(0, inplace=True)
    
    return result_df[['trade_datetime', 'underlying', 'option_type', 'strike_price', 'days_to_expiry', 'trade_price', 'predicted_price', 'residual', 'residual_pct']]

def run_regression(df):
    X = df[['strike_price', 'underlying_price', 'option_type_encoded', 'days_to_expiry', 'volatility']]
    y = df['trade_price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
    features = ['strike_price', 'underlying_price', 'option_type_encoded', 'days_to_expiry', 'volatility', 'moneyness', 'volatility_sq', 'days_to_expiry_sq']
    df = df.dropna(subset=features + ['trade_price'])

    X = df[features]
    y = df['trade_price'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = pd.DataFrame(X_train, columns=features)
    X_test = pd.DataFrame(X_test, columns=features)
    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)

    train_mask = ~(X_train.isna().any(axis=1) | y_train.isna())
    test_mask = ~(X_test.isna().any(axis=1) | y_test.isna())
    X_train = X_train[train_mask]
    y_train = y_train[train_mask].values
    X_test = X_test[test_mask]
    y_test = y_test[test_mask].values

    X_train_scaled_clean, mu, sigma = scale_features(X_train)
    X_test_scaled_clean = (X_test - mu) / sigma
    y_train_clean = pd.Series(y_train)
    y_test_clean = pd.Series(y_test)

    print(f"After NaN removal: X_train_scaled: {X_train_scaled_clean.shape}, y_train: {y_train_clean.shape}, X_test_scaled: {X_test_scaled_clean.shape}, y_test: {y_test_clean.shape}")

    if X_train_scaled_clean.empty or X_test_scaled_clean.empty:
        print("ERROR: No data left after NaN removal. Check your data and feature engineering steps.")
        print("Sample of X_train_scaled_clean:", X_train_scaled_clean.head())
        raise ValueError("No data left after NaN removal. Cannot train model.")

    print(f"Training with gradient descent (α={alpha}, λ={lambda_}, iterations={num_iters})...")
    theta = gradient_descent(X_train_scaled_clean.values, y_train_clean.values, alpha, lambda_, num_iters)
    
    preds = predict_with_gradient_descent(X_test_scaled_clean, theta, mu, sigma)
    y_test_exp = y_test_clean.values
    print("R² Score (trade_price):", r2_score(y_test_exp, preds))
    print("MSE (trade_price):", mean_squared_error(y_test_exp, preds))

    feature_names = ['intercept'] + list(X.columns)
    coeff_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': theta
    })
    print("\nFeature Coefficients:")
    print(coeff_df)

    print("\nGenerating prediction comparison...")
    prediction_df = generate_prediction_dataframe(df, theta, mu, sigma)
    
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
    df = load_option_data()
    if df.empty:
        return {"error": "No data available for regression."}
    df_processed = preprocess(df)
    if df_processed.empty:
        return {"error": "No data after preprocessing (missing features)."}

    theta, scaling_params, prediction_df = run_gradient_descent_regression(
        df_processed, alpha=alpha, lambda_=lambda_, num_iters=num_iters
    )

    y_true = prediction_df['trade_price'].values
    y_pred = prediction_df['predicted_price'].values
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

    feature_names = ['intercept', 'strike_price', 'underlying_price', 'option_type_encoded', 'days_to_expiry', 'volatility', 'moneyness', 'volatility_sq', 'days_to_expiry_sq']
    coefficients = [
        {"feature": name, "coefficient": float(coef)}
        for name, coef in zip(feature_names, theta)
    ]

    summary_stats = {
        "mean_absolute_error": float(np.abs(prediction_df['residual']).mean()),
        "mean_absolute_percentage_error": float(np.abs(prediction_df['residual_pct']).mean()),
        "std_residuals": float(prediction_df['residual'].std()),
        "max_overprediction": float(prediction_df['residual'].min()),
        "max_underprediction": float(prediction_df['residual'].max()),
    }

    predictions = prediction_df.head(100).to_dict(orient="records")

    return {
        "r2_score": r2,
        "mse": mse,
        "coefficients": coefficients,
        "predictions": predictions,
        "summary_stats": summary_stats,
    }

if __name__ == "__main__":
    df = load_option_data()
    print(f"Loaded {len(df)} records from TestingOptionData.")
    n_total = len(df)
    n_missing_price = df['close_price'].isna().sum() if 'close_price' in df.columns else 0
    pct_missing_price = 100 * n_missing_price / n_total if n_total else 0
    print(f"Rows missing close_price: {n_missing_price} ({pct_missing_price:.1f}%)")
    print("Note: Volatility is not a DB field; it is computed in preprocessing and may be missing due to missing price history.")
    df_processed = preprocess(df)
    n_missing_vol = df_processed['volatility'].isna().sum() if 'volatility' in df_processed.columns else 0
    n_proc = len(df_processed)
    pct_missing_vol = 100 * n_missing_vol / n_proc if n_proc else 0
    print(f"Rows missing volatility after preprocessing: {n_missing_vol} ({pct_missing_vol:.1f}%) out of {n_proc} preprocessed rows.")
    
    print("\nTraining regression model with gradient descent...")
    theta, scaling_params, prediction_df = run_gradient_descent_regression(df_processed)