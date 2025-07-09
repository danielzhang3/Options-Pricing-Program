import os
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime
from django.conf import settings
import tensorflow as tf
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.models import save_model as keras_save_model

MODELS_DIR = os.path.join(settings.BASE_DIR, 'api', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

def save_ml_regression_model(theta, scaling_params, model_info):
    model_data = {
        'theta': theta,
        'scaling_params': scaling_params,
        'model_info': model_info,
        'saved_at': datetime.now().isoformat(),
        'model_type': 'ml_regression'
    }
    
    model_path = os.path.join(MODELS_DIR, 'ml_regression_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"ML Regression model saved to {model_path}")
    return model_path

def load_ml_regression_model():
    model_path = os.path.join(MODELS_DIR, 'ml_regression_model.pkl')
    
    if not os.path.exists(model_path):
        return None
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        return (
            model_data['theta'],
            model_data['scaling_params'],
            model_data['model_info']
        )
    except Exception as e:
        print(f"Error loading ML regression model: {e}")
        return None

def save_neural_network_model(model, scaler, model_info):
    model_path = os.path.join(MODELS_DIR, 'neural_network_model.keras')
    keras_save_model(model, model_path)
    
    scaler_path = os.path.join(MODELS_DIR, 'neural_network_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    info_path = os.path.join(MODELS_DIR, 'neural_network_info.json')
    model_info['saved_at'] = datetime.now().isoformat()
    model_info['model_type'] = 'neural_network'
    
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"Neural network model saved to {model_path}")
    return model_path

def load_neural_network_model():
    model_path = os.path.join(MODELS_DIR, 'neural_network_model.keras')
    scaler_path = os.path.join(MODELS_DIR, 'neural_network_scaler.pkl')
    info_path = os.path.join(MODELS_DIR, 'neural_network_info.json')
    
    if not all(os.path.exists(p) for p in [model_path, scaler_path, info_path]):
        return None
    
    try:
        model = keras_load_model(model_path)
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        with open(info_path, 'r') as f:
            model_info = json.load(f)
        
        return model, scaler, model_info
    except Exception as e:
        print(f"Error loading neural network model: {e}")
        return None

def predict_ml_regression(option_data):
    model_data = load_ml_regression_model()
    if model_data is None:
        return None
    
    theta, scaling_params, model_info = model_data
    
    feature_names = [
        'strike_price', 'underlying_price', 'option_type_encoded',
        'days_to_expiry', 'volatility', 'moneyness', 'volatility_sq', 'days_to_expiry_sq'
    ]
    
    feature_names = [f for f in feature_names if f != 'intercept']
    
    X = pd.DataFrame([[option_data[feature] for feature in feature_names]], columns=feature_names)
    
    mu = scaling_params['mu']
    sigma = scaling_params['sigma']
    
    mu = mu[feature_names]
    sigma = sigma[feature_names]
    
    X_scaled = (X - mu) / sigma
    X_with_intercept = np.c_[np.ones(X_scaled.shape[0]), X_scaled]
    prediction = X_with_intercept @ theta
    return float(prediction[0])

def predict_neural_network(option_data):
    model_data = load_neural_network_model()
    if model_data is None:
        return None
    
    model, scaler, model_info = model_data
    
    features = [
        'strike_price', 'underlying_price', 'option_type_encoded',
        'days_to_expiry', 'volatility', 'moneyness', 'risk_free_rate',
        'volatility_sq', 'days_to_expiry_sq', 'time_to_expiry_years',
        'log_moneyness', 'volatility_time'
    ]
    
    X = np.array([[option_data[feature] for feature in features]])
    
    X_scaled = scaler.transform(X)
    
    prediction = model.predict(X_scaled, verbose=0)[0][0]
    
    return float(prediction)

def get_model_status():
    status = {
        'ml_regression': {
            'available': False,
            'saved_at': None,
            'model_info': None
        },
        'neural_network': {
            'available': False,
            'saved_at': None,
            'model_info': None
        }
    }
    
    ml_data = load_ml_regression_model()
    if ml_data is not None:
        theta, scaling_params, model_info = ml_data
        status['ml_regression']['available'] = True
        status['ml_regression']['saved_at'] = model_info.get('saved_at')
        status['ml_regression']['model_info'] = model_info
    
    nn_data = load_neural_network_model()
    if nn_data is not None:
        model, scaler, model_info = nn_data
        status['neural_network']['available'] = True
        status['neural_network']['saved_at'] = model_info.get('saved_at')
        status['neural_network']['model_info'] = model_info
    
    return status 