import math
import csv
import io
import logging
import traceback
from io import TextIOWrapper
import re
from datetime import date, datetime
from django.utils import timezone
from dateutil.parser import parse as parse_dateutil
from rest_framework import status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.viewsets import ModelViewSet
from rest_framework.parsers import MultiPartParser, FormParser
from .models import Option, OptionPrice, MarketData, TestingOptionData, ModelTrainingResult
from .serializers import (
    OptionSerializer, 
    OptionPriceSerializer, 
    MarketDataSerializer,
    OptionCalculationSerializer,
    TestingOptionDataSerializer,
    ModelTrainingResultSerializer
)
from .multiple_linear_regression import train_multiple_linear_regression
from .stockutils import ticker_symbols, fallback_map, get_risk_free_rate
from .neural_network import train_neural_network
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BlackScholesCalculator:
    
    @staticmethod
    def normal_cdf(x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    
    @staticmethod
    def calculate_d1(S, K, T, r, sigma):
        return (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    
    @staticmethod
    def calculate_d2(d1, sigma, T):
        return d1 - sigma * math.sqrt(T)
    
    @staticmethod
    def call_price(S, K, T, r, sigma):
        d1 = BlackScholesCalculator.calculate_d1(S, K, T, r, sigma)
        d2 = BlackScholesCalculator.calculate_d2(d1, sigma, T)
        
        price = S * BlackScholesCalculator.normal_cdf(d1) - K * math.exp(-r * T) * BlackScholesCalculator.normal_cdf(d2)
        return price
    
    @staticmethod
    def put_price(S, K, T, r, sigma):
        d1 = BlackScholesCalculator.calculate_d1(S, K, T, r, sigma)
        d2 = BlackScholesCalculator.calculate_d2(d1, sigma, T)
        
        price = K * math.exp(-r * T) * BlackScholesCalculator.normal_cdf(-d2) - S * BlackScholesCalculator.normal_cdf(-d1)
        return price
    
    @staticmethod
    def calculate_greeks(S, K, T, r, sigma, option_type):
        d1 = BlackScholesCalculator.calculate_d1(S, K, T, r, sigma)
        d2 = BlackScholesCalculator.calculate_d2(d1, sigma, T)
        
        if option_type == 'call':
            delta = BlackScholesCalculator.normal_cdf(d1)
        else:  
            delta = BlackScholesCalculator.normal_cdf(d1) - 1
        
        gamma = math.exp(-0.5 * d1**2) / (S * sigma * math.sqrt(T) * math.sqrt(2 * math.pi))
        
        theta_term = -(S * sigma * math.exp(-0.5 * d1**2)) / (2 * math.sqrt(T) * math.sqrt(2 * math.pi))
        if option_type == 'call':
            theta = theta_term - r * K * math.exp(-r * T) * BlackScholesCalculator.normal_cdf(d2)
        else:  
            theta = theta_term + r * K * math.exp(-r * T) * BlackScholesCalculator.normal_cdf(-d2)
        
        vega = S * math.sqrt(T) * math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
        
        if option_type == 'call':
            rho = K * T * math.exp(-r * T) * BlackScholesCalculator.normal_cdf(d2)
        else:  
            rho = -K * T * math.exp(-r * T) * BlackScholesCalculator.normal_cdf(-d2)
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }


class OptionViewSet(ModelViewSet):
    queryset = Option.objects.all()
    serializer_class = OptionSerializer


class OptionPriceViewSet(ModelViewSet):
    queryset = OptionPrice.objects.all()
    serializer_class = OptionPriceSerializer


class MarketDataViewSet(ModelViewSet):
    queryset = MarketData.objects.all()
    serializer_class = MarketDataSerializer


class TestingOptionDataViewSet(ModelViewSet):
    queryset = TestingOptionData.objects.all()
    serializer_class = TestingOptionDataSerializer
    
    def get_queryset(self):
        queryset = TestingOptionData.objects.all()
        underlying = self.request.query_params.get('underlying', None)
        if underlying:
            queryset = queryset.filter(underlying__icontains=underlying)
        return queryset.order_by('-trade_datetime')


class ModelTrainingResultViewSet(ModelViewSet):
    queryset = ModelTrainingResult.objects.all()
    serializer_class = ModelTrainingResultSerializer
    
    def get_queryset(self):
        queryset = ModelTrainingResult.objects.all()
        model_type = self.request.query_params.get('model_type', None)
        if model_type:
            queryset = queryset.filter(model_type=model_type)
        return queryset.order_by('-training_date')





@api_view(['GET'])
def option_price_history(request, option_id):
    try:
        option = Option.objects.get(id=option_id)
        prices = OptionPrice.objects.filter(option=option).order_by('-calculated_at')
        serializer = OptionPriceSerializer(prices, many=True)
        return Response(serializer.data)
    except Option.DoesNotExist:
        return Response(
            {'error': 'Option not found'}, 
            status=status.HTTP_404_NOT_FOUND
        )


@api_view(['GET'])
def market_data_latest(request, symbol):
    try:
        latest_data = MarketData.objects.filter(symbol=symbol).latest('timestamp')
        serializer = MarketDataSerializer(latest_data)
        return Response(serializer.data)
    except MarketData.DoesNotExist:
        return Response(
            {'error': 'No market data found for symbol'}, 
            status=status.HTTP_404_NOT_FOUND
        )

def parse_float(value):
    try:
        return float(value.replace(',', '').strip()) if value.strip() else None
    except Exception:
        return None


class CSVUploadView(APIView):
    def post(self, request):
        file = request.FILES.get("file")
        if not file:
            return Response({"error": "No file provided"}, status=status.HTTP_400_BAD_REQUEST)

        decoded_file = TextIOWrapper(file.file, encoding="utf-8", errors="ignore")
        reader = csv.reader(decoded_file)

        expected_headers = [
            "Asset Category", "Currency", "Symbol", "Date/Time", "Quantity",
            "T. Price", "C. Price", "Proceeds", "Comm/Fee", "Basis",
            "Realized P/L", "MTM P/L", "Code"
        ]

        headers = []
        successes = 0
        failures = 0

        for row_num, row in enumerate(reader, start=1):
            if len(row) < 4:
                continue

            if row[0].lower() == "trades" and row[1].lower() == "header":
                try:
                    discriminator_index = row.index("DataDiscriminator")
                    headers = row[discriminator_index + 1: discriminator_index + 1 + len(expected_headers)]
                except ValueError:
                    pass
                continue

            if not headers or row[0].lower() != "trades" or row[1].lower() != "data":
                continue

            content = row[3:3 + len(headers)]
            row_dict = dict(zip(headers, content))

            asset_category = row_dict.get("Asset Category", "").strip().lower()
            if asset_category not in ["equity and index options", "options on futures"]:
                continue

            try:
                symbol = row_dict["Symbol"].strip()
                quantity = float(row_dict["Quantity"].strip())
                trade_price = float(row_dict["T. Price"].strip())
                date_time = datetime.strptime(row_dict["Date/Time"].strip().split(',')[0], "%Y-%m-%d")
                code = row_dict.get("Code", "").strip()
            except Exception:
                failures += 1
                continue

            close_price = parse_float(row_dict.get("C. Price", ""))
            proceeds = parse_float(row_dict.get("Proceeds", ""))
            comm_fee = parse_float(row_dict.get("Comm/Fee", ""))
            basis = parse_float(row_dict.get("Basis", ""))
            realized_pl = parse_float(row_dict.get("Realized P/L", ""))
            mtm_pl = parse_float(row_dict.get("MTM P/L", ""))

            parsed = re.match(r"(?P<underlying>.+?) (?P<exp>\d{2}[A-Z]{3}\d{2}) (?P<strike>\d+(\.\d+)?) (?P<type>[CP])", symbol)
            if not parsed:
                failures += 1
                continue

            try:
                expiration_date = datetime.strptime(parsed.group("exp"), "%d%b%y").date()
            except ValueError:
                failures += 1
                continue

            serializer = TestingOptionDataSerializer(data={
                "symbol": symbol,
                "underlying": parsed.group("underlying"),
                "strike_price": float(parsed.group("strike")),
                "expiration_date": expiration_date,
                "option_type": parsed.group("type"),
                "trade_price": trade_price,
                "quantity": int(quantity),
                "trade_datetime": date_time,
                "code": code,
                "close_price": close_price,
                "comm_fee": comm_fee,
                "proceeds": proceeds,
                "basis": basis,
                "realized_pl": realized_pl,
                "mtm_pl": mtm_pl,
            })

            if serializer.is_valid():
                serializer.save()
                successes += 1
            else:
                failures += 1

        return Response({"success": successes, "failed": failures}, status=status.HTTP_200_OK)


@api_view(['POST'])
def train_regression_model(request):
    try:
        alpha = request.data.get('alpha', 0.01)
        lambda_ = request.data.get('lambda_', 0.1)
        num_iters = request.data.get('num_iters', 1000)
        
        from .multiple_linear_regression import load_option_data, preprocess, run_gradient_descent_regression
        from .model_manager import save_ml_regression_model
        
        df = load_option_data()
        df = preprocess(df)
        feature_names = [
            'strike_price', 'underlying_price', 'option_type_encoded',
            'days_to_expiry', 'volatility', 'risk_free_rate', 'moneyness',
            'volatility_sq', 'days_to_expiry_sq'
        ]
        X = df[feature_names]
        y = df['trade_price']
        
        theta, scaling_params, prediction_df = run_gradient_descent_regression(
            df, alpha=alpha, lambda_=lambda_, num_iters=num_iters
        )
        
        model_info = {
            'alpha': alpha,
            'lambda': lambda_,
            'num_iters': num_iters,
            'n_features': len(theta) - 1,  
            'training_date': timezone.now().isoformat(),
            'feature_names': feature_names
        }
        save_ml_regression_model(theta, scaling_params, model_info)
        
        results = train_multiple_linear_regression(
            alpha=alpha,
            lambda_=lambda_,
            num_iters=num_iters
        )
        
        training_inputs = {
            'features': X.to_dict(orient='list'),
            'targets': y.tolist(),
            'trade_datetimes': df['trade_datetime'].astype(str).tolist(),
            'underlyings': df['underlying'].tolist(),
            'option_types': df['option_type'].tolist(),
        }
        
        summary_stats = results.get('summary_stats', {})
        training_result = ModelTrainingResult.objects.create(
            
        )
        
        results['training_result_id'] = training_result.id
        
        return Response(results, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Error training regression model: {str(e)}")
        logger.error(traceback.format_exc())
        return Response(
            {'error': f'Failed to train model: {str(e)}'}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
def train_neural_network_model(request):
    try:
        epochs = request.data.get('epochs', 100)
        batch_size = request.data.get('batch_size', 32)
        validation_split = request.data.get('validation_split', 0.2)

        from .neural_network import load_and_preprocess_data
        from .model_manager import save_neural_network_model
        
        X, y = load_and_preprocess_data()
        summary, model, scaler = train_neural_network(
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split
        )
        
        model_info = {
            'epochs': epochs,
            'batch_size': batch_size,
            'validation_split': validation_split,
            'n_features': X.shape[1],
            'training_date': timezone.now().isoformat(),
            'feature_names': list(X.columns) if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]
        }
        save_neural_network_model(model, scaler, model_info)
        
        training_inputs = {
            'features': X.to_dict(orient='list'),
            'targets': y.tolist(),
        }
        training_result = ModelTrainingResult.objects.create(
            model_type='neural_network',
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            mean_absolute_error=summary['mae'],
            mean_squared_error=summary['mse'],
            root_mean_squared_error=summary['rmse'],
            r2_score=summary['r2'],
            mean_absolute_percentage_error=summary['mape'],
            residual_std=summary['residual_std'],
            max_overprediction=summary['max_overprediction'],
            max_underprediction=summary['max_underprediction'],
            n_training_samples=summary.get('n_training_samples', 0),
            n_test_samples=summary['n_test_samples'],
            training_epochs=summary['training_epochs'],
            final_training_loss=summary['final_training_loss'],
            final_validation_loss=summary['final_validation_loss'],
            training_inputs=training_inputs
        )
        
        summary['training_result_id'] = training_result.id
        
        return Response(summary, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Error training neural network model: {str(e)}")
        logger.error(traceback.format_exc())
        return Response(
            {'error': f'Failed to train neural network model: {str(e)}'}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
def black_scholes_batch(request):
    queryset = TestingOptionData.objects.all().values()
    df = pd.DataFrame(list(queryset))
    if df.empty:
        return Response({"error": "No data available."}, status=400)

    df = df.dropna(subset=['trade_price', 'strike_price', 'trade_datetime', 'expiration_date', 'underlying'])
    df = df[df['trade_price'] > 0]
    df['trade_datetime'] = pd.to_datetime(df['trade_datetime']).dt.tz_localize(None)
    df['expiration_date'] = pd.to_datetime(df['expiration_date']).dt.tz_localize(None)
    df['days_to_expiry'] = (df['expiration_date'] - df['trade_datetime']).dt.days
    df['option_type_encoded'] = df['option_type'].map({'C': 1, 'P': 0, 'call': 1, 'put': 0})

    unique_underlyings = set(df['underlying'].unique())
    all_tickers = set()
    for underlying in unique_underlyings:
        base_symbol = underlying.upper().replace('$', '')
        primary = ticker_symbols.get(base_symbol, base_symbol)
        fallback = fallback_map.get(primary)
        all_tickers.add(primary)
        if fallback:
            all_tickers.add(fallback)
    from .multiple_linear_regression import bulk_fetch_price_history, get_price_from_cache, get_vol_from_cache
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

    df['underlying_price'] = df.apply(lookup_price, axis=1)
    df['volatility'] = df.apply(lookup_vol, axis=1)
    df['risk_free_rate'] = df.apply(lambda row: get_risk_free_rate(row['trade_datetime']), axis=1)
    df = df.dropna(subset=['underlying_price', 'volatility', 'risk_free_rate'])

    df['T'] = df['days_to_expiry'] / 365.0
    df = df[df['T'] > 0]

    def compute_bs_price(row):
        S = float(row['underlying_price'])
        K = float(row['strike_price'])
        T = float(row['T'])
        r = float(row['risk_free_rate'])
        sigma = float(row['volatility'])
        if row['option_type'] in ['C', 'call']:
            return BlackScholesCalculator.call_price(S, K, T, r, sigma)
        else:
            return BlackScholesCalculator.put_price(S, K, T, r, sigma)

    df['bs_price'] = df.apply(compute_bs_price, axis=1)
    df['residual'] = df['trade_price'] - df['bs_price']
    df['residual_pct'] = (df['residual'] / df['trade_price']) * 100
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    summary_stats = {
        "mean_absolute_error": float(np.abs(df['residual']).mean()),
        "mean_absolute_percentage_error": float(np.abs(df['residual_pct']).mean()),
        "std_residuals": float(df['residual'].std()),
        "max_overprediction": float(df['residual'].min()),
        "max_underprediction": float(df['residual'].max()),
    }

    results = df[['trade_datetime', 'underlying', 'option_type', 'strike_price', 'days_to_expiry', 'trade_price', 'bs_price', 'residual', 'residual_pct']].head(100).to_dict(orient="records")

    features = [
        'strike_price', 'underlying_price', 'option_type_encoded',
        'days_to_expiry', 'volatility', 'risk_free_rate', 'T'
    ]
    training_inputs = {
        'features': df[features].to_dict(orient='list'),
        'targets': df['trade_price'].tolist(),
        'bs_prices': df['bs_price'].tolist(),
        'trade_datetimes': df['trade_datetime'].astype(str).tolist(),
        'underlyings': df['underlying'].tolist(),
        'option_types': df['option_type'].tolist(),
    }
    training_result = ModelTrainingResult.objects.create(
        model_type='black_scholes',
        mean_absolute_error=summary_stats['mean_absolute_error'],
        mean_squared_error=float(np.mean(df['residual'] ** 2)),
        root_mean_squared_error=float(np.sqrt(np.mean(df['residual'] ** 2))),
        r2_score=float(1 - np.sum((df['trade_price'] - df['bs_price']) ** 2) / np.sum((df['trade_price'] - np.mean(df['trade_price'])) ** 2)),
        mean_absolute_percentage_error=summary_stats['mean_absolute_percentage_error'],
        residual_std=summary_stats['std_residuals'],
        max_overprediction=summary_stats['max_overprediction'],
        max_underprediction=summary_stats['max_underprediction'],
        n_training_samples=0,
        n_test_samples=len(df),
        training_inputs=training_inputs
    )

    return Response({
        "summary_stats": summary_stats,
        "results": results,
        "row_count": len(df),
        "training_result_id": training_result.id
    })


@api_view(['POST'])
def predict_option_price(request):
    try:
        required_fields = ['symbol', 'option_type', 'strike_price', 'expiration_date', 
                          'underlying_price', 'risk_free_rate', 'volatility']
        
        for field in required_fields:
            if field not in request.data:
                return Response(
                    {'error': f'Missing required field: {field}'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
        
        data = request.data
        symbol = data['symbol']
        option_type = data['option_type']
        strike_price = float(data['strike_price'])
        expiration_date = parse_dateutil(data['expiration_date']).date()
        underlying_price = float(data['underlying_price'])
        risk_free_rate = float(data['risk_free_rate'])
        volatility = float(data['volatility'])
        
        today = date.today()
        T = (expiration_date - today).days / 365.0
        
        if T <= 0:
            return Response(
                {'error': 'Expiration date must be in the future'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        results = {}
        
        if option_type == 'call':
            bs_price = BlackScholesCalculator.call_price(underlying_price, strike_price, T, risk_free_rate, volatility)
        else:  
            bs_price = BlackScholesCalculator.put_price(underlying_price, strike_price, T, risk_free_rate, volatility)
        
        results['black_scholes'] = {
            'price': bs_price,
            'model_type': 'Black-Scholes',
            'confidence': 'High (theoretical model)'
        }
        
        try:
            from .model_manager import predict_ml_regression, get_model_status
            
            option_data = {
                'strike_price': strike_price,
                'underlying_price': underlying_price,
                'option_type_encoded': 1 if option_type == 'call' else 0,
                'days_to_expiry': T * 365,
                'volatility': volatility,
                'risk_free_rate': risk_free_rate,
                'moneyness': underlying_price / strike_price,
                'volatility_sq': volatility ** 2,
                'days_to_expiry_sq': (T * 365) ** 2
            }
            
            ml_price = predict_ml_regression(option_data)
            
            if ml_price is not None:
                regression_results = ModelTrainingResult.objects.filter(
                    model_type='linear_regression'
                ).order_by('mean_absolute_error', 'root_mean_squared_error', '-training_date').first()
                
                results['ml_regression'] = {
                    'price': ml_price,
                    'model_type': 'ML Regression',
                    'confidence': 'Medium (trained on historical data)',
                    'model_performance': {
                        'r2_score': regression_results.r2_score if regression_results else 0,
                        'mae': regression_results.mean_absolute_error if regression_results else 0
                    }
                }
            else:
                results['ml_regression'] = {
                    'price': None,
                    'model_type': 'ML Regression',
                    'confidence': 'Not available (no trained model)',
                    'error': 'Please train the ML regression model first'
                }
        except Exception as e:
            results['ml_regression'] = {
                'price': None,
                'model_type': 'ML Regression',
                'confidence': 'Error',
                'error': str(e)
            }
        
        try:
            from .model_manager import predict_neural_network
            
            option_data = {
                'strike_price': strike_price,
                'underlying_price': underlying_price,
                'option_type_encoded': 1 if option_type == 'call' else 0,
                'days_to_expiry': T * 365,
                'volatility': volatility,
                'risk_free_rate': risk_free_rate,
                'moneyness': underlying_price / strike_price,
                'volatility_sq': volatility ** 2,
                'days_to_expiry_sq': (T * 365) ** 2,
                'time_to_expiry_years': T,
                'log_moneyness': np.log(underlying_price / strike_price),
                'volatility_time': volatility * np.sqrt(T)
            }
            
            nn_price = predict_neural_network(option_data)
            
            if nn_price is not None:
                nn_results = ModelTrainingResult.objects.filter(
                    model_type='neural_network'
                ).order_by('mean_absolute_error', 'root_mean_squared_error', '-training_date').first()
                
                results['neural_network'] = {
                    'price': nn_price,
                    'model_type': 'Neural Network',
                    'confidence': 'Medium (trained on historical data)',
                    'model_performance': {
                        'r2_score': nn_results.r2_score if nn_results else 0,
                        'mae': nn_results.mean_absolute_error if nn_results else 0
                    }
                }
            else:
                results['neural_network'] = {
                    'price': None,
                    'model_type': 'Neural Network',
                    'confidence': 'Not available (no trained model)',
                    'error': 'Please train the neural network model first'
                }
        except Exception as e:
            results['neural_network'] = {
                'price': None,
                'model_type': 'Neural Network',
                'confidence': 'Error',
                'error': str(e)
            }
        
        return Response({
            'option_details': {
                'symbol': symbol,
                'option_type': option_type,
                'strike_price': strike_price,
                'expiration_date': expiration_date.isoformat(),
                'underlying_price': underlying_price,
                'risk_free_rate': risk_free_rate,
                'volatility': volatility,
                'time_to_expiration': T
            },
            'predictions': results
        })
        
    except Exception as e:
        logger.error(f"Error in predict_option_price: {str(e)}")
        logger.error(traceback.format_exc())
        return Response(
            {'error': f'An error occurred: {str(e)}'}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


