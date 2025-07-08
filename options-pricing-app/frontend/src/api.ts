import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface OptionCalculationRequest {
  symbol: string;
  option_type: 'call' | 'put';
  strike_price: number;
  expiration_date: string;
  underlying_price: number;
  risk_free_rate: number;
  volatility: number;
}



export interface MarketData {
  id: number;
  symbol: string;
  price: string;
  volume: number;
  timestamp: string;
}

export interface BlackScholesBatchResult {
  summary_stats: {
    mean_absolute_error: number;
    mean_absolute_percentage_error: number;
    std_residuals: number;
    max_overprediction: number;
    max_underprediction: number;
  };
  results: Array<{
    trade_datetime: string;
    underlying: string;
    option_type: string;
    strike_price: number;
    days_to_expiry: number;
    trade_price: number;
    bs_price: number;
    residual: number;
    residual_pct: number;
  }>;
  row_count: number;
}

export interface NeuralNetworkTrainingParams {
  epochs?: number;
  batch_size?: number;
  validation_split?: number;
}

export interface NeuralNetworkTrainingResult {
  mae: number;
  mse: number;
  rmse: number;
  r2: number;
  mape: number;
  residual_std: number;
  max_overprediction: number;
  max_underprediction: number;
  n_test_samples: number;
  training_epochs: number;
  final_training_loss: number;
  final_validation_loss: number;
}

export interface ModelPrediction {
  price: number | null;
  model_type: string;
  confidence: string;
  model_performance?: {
    r2_score: number;
    mae: number;
  };
  error?: string;
}

export interface OptionPredictionResponse {
  option_details: {
    symbol: string;
    option_type: string;
    strike_price: number;
    expiration_date: string;
    underlying_price: number;
    risk_free_rate: number;
    volatility: number;
    time_to_expiration: number;
  };
  predictions: {
    black_scholes: ModelPrediction;
    ml_regression: ModelPrediction;
    neural_network: ModelPrediction;
  };
}

export const optionsApi = {

  // Get market data
  getMarketData: async (): Promise<MarketData[]> => {
    const response = await api.get('/market-data/');
    return response.data;
  },

  // Get latest market data for symbol
  getLatestMarketData: async (symbol: string): Promise<MarketData> => {
    const response = await api.get(`/market-data/${symbol}/latest/`);
    return response.data;
  },

  // Get Black-Scholes batch results
  getBlackScholesBatchResults: async (): Promise<BlackScholesBatchResult> => {
    const response = await api.get('/black-scholes-batch/');
    return response.data;
  },

  // Train neural network model
  trainNeuralNetwork: async (params: NeuralNetworkTrainingParams = {}): Promise<NeuralNetworkTrainingResult> => {
    const response = await api.post('/train-neural-network/', params);
    return response.data;
  },

  // Predict option price using all models
  predictOptionPrice: async (data: OptionCalculationRequest): Promise<OptionPredictionResponse> => {
    const response = await api.post('/predict/', data);
    return response.data;
  },
}; 