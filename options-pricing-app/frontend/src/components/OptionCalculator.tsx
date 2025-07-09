import React, { useState } from 'react';
import { optionsApi, OptionCalculationRequest, OptionPredictionResponse } from '../api';
import './OptionCalculator.css';

const OptionCalculator: React.FC = () => {
  const [formData, setFormData] = useState<OptionCalculationRequest>({
    symbol: 'AAPL',
    option_type: 'call',
    strike_price: 150,
    expiration_date: '',
    underlying_price: 150,
    risk_free_rate: 0.0424,
    volatility: 0.26,
  });

  const [result, setResult] = useState<OptionPredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: name === 'option_type' ? value : 
              ['strike_price', 'underlying_price', 'risk_free_rate', 'volatility'].includes(name) 
                ? parseFloat(value) || 0 : value
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const response = await optionsApi.predictOptionPrice(formData);
      setResult(response);
    } catch (err: any) {
      setError(err.response?.data?.error || 'An error occurred while calculating the option price');
    } finally {
      setLoading(false);
    }
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(value);
  };

  const formatNumber = (value: number, decimals: number = 4) => {
    return value.toFixed(decimals);
  };

  return (
    <div className="option-calculator">
      <h2>Options Pricing Calculator</h2>
      <p>Calculate option prices using Black-Scholes, ML Regression, and Neural Network models</p>

      <form onSubmit={handleSubmit} className="calculator-form">
        <div className="form-row">
          <div className="form-group">
            <label htmlFor="symbol">Symbol:</label>
            <input
              type="text"
              id="symbol"
              name="symbol"
              value={formData.symbol}
              onChange={handleInputChange}
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="option_type">Option Type:</label>
            <select
              id="option_type"
              name="option_type"
              value={formData.option_type}
              onChange={handleInputChange}
              required
            >
              <option value="call">Call</option>
              <option value="put">Put</option>
            </select>
          </div>
        </div>

        <div className="form-row">
          <div className="form-group">
            <label htmlFor="strike_price">Strike Price ($):</label>
            <input
              type="number"
              id="strike_price"
              name="strike_price"
              value={formData.strike_price}
              onChange={handleInputChange}
              step="0.01"
              min="0"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="underlying_price">Underlying Price ($):</label>
            <input
              type="number"
              id="underlying_price"
              name="underlying_price"
              value={formData.underlying_price}
              onChange={handleInputChange}
              step="0.01"
              min="0"
              required
            />
          </div>
        </div>

        <div className="form-row">
          <div className="form-group">
            <label htmlFor="expiration_date">Expiration Date:</label>
            <input
              type="date"
              id="expiration_date"
              name="expiration_date"
              value={formData.expiration_date}
              onChange={handleInputChange}
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="risk_free_rate">Risk-Free Rate:</label>
            <input
              type="number"
              id="risk_free_rate"
              name="risk_free_rate"
              value={formData.risk_free_rate}
              onChange={handleInputChange}
              step="0.0001"
              min="0"
              max="1"
              required
            />
          </div>
        </div>

        <div className="form-row">
          <div className="form-group">
            <label htmlFor="volatility">Volatility:</label>
            <input
              type="number"
              id="volatility"
              name="volatility"
              value={formData.volatility}
              onChange={handleInputChange}
              step="0.01"
              min="0"
              max="1"
              required
            />
          </div>
        </div>

        <button type="submit" disabled={loading} className="calculate-btn">
          {loading ? 'Calculating...' : 'Calculate Option Price'}
        </button>
      </form>

      {error && (
        <div className="error-message">
          <p>{error}</p>
        </div>
      )}

      {result && (
        <div className="results">
          <h3>Multi-Model Prediction Results</h3>
          
          <div className="option-details">
            <h4>Option Details</h4>
            <div className="details-grid">
              <div className="detail-item">
                <span className="detail-label">Symbol:</span>
                <span className="detail-value">{result.option_details.symbol}</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">Type:</span>
                <span className="detail-value">{result.option_details.option_type.toUpperCase()}</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">Strike:</span>
                <span className="detail-value">{formatCurrency(result.option_details.strike_price)}</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">Expiration:</span>
                <span className="detail-value">{new Date(result.option_details.expiration_date).toLocaleDateString()}</span>
              </div>
              <div className="detail-item">
                <span className="detail-label">Time to Expiration:</span>
                <span className="detail-value">{formatNumber(result.option_details.time_to_expiration * 365)} days</span>
              </div>
            </div>
          </div>

          <div className="predictions-section">
            <h4>Model Predictions</h4>
            <div className="predictions-grid">
              <div className="prediction-card">
                <h5>{result.predictions.black_scholes.model_type}</h5>
                <div className="prediction-price">
                  {result.predictions.black_scholes.price !== null ? (
                    formatCurrency(result.predictions.black_scholes.price)
                  ) : (
                    <span className="error-text">Not available</span>
                  )}
                </div>
                <div className="prediction-confidence">
                  {result.predictions.black_scholes.confidence}
                </div>
              </div>

              <div className="prediction-card">
                <h5>{result.predictions.ml_regression.model_type}</h5>
                <div className="prediction-price">
                  {result.predictions.ml_regression.price !== null ? (
                    formatCurrency(result.predictions.ml_regression.price)
                  ) : (
                    <span className="error-text">
                      {result.predictions.ml_regression.error || 'Not available'}
                    </span>
                  )}
                </div>
                <div className="prediction-confidence">
                  {result.predictions.ml_regression.confidence}
                </div>
                {result.predictions.ml_regression.model_performance && (
                  <div className="model-performance">
                    <small>R²: {formatNumber(result.predictions.ml_regression.model_performance.r2_score, 3)}</small>
                    <small>MAE: {formatNumber(result.predictions.ml_regression.model_performance.mae, 4)}</small>
                  </div>
                )}
              </div>

              <div className="prediction-card">
                <h5>{result.predictions.neural_network.model_type}</h5>
                <div className="prediction-price">
                  {result.predictions.neural_network.price !== null ? (
                    formatCurrency(result.predictions.neural_network.price)
                  ) : (
                    <span className="error-text">
                      {result.predictions.neural_network.error || 'Not available'}
                    </span>
                  )}
                </div>
                <div className="prediction-confidence">
                  {result.predictions.neural_network.confidence}
                </div>
                {result.predictions.neural_network.model_performance && (
                  <div className="model-performance">
                    <small>R²: {formatNumber(result.predictions.neural_network.model_performance.r2_score, 3)}</small>
                    <small>MAE: {formatNumber(result.predictions.neural_network.model_performance.mae, 4)}</small>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default OptionCalculator; 