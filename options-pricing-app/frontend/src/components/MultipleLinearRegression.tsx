import React, { useState, useEffect } from 'react';
import './MultipleLinearRegression.css';

interface RegressionResults {
  r2_score: number;
  mse: number;
  coefficients: Array<{ feature: string; coefficient: number }>;
  predictions: Array<{
    trade_datetime: string;
    underlying: string;
    option_type: string;
    strike_price: number;
    days_to_expiry: number;
    trade_price: number;
    predicted_price: number;
    residual: number;
    residual_pct: number;
  }>;
  summary_stats: {
    mean_absolute_error: number;
    mean_absolute_percentage_error: number;
    std_residuals: number;
    max_overprediction: number;
    max_underprediction: number;
  };
}

const MultipleLinearRegression: React.FC = () => {
  const [isTraining, setIsTraining] = useState(false);
  const [results, setResults] = useState<RegressionResults | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [trainingProgress, setTrainingProgress] = useState<string>('');

  const trainModel = async () => {
    setIsTraining(true);
    setError(null);
    setTrainingProgress('Loading data...');

    try {
      const response = await fetch('http://localhost:8000/api/train-regression/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          alpha: 0.01,
          lambda_: 0.1,
          num_iters: 1000
        })
      });

      if (!response.ok) {
        throw new Error('Failed to train model');
      }

      const data = await response.json();
      setResults(data);
      setTrainingProgress('Training completed!');
    } catch (err: any) {
      setError(err.message);
      setTrainingProgress('');
    } finally {
      setIsTraining(false);
    }
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(value);
  };

  const formatPercentage = (value: number) => {
    return `${value.toFixed(2)}%`;
  };

  return (
    <div className="mlr-container">
      <h2>Multiple Linear Regression Model</h2>
      <p>Train and evaluate a multiple linear regression model for options pricing</p>

      {/* Training Section */}
      <div className="training-section">
        <h3>Model Training</h3>
        <div className="training-controls">
          <button 
            onClick={trainModel} 
            disabled={isTraining}
            className="train-btn"
          >
            {isTraining ? 'Training...' : 'Train Model'}
          </button>
          {trainingProgress && (
            <div className="training-progress">
              {trainingProgress}
            </div>
          )}
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="error-message">
          <p>{error}</p>
        </div>
      )}

      {/* Results Display */}
      {results && (
        <div className="results-section">
          {/* Model Performance */}
          <div className="performance-metrics">
            <h3>Model Performance</h3>
            <div className="metrics-grid">
              <div className="metric-card">
                <h4>R² Score</h4>
                <div className="metric-value">{results.r2_score.toFixed(4)}</div>
              </div>
              <div className="metric-card">
                <h4>Mean Squared Error</h4>
                <div className="metric-value">{results.mse.toFixed(4)}</div>
              </div>
            </div>
          </div>

          {/* Feature Coefficients */}
          <div className="coefficients-section">
            <h3>Feature Coefficients</h3>
            <div className="coefficients-table">
              <table>
                <thead>
                  <tr>
                    <th>Feature</th>
                    <th>Coefficient</th>
                  </tr>
                </thead>
                <tbody>
                  {results.coefficients.map((coef, index) => (
                    <tr key={index}>
                      <td>{coef.feature}</td>
                      <td className={coef.coefficient >= 0 ? 'positive' : 'negative'}>
                        {coef.coefficient.toFixed(6)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Summary Statistics */}
          <div className="summary-stats">
            <h3>Prediction Summary</h3>
            <div className="stats-grid">
              <div className="stat-card">
                <h4>Mean Absolute Error</h4>
                <div className="stat-value">{formatCurrency(results.summary_stats.mean_absolute_error)}</div>
              </div>
              <div className="stat-card">
                <h4>Mean Absolute % Error</h4>
                <div className="stat-value">{formatPercentage(results.summary_stats.mean_absolute_percentage_error)}</div>
              </div>
              <div className="stat-card">
                <h4>Std Dev of Residuals</h4>
                <div className="stat-value">{formatCurrency(results.summary_stats.std_residuals)}</div>
              </div>
              <div className="stat-card">
                <h4>Max Overprediction</h4>
                <div className="stat-value negative">{formatCurrency(results.summary_stats.max_overprediction)}</div>
              </div>
              <div className="stat-card">
                <h4>Max Underprediction</h4>
                <div className="stat-value positive">{formatCurrency(results.summary_stats.max_underprediction)}</div>
              </div>
            </div>
          </div>

          {/* Predictions Table */}
          <div className="predictions-section">
            <h3>Sample Predictions</h3>
            <div className="predictions-table">
              <table>
                <thead>
                  <tr>
                    <th>Date</th>
                    <th>Underlying</th>
                    <th>Type</th>
                    <th>Strike</th>
                    <th>Days to Expiry</th>
                    <th>Actual Price</th>
                    <th>Predicted Price</th>
                    <th>Residual</th>
                    <th>Residual %</th>
                  </tr>
                </thead>
                <tbody>
                  {results.predictions.slice(0, 20).map((pred, index) => (
                    <tr key={index}>
                      <td>{new Date(pred.trade_datetime).toLocaleDateString()}</td>
                      <td>{pred.underlying}</td>
                      <td>
                        <span className={`option-type ${pred.option_type.toLowerCase()}`}>
                          {pred.option_type}
                        </span>
                      </td>
                      <td>{formatCurrency(pred.strike_price)}</td>
                      <td>{pred.days_to_expiry}</td>
                      <td>{formatCurrency(pred.trade_price)}</td>
                      <td>{formatCurrency(pred.predicted_price)}</td>
                      <td className={pred.residual >= 0 ? 'positive' : 'negative'}>
                        {formatCurrency(pred.residual)}
                      </td>
                      <td className={pred.residual_pct >= 0 ? 'positive' : 'negative'}>
                        {formatPercentage(pred.residual_pct)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default MultipleLinearRegression; 