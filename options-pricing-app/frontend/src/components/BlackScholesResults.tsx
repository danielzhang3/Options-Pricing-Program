import React, { useState } from 'react';
import { optionsApi, BlackScholesBatchResult } from '../api';
import './MultipleLinearRegression.css';

const BlackScholesResults: React.FC = () => {
  const [results, setResults] = useState<BlackScholesBatchResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchResults = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await optionsApi.getBlackScholesBatchResults();
      setResults(data);
    } catch (err: any) {
      setError(err.message || 'Failed to fetch Black-Scholes results');
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

  const formatPercentage = (value: number) => {
    return `${value.toFixed(2)}%`;
  };

  return (
    <div className="mlr-container">
      <h2>Black-Scholes Model Results</h2>
      <p>Evaluate Black-Scholes pricing against your uploaded options data</p>

      <div className="training-section">
        <h3>Run Black-Scholes Batch</h3>
        <div className="training-controls">
          <button onClick={fetchResults} disabled={loading} className="train-btn">
            {loading ? 'Loading...' : 'Run Black-Scholes'}
          </button>
        </div>
      </div>

      {error && (
        <div className="error-message">
          <p>{error}</p>
        </div>
      )}

      {results && (
        <div className="results-section">
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
              <div className="stat-card">
                <h4>Total Records</h4>
                <div className="stat-value">{results.row_count}</div>
              </div>
            </div>
          </div>

          {/* Predictions Table */}
          <div className="predictions-section">
            <h3>Sample Black-Scholes Predictions</h3>
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
                    <th>BS Price</th>
                    <th>Residual</th>
                    <th>Residual %</th>
                  </tr>
                </thead>
                <tbody>
                  {results.results.slice(0, 20).map((pred, index) => (
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
                      <td>{formatCurrency(pred.bs_price)}</td>
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

export default BlackScholesResults; 