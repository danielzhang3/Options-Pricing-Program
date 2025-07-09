import React, { useState, useEffect } from 'react';
import './ModelComparison.css';

interface ModelTrainingResult {
  id: number;
  model_type: 'black_scholes' | 'linear_regression' | 'neural_network';
  model_type_display: string;
  training_date: string;
  mean_absolute_error: number;
  mean_squared_error: number;
  root_mean_squared_error: number;
  r2_score: number;
  mean_absolute_percentage_error: number;
  residual_std: number;
  max_overprediction: number;
  max_underprediction: number;
  n_training_samples: number;
  n_test_samples: number;
  training_epochs?: number;
  final_training_loss?: number;
  final_validation_loss?: number;
}

const ModelComparison: React.FC = () => {
  const [results, setResults] = useState<ModelTrainingResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedModelType, setSelectedModelType] = useState<string>('all');

  useEffect(() => {
    fetchTrainingResults();
  }, []);

  const fetchTrainingResults = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://localhost:8000/api/training-results/');
      if (!response.ok) {
        throw new Error('Failed to fetch training results');
      }
      const data = await response.json();
      setResults(data.results || data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch results');
    } finally {
      setLoading(false);
    }
  };

  const getBestResults = () => {
    const bestResults: { [key: string]: ModelTrainingResult } = {};
    
    results.forEach(result => {
      const existing = bestResults[result.model_type];
      
      if (!existing) {
        bestResults[result.model_type] = result;
        return;
      }
      
      let betterMetrics = 0;
      
      if (result.mean_absolute_error < existing.mean_absolute_error) betterMetrics++;
      if (result.root_mean_squared_error < existing.root_mean_squared_error) betterMetrics++;
      if (result.r2_score > existing.r2_score) betterMetrics++;
      
      if (betterMetrics >= 2) {
        bestResults[result.model_type] = result;
      }
    });
    
    return Object.values(bestResults);
  };

  const filteredResults = selectedModelType === 'all' 
    ? getBestResults() 
    : getBestResults().filter((r: ModelTrainingResult) => r.model_type === selectedModelType);

  const getModelIcon = (modelType: string) => {
    switch (modelType) {
      case 'black_scholes':
        return '📊';
      case 'linear_regression':
        return '📈';
      case 'neural_network':
        return '🧠';
      default:
        return '📋';
    }
  };

  const getModelColor = (modelType: string) => {
    switch (modelType) {
      case 'black_scholes':
        return '#3498db';
      case 'linear_regression':
        return '#e74c3c';
      case 'neural_network':
        return '#9b59b6';
      default:
        return '#95a5a6';
    }
  };

  if (loading) {
    return (
      <div className="model-comparison">
        <h2>Model Comparison</h2>
        <div className="loading">Loading training results...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="model-comparison">
        <h2>Model Comparison</h2>
        <div className="error-message">
          <h3>Error</h3>
          <p>{error}</p>
          <button onClick={fetchTrainingResults} className="retry-button">
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (filteredResults.length === 0) {
    return (
      <div className="model-comparison">
        <h2>Model Comparison</h2>
        <div className="no-results">
          <p>No training results found. Train some models first!</p>
          <div className="model-tabs">
            <button 
              className={`model-tab ${selectedModelType === 'all' ? 'active' : ''}`}
              onClick={() => setSelectedModelType('all')}
            >
              All Models
            </button>
            <button 
              className={`model-tab ${selectedModelType === 'black_scholes' ? 'active' : ''}`}
              onClick={() => setSelectedModelType('black_scholes')}
            >
              Black-Scholes
            </button>
            <button 
              className={`model-tab ${selectedModelType === 'linear_regression' ? 'active' : ''}`}
              onClick={() => setSelectedModelType('linear_regression')}
            >
              ML Regression
            </button>
            <button 
              className={`model-tab ${selectedModelType === 'neural_network' ? 'active' : ''}`}
              onClick={() => setSelectedModelType('neural_network')}
            >
              Neural Network
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="model-comparison">
      <h2>Model Comparison</h2>
      
      <div className="model-tabs">
        <button 
          className={`model-tab ${selectedModelType === 'all' ? 'active' : ''}`}
          onClick={() => setSelectedModelType('all')}
        >
          All Models
        </button>
        <button 
          className={`model-tab ${selectedModelType === 'black_scholes' ? 'active' : ''}`}
          onClick={() => setSelectedModelType('black_scholes')}
        >
          Black-Scholes
        </button>
        <button 
          className={`model-tab ${selectedModelType === 'linear_regression' ? 'active' : ''}`}
          onClick={() => setSelectedModelType('linear_regression')}
        >
          ML Regression
        </button>
        <button 
          className={`model-tab ${selectedModelType === 'neural_network' ? 'active' : ''}`}
          onClick={() => setSelectedModelType('neural_network')}
        >
          Neural Network
        </button>
      </div>

      <div className="comparison-grid">
        {filteredResults.map((result) => (
          <div 
            key={result.id} 
            className="model-card"
            style={{ borderColor: getModelColor(result.model_type) }}
          >
            <div className="model-header">
              <span className="model-icon">{getModelIcon(result.model_type)}</span>
              <h3>{result.model_type_display}</h3>
              <span className="training-date">
                {new Date(result.training_date).toLocaleDateString()}
              </span>
            </div>

            <div className="metrics-grid">
              <div className="metric-item">
                <span className="metric-label">Mean Absolute Error</span>
                <span className="metric-value">${result.mean_absolute_error.toFixed(4)}</span>
              </div>
              
              <div className="metric-item">
                <span className="metric-label">R² Score</span>
                <span className="metric-value">{(result.r2_score * 100).toFixed(2)}%</span>
              </div>
              
              <div className="metric-item">
                <span className="metric-label">Mean Squared Error</span>
                <span className="metric-value">{result.mean_squared_error.toFixed(4)}</span>
              </div>
              
              <div className="metric-item">
                <span className="metric-label">Std of Residuals</span>
                <span className="metric-value">${result.residual_std.toFixed(4)}</span>
              </div>
              
              <div className="metric-item">
                <span className="metric-label">Root MSE</span>
                <span className="metric-value">${result.root_mean_squared_error.toFixed(4)}</span>
              </div>
              
              <div className="metric-item">
                <span className="metric-label">MAPE</span>
                <span className="metric-value">{result.mean_absolute_percentage_error.toFixed(2)}%</span>
              </div>
            </div>

            <div className="additional-info">
              <div className="info-row">
                <span>Test Samples: {result.n_test_samples}</span>
                {result.training_epochs && (
                  <span>Epochs: {result.training_epochs}</span>
                )}
              </div>
              <div className="info-row">
                <span>Max Over: ${result.max_overprediction.toFixed(4)}</span>
                <span>Max Under: ${result.max_underprediction.toFixed(4)}</span>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="refresh-section">
        <button onClick={fetchTrainingResults} className="refresh-button">
          Refresh Results
        </button>
        <p className="refresh-note">
          Showing best performing results for each model type. Click refresh to get updated results.
        </p>
      </div>
    </div>
  );
};

export default ModelComparison; 