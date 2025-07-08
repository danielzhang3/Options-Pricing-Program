import React, { useState } from 'react';
import { optionsApi, NeuralNetworkTrainingParams, NeuralNetworkTrainingResult } from '../api';
import './NeuralNetworkTraining.css';

const NeuralNetworkTraining: React.FC = () => {
  const [isTraining, setIsTraining] = useState(false);
  const [results, setResults] = useState<NeuralNetworkTrainingResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [trainingParams, setTrainingParams] = useState<NeuralNetworkTrainingParams>({
    epochs: 100,
    batch_size: 32,
    validation_split: 0.2,
  });

  const handleParamChange = (field: keyof NeuralNetworkTrainingParams, value: number) => {
    setTrainingParams(prev => ({
      ...prev,
      [field]: value,
    }));
  };

  const handleTrain = async () => {
    setIsTraining(true);
    setError(null);
    setResults(null);

    try {
      const result = await optionsApi.trainNeuralNetwork(trainingParams);
      setResults(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred during training');
    } finally {
      setIsTraining(false);
    }
  };

  return (
    <div className="neural-network-training">
      <h2>Neural Network Training</h2>
      
      <div className="training-form">
        <h3>Training Parameters</h3>
        <div className="form-group">
          <label htmlFor="epochs">Epochs:</label>
          <input
            type="number"
            id="epochs"
            value={trainingParams.epochs}
            onChange={(e) => handleParamChange('epochs', parseInt(e.target.value) || 100)}
            min="1"
            max="1000"
          />
        </div>
        
        <div className="form-group">
          <label htmlFor="batch_size">Batch Size:</label>
          <input
            type="number"
            id="batch_size"
            value={trainingParams.batch_size}
            onChange={(e) => handleParamChange('batch_size', parseInt(e.target.value) || 32)}
            min="1"
            max="128"
          />
        </div>
        
        <div className="form-group">
          <label htmlFor="validation_split">Validation Split:</label>
          <input
            type="number"
            id="validation_split"
            value={trainingParams.validation_split}
            onChange={(e) => handleParamChange('validation_split', parseFloat(e.target.value) || 0.2)}
            min="0.1"
            max="0.5"
            step="0.1"
          />
        </div>
        
        <button 
          onClick={handleTrain} 
          disabled={isTraining}
          className="train-button"
        >
          {isTraining ? 'Training...' : 'Train Neural Network'}
        </button>
      </div>

      {error && (
        <div className="error-message">
          <h3>Error</h3>
          <p>{error}</p>
        </div>
      )}

      {results && (
        <div className="training-results">
          <h3>Training Results</h3>
          
          <div className="metrics-grid">
            <div className="metric-card">
              <h4>Mean Absolute Error</h4>
              <p className="metric-value">${results.mae.toFixed(4)}</p>
            </div>
            
            <div className="metric-card">
              <h4>Root Mean Square Error</h4>
              <p className="metric-value">${results.rmse.toFixed(4)}</p>
            </div>
            
            <div className="metric-card">
              <h4>R² Score</h4>
              <p className="metric-value">{(results.r2 * 100).toFixed(2)}%</p>
            </div>
            
            <div className="metric-card">
              <h4>Mean Absolute Percentage Error</h4>
              <p className="metric-value">{results.mape.toFixed(2)}%</p>
            </div>
            
            <div className="metric-card">
              <h4>Residual Standard Deviation</h4>
              <p className="metric-value">${results.residual_std.toFixed(4)}</p>
            </div>
            
            <div className="metric-card">
              <h4>Test Samples</h4>
              <p className="metric-value">{results.n_test_samples}</p>
            </div>
          </div>
          
          <div className="additional-metrics">
            <h4>Additional Metrics</h4>
            <div className="metrics-row">
              <div className="metric-item">
                <span className="metric-label">Max Overprediction:</span>
                <span className="metric-value">${results.max_overprediction.toFixed(4)}</span>
              </div>
              <div className="metric-item">
                <span className="metric-label">Max Underprediction:</span>
                <span className="metric-value">${results.max_underprediction.toFixed(4)}</span>
              </div>
            </div>
            <div className="metrics-row">
              <div className="metric-item">
                <span className="metric-label">Training Epochs:</span>
                <span className="metric-value">{results.training_epochs}</span>
              </div>
              <div className="metric-item">
                <span className="metric-label">Final Training Loss:</span>
                <span className="metric-value">{results.final_training_loss.toFixed(6)}</span>
              </div>
              <div className="metric-item">
                <span className="metric-label">Final Validation Loss:</span>
                <span className="metric-value">{results.final_validation_loss.toFixed(6)}</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default NeuralNetworkTraining; 