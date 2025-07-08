import React, { useState } from 'react';
import OptionCalculator from './components/OptionCalculator';
import CSVUpload from './components/CSVUpload';
import TestingDataList from './components/TestingDataList';
import MultipleLinearRegression from './components/MultipleLinearRegression';
import BlackScholesResults from './components/BlackScholesResults';
import NeuralNetworkTraining from './components/NeuralNetworkTraining';
import ModelComparison from './components/ModelComparison';
import './App.css';

function App() {
  const [activeTab, setActiveTab] = useState<'calculator' | 'upload' | 'testing' | 'regression' | 'black-scholes-results' | 'neural-network' | 'model-comparison'>('calculator');

  return (
    <div className="App">
      <header className="App-header">
        <h1>Options Pricing Program</h1>
        <p>Options Pricing Using Black-Scholes Model, Multiple Linear Regression, and Neural Networks</p>
      </header>

      <nav className="App-nav">
        <button
          className={`nav-btn ${activeTab === 'calculator' ? 'active' : ''}`}
          onClick={() => setActiveTab('calculator')}
        >
          Calculator
        </button>
        <button
          className={`nav-btn ${activeTab === 'upload' ? 'active' : ''}`}
          onClick={() => setActiveTab('upload')}
        >
          Upload Data
        </button>
        <button
          className={`nav-btn ${activeTab === 'testing' ? 'active' : ''}`}
          onClick={() => setActiveTab('testing')}
        >
          Testing Data
        </button>
        <button
          className={`nav-btn ${activeTab === 'regression' ? 'active' : ''}`}
          onClick={() => setActiveTab('regression')}
        >
          ML Regression
        </button>
        <button
          className={`nav-btn ${activeTab === 'black-scholes-results' ? 'active' : ''}`}
          onClick={() => setActiveTab('black-scholes-results')}
        >
          Black-Scholes Results
        </button>
        <button
          className={`nav-btn ${activeTab === 'neural-network' ? 'active' : ''}`}
          onClick={() => setActiveTab('neural-network')}
        >
          Neural Network
        </button>
        <button
          className={`nav-btn ${activeTab === 'model-comparison' ? 'active' : ''}`}
          onClick={() => setActiveTab('model-comparison')}
        >
          Model Comparison
        </button>
      </nav>

      <main className="App-main">
        {activeTab === 'calculator' ? <OptionCalculator /> : 
         activeTab === 'upload' ? <CSVUpload /> :
         activeTab === 'testing' ? <TestingDataList /> :
         activeTab === 'regression' ? <MultipleLinearRegression /> :
         activeTab === 'black-scholes-results' ? <BlackScholesResults /> :
         activeTab === 'neural-network' ? <NeuralNetworkTraining /> :
         <ModelComparison />}
      </main>

      <footer className="App-footer">
        <p>&copy; 2025 Options Pricing Program. Built with React, Django, and PostgreSQL.</p>
      </footer>
    </div>
  );
}

export default App;
