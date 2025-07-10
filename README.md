# Options Pricing Program

A comprehensive options pricing application built with React TypeScript frontend, Django Python backend, and PostgreSQL database. The application implements the Black-Scholes option pricing model with full Greeks calculations, a Multiple Linear Regression, and a Neural Network with ReLU activation.

## Features

- **Black-Scholes Option Pricing**: Calculate option prices for both call and put options
- **Option Greeks**: Full calculation of Delta, Gamma, Theta, Vega, and Rho
- **Multiple Linear Regression**: Train and evaluate a regression model on your uploaded options data
- **Neural Network Option Pricing**: Deep learning-based model with SHAP interpretability for improved option price prediction
- **Batch Black-Scholes Evaluation**: Run Black-Scholes pricing against all uploaded options and compare to actual prices
- **Modern UI**: Beautiful, responsive React interface with TypeScript
- **RESTful API**: Django REST Framework backend with comprehensive endpoints
- **Database Storage**: PostgreSQL database with PgAdmin4 for management
- **Real-time Calculations**: Instant option pricing with validation
- **Option Management**: Save, view, and manage calculated options

## Technology Stack

### Frontend
- **React 18** with TypeScript
- **Axios** for API communication
- **CSS3** with modern styling and responsive design

### Backend
- **Django 5.2** with Python 3.12
- **Django REST Framework** for API endpoints
- **PostgreSQL** database
- **Black-Scholes** mathematical implementation
- **TensorFlow/Keras** for neural network implementation
- **SHAP** for model interpretability and feature importance analysis
- **scikit-learn** for multiple linear regression
- **yfinance** for market data fetching

### Database
- **PostgreSQL** for data persistence
- **PgAdmin4** for database management

## Project Structure

```
options-pricing-app/
├── backend/
│   ├── manage.py
│   ├── backend/                    # Django project
│   │   ├── settings.py
│   │   ├── urls.py
│   │   ├── asgi.py
│   │   └── wsgi.py
│   ├── api/                        # Django app
│   │   ├── models.py               # Database models
│   │   ├── views.py                # API views and Black-Scholes calculator
│   │   ├── urls.py                 # API URL routing
│   │   ├── serializers.py          # DRF serializers
│   │   ├── multiple_linear_regression.py  # ML regression implementation
│   │   ├── neural_network.py       # Neural network with SHAP interpretability
│   │   ├── stockutils.py           # Market data utilities
│   │   ├── model_manager.py        # Model comparison and evaluation utilities
│   │   └── migrations/             # Database migrations
│   ├── requirements.txt            # Python dependencies
│   ├── Dockerfile                  # Backend containerization
│   └── venv/                       # Python virtual environment
├── frontend/                       # React app
│   ├── src/
│   │   ├── components/
│   │   │   ├── OptionCalculator.tsx      # Black-Scholes calculator
│   │   │   ├── OptionCalculator.css
│   │   │   ├── CSVUpload.tsx             # Data upload interface
│   │   │   ├── CSVUpload.css
│   │   │   ├── TestingDataList.tsx       # Uploaded data display
│   │   │   ├── TestingDataList.css
│   │   │   ├── MultipleLinearRegression.tsx  # ML regression interface
│   │   │   ├── MultipleLinearRegression.css
│   │   │   ├── BlackScholesResults.tsx   # Batch Black-Scholes results
│   │   │   ├── BlackScholesResults.css
│   │   │   ├── NeuralNetworkTraining.tsx # Neural network training interface
│   │   │   ├── NeuralNetworkTraining.css
│   │   │   ├── ModelComparison.tsx       # Model comparison dashboard
│   │   │   └── ModelComparison.css
│   │   ├── App.tsx                 # Main application component
│   │   ├── App.css                 # Main application styles
│   │   └── api.ts                  # API client and TypeScript interfaces
│   ├── package.json                # Node.js dependencies
│   ├── Dockerfile                  # Frontend containerization
│   └── public/                     # Static assets
├── docker-compose.yml              # Multi-container orchestration
├── start.sh                        # Development startup script
├── QUICK_START.md                  # Quick setup guide
└── README.md                       # This file
```

### Key Components:

**Backend ML Models:**
- **Black-Scholes**: Classical options pricing with Greeks calculation
- **Multiple Linear Regression**: Statistical model for options pricing
- **Neural Network**: Deep learning model with SHAP interpretability
- **Model Manager**: Utilities for model comparison and evaluation

**Frontend Interfaces:**
- **Option Calculator**: Interactive Black-Scholes pricing
- **Data Management**: CSV upload and testing data display
- **Model Training**: Separate interfaces for each ML model
- **Model Comparison**: Dashboard for comparing model performance
- **Results Visualization**: Comprehensive results display with charts

## Installation and Setup

### Prerequisites

- Python 3.12+
- Node.js 18+
- PostgreSQL 12+
- PgAdmin4

### Backend Setup

1. **Navigate to backend directory:**
   ```bash
   cd options-pricing-app/backend
   ```

2. **Create and activate virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up PostgreSQL database:**
   - Create a database named `options_pricing_db`
   - Update database settings in `backend/settings.py` if needed

5. **Run Django migrations:**
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

6. **Create superuser (optional):**
   ```bash
   python manage.py createsuperuser
   ```

7. **Start Django development server:**
   ```bash
   python manage.py runserver
   ```

### Frontend Setup

1. **Navigate to frontend directory:**
   ```bash
   cd options-pricing-app/frontend
   ```

2. **Install Node.js dependencies:**
   ```bash
   npm install
   ```

3. **Start React development server:**
   ```bash
   npm start
   ```

## Usage

### Accessing the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000/api/
- **Django Admin**: http://localhost:8000/admin/
- **PgAdmin4**: http://localhost:5050 (if running locally)

### Using the Options Calculator

1. **Navigate to the Calculator tab**
2. **Fill in the option parameters:**
   - Symbol (e.g., AAPL, TSLA)
   - Option Type (Call or Put)
   - Strike Price
   - Expiration Date
   - Underlying Price
   - Risk-Free Rate
   - Volatility

3. **Click "Calculate Option Price"**
4. **View results including:**
   - Option Price
   - Time to Expiration
   - Option Greeks (Delta, Gamma, Theta, Vega, Rho)
   - Option Details
   - Real-time validation and error handling

### Using the Black-Scholes Batch Evaluation

1. **Navigate to the "Black-Scholes Results" tab**
2. **Click "Run Black-Scholes"** to compute Black-Scholes prices for all uploaded options
3. **View results including:**
   - Actual trade price
   - Black-Scholes price
   - Residuals and percent error
   - Summary statistics (MAE, MAPE, etc.)

### Using the Multiple Linear Regression Model

1. **Navigate to the "ML Regression" tab**
2. **Click "Train Model"** to train a regression model on your uploaded options data
3. **View results including:**
   - Model performance (R², MSE)
   - Feature coefficients
   - Sample predictions and residuals
   - Summary statistics

### Using the Neural Network Option Pricing Model

1. **Navigate to the "Neural Network" tab**
2. **Click "Train Model"** to build a deep learning model on your uploaded options data
3. **View results including:**
   - Neural network training progress (MAE, RMSE, R²)
   - Real-time predictions using TensorFlow
   - **SHAP feature importance analysis** with waterfall plots for sample predictions
   - Summary metrics and comparison to Black-Scholes and linear regression
   - **Model interpretability** showing how each feature contributes to predictions

### Comparing Models

- Use the "Black-Scholes Results," "ML Regression," and "Neural Network" tabs to compare model performance on your data
- Both tabs show summary statistics and sample predictions for easy comparison

## Model Evaluation and Comparison 

### 📊 Summary of Model Results

| Model                    | Mean Absolute Error | Std. Dev. of Residuals | R² Score |
|--------------------------|---------------------|-------------------------|----------|
| Black-Scholes            | $103.80             | $207.70                 | —        |
| Multiple Linear Regression | $56.62              | $105.17                 | 0.413    |
| Neural Network (TF/Keras) | **$14.73**          | **$28.68**              | **0.93** |

### 💡 Key Takeaway

Despite its theoretical elegance, the Black-Scholes model underperforms on real-world noisy data. Surprisingly, even a simple multiple linear regression model improved predictive accuracy —but the **best performance came from a custom-trained neural network**, which captured nonlinearities and interaction effects across key option features.

This confirms that modern machine learning methods can complement or outperform classical finance models when properly engineered and trained on realistic data.

**Note: The neural network results come from two related model configurations. The robust pricing model, optimized for practical accuracy, achieved an MAE of $15.53 and a residual standard deviation of $28.68. Meanwhile, the variance-explaining model, trained on the full dataset without outlier filtering, achieved an R² score of 0.93.**

### 🔍 Model Interpretability with SHAP

The neural network implementation includes **SHAP (SHapley Additive exPlanations)** analysis for model interpretability:

- **Feature Importance Analysis**: Understand which features contribute most to option price predictions
- **Waterfall Plots**: Visual representation of how each feature pushes predictions up or down
- **Sample Predictions**: Detailed analysis of individual option predictions with error breakdown
- **Cross-Platform Compatibility**: Non-interactive plotting for server environments

**Key Features:**
- **12 engineered features** including moneyness, volatility interactions, and time decay
- **Binary option type encoding** (1 for calls, 0 for puts)
- **Automatic SHAP analysis** during training for the first 3 test samples
- **High-quality plot generation** saved as PNG files for easy viewing

## API Endpoints

### Option Prediction
- `POST /api/predict/` - Predict option price using Black-Scholes, ML Regression, and Neural Network models

### Market Data
- `GET /api/market-data/` - List all market data
- `GET /api/market-data/{symbol}/latest/` - Get latest market data for symbol

### Black-Scholes Batch Evaluation
- `GET /api/black-scholes-batch/` - Compute Black-Scholes prices for all uploaded options and return results with summary statistics

### Multiple Linear Regression
- `POST /api/train-regression/` - Train and evaluate a regression model on all uploaded options data

### Neural Network Training
- `POST /api/train-neural-network/` - Train and evaluate a neural network model on all uploaded options data with SHAP interpretability analysis

## Mathematical Implementation

### Black-Scholes Model

The application implements the complete Black-Scholes option pricing model:

**Call Option Price:**
```
C = S * N(d1) - K * e^(-rT) * N(d2)
```

**Put Option Price:**
```
P = K * e^(-rT) * N(-d2) - S * N(-d1)
```

**Where:**
- `d1 = (ln(S/K) + (r + σ²/2)T) / (σ√T)`
- `d2 = d1 - σ√T`
- `S` = Current stock price
- `K` = Strike price
- `T` = Time to expiration
- `r` = Risk-free rate
- `σ` = Volatility
- `N()` = Cumulative normal distribution function

### Option Greeks

The application calculates all major option Greeks:

- **Delta**: Rate of change of option price with respect to underlying price
- **Gamma**: Rate of change of delta with respect to underlying price
- **Theta**: Rate of change of option price with respect to time
- **Vega**: Rate of change of option price with respect to volatility
- **Rho**: Rate of change of option price with respect to interest rate

## Database Schema

### TestingOptionData Model
- Uploaded options data for model training and evaluation
- Trade datetime, expiration date, underlying symbol
- Strike price, option type, trade price
- Market data integration for underlying prices and volatility

### MarketData Model
- Symbol, Price, Volume
- Timestamp

## Development

### Running Tests

**Backend:**
```bash
cd backend
python manage.py test
```

**Frontend:**
```bash
cd frontend
npm test
```

### Code Quality

**Backend:**
```bash
cd backend
pip install flake8 black
flake8 .
black .
```

**Frontend:**
```bash
cd frontend
npm run lint
```

## Deployment

### Production Setup

1. **Configure environment variables**
2. **Set up production database**
3. **Configure static files**
4. **Set up reverse proxy (nginx)**
5. **Use production WSGI server (gunicorn)**

### Docker Deployment

Docker configuration files can be added for containerized deployment.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For support and questions, please open an issue in the repository.

## Acknowledgments

- Black-Scholes model implementation
- Django REST Framework documentation
- React TypeScript best practices
- Modern CSS design patterns
- SHAP library for model interpretability
- TensorFlow/Keras for deep learning implementation 

## Start by entering into the backend directory: 
- cd /Users/danielzhang/options-pricing-program/options-pricing-app/backend
- source venv/bin/activate

## Entering the frontend: 
- cd /Users/danielzhang/options-pricing-program/options-pricing-app/frontend
- npm install
- npm start