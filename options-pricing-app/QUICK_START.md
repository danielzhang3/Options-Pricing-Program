# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Option 1: Docker (Recommended)

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd options-pricing-app
   ```

2. **Start the application:**
   ```bash
   ./start.sh
   ```

3. **Access the application:**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000/api/
   - PgAdmin4: http://localhost:5050

### Option 2: Manual Setup

#### Backend Setup
```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

#### Frontend Setup
```bash
cd frontend
npm install
npm start
```

## 🎯 First Steps

1. **Open the application** at http://localhost:3000
2. **Navigate to the Calculator tab**
3. **Fill in sample data:**
   - Symbol: AAPL
   - Option Type: Call
   - Strike Price: 150
   - Expiration Date: 2024-12-20
   - Underlying Price: 150
   - Risk-Free Rate: 0.02
   - Volatility: 0.3
4. **Click "Calculate Option Price"**
5. **View the results** including price and Greeks

## 📊 Features Overview

- **Real-time option pricing** using Black-Scholes model
- **Complete Greeks calculations** (Delta, Gamma, Theta, Vega, Rho)
- **Option management** - save and view calculated options
- **Modern, responsive UI** built with React TypeScript
- **RESTful API** with Django REST Framework
- **PostgreSQL database** with PgAdmin4 management

## 🔧 Troubleshooting

### Common Issues

1. **Port already in use:**
   ```bash
   # Stop existing services
   docker-compose down
   # Or change ports in docker-compose.yml
   ```

2. **Database connection issues:**
   - Ensure PostgreSQL is running
   - Check database credentials in settings.py

3. **Frontend not connecting to backend:**
   - Verify backend is running on port 8000
   - Check CORS settings in Django

### Useful Commands

```bash
# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Restart services
docker-compose restart

# Rebuild services
docker-compose up --build
```

## 📚 Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore the API endpoints at http://localhost:8000/api/
- Access PgAdmin4 to view the database structure
- Customize the application for your needs

## 🆘 Support

If you encounter any issues:
1. Check the troubleshooting section above
2. Review the logs with `docker-compose logs`
3. Open an issue in the repository
4. Check the full documentation in README.md 