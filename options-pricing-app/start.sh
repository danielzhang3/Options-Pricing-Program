#!/bin/bash

echo "🚀 Starting Options Pricing Program..."
echo "======================================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "✅ Docker and Docker Compose are installed"

# Create necessary directories if they don't exist
mkdir -p backend/staticfiles
mkdir -p frontend/build

echo "📦 Building and starting services..."

# Start the services
docker-compose up --build -d

echo "⏳ Waiting for services to start..."
sleep 10

# Check if services are running
if docker-compose ps | grep -q "Up"; then
    echo "✅ Services are running!"
    echo ""
    echo "🌐 Access your application at:"
    echo "   Frontend: http://localhost:3000"
    echo "   Backend API: http://localhost:8000/api/"
    echo "   Django Admin: http://localhost:8000/admin/"
    echo "   PgAdmin4: http://localhost:5050"
    echo ""
    echo "📧 PgAdmin4 credentials:"
    echo "   Email: admin@optionspricing.com"
    echo "   Password: admin"
    echo ""
    echo "🔧 To stop the services, run: docker-compose down"
    echo "📊 To view logs, run: docker-compose logs -f"
else
    echo "❌ Failed to start services. Check the logs with: docker-compose logs"
    exit 1
fi 