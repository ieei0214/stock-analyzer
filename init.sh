#!/bin/bash

# Stock Analyzer - Development Environment Setup Script
# This script sets up and runs the development environment

set -e

echo "=========================================="
echo "  Stock Analyzer - Environment Setup"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
    MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

    if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 10 ]; then
        echo -e "${GREEN}✓ Python $PYTHON_VERSION detected${NC}"
    else
        echo -e "${RED}✗ Python 3.10+ required, found $PYTHON_VERSION${NC}"
        exit 1
    fi
else
    echo -e "${RED}✗ Python 3 not found. Please install Python 3.10+${NC}"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Install dependencies
echo ""
echo "Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install --upgrade pip > /dev/null
    pip install -r requirements.txt
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${YELLOW}! requirements.txt not found, creating with default dependencies...${NC}"
    cat > requirements.txt << 'EOF'
# Web Framework
fastapi>=0.104.0
uvicorn[standard]>=0.24.0

# Database
aiosqlite>=0.19.0

# Stock Data
yfinance>=0.2.33

# LLM Integration
openai>=1.3.0
google-generativeai>=0.3.0

# Data Processing
pandas>=2.1.0
numpy>=1.26.0

# Charts
matplotlib>=3.8.0
plotly>=5.18.0

# Environment
python-dotenv>=1.0.0

# Utilities
httpx>=0.25.0
pydantic>=2.5.0
pydantic-settings>=2.1.0

# Development
pytest>=7.4.0
pytest-asyncio>=0.21.0
EOF
    pip install --upgrade pip > /dev/null
    pip install -r requirements.txt
    echo -e "${GREEN}✓ Dependencies installed${NC}"
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo ""
    echo "Creating .env file from template..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
    else
        cat > .env << 'EOF'
# LLM Configuration
LLM_PROVIDER=openai
OPENAI_API_KEY=your-openai-api-key-here
GEMINI_API_KEY=your-gemini-api-key-here
MODEL_NAME=gpt-4

# Application Settings
DEBUG=true
HOST=127.0.0.1
PORT=8000

# Cache Settings
CACHE_DURATION_HOURS=1

# Analysis Settings
DEFAULT_ANALYSIS_STYLE=Conservative
EOF
    fi
    echo -e "${YELLOW}! Created .env file - Please update with your API keys${NC}"
else
    echo -e "${GREEN}✓ .env file exists${NC}"
fi

# Create necessary directories
echo ""
echo "Creating directory structure..."
mkdir -p src/agents
mkdir -p src/api
mkdir -p src/database
mkdir -p src/charts
mkdir -p src/reports
mkdir -p data
mkdir -p reports
mkdir -p static/css
mkdir -p static/js
mkdir -p templates
mkdir -p tests

echo -e "${GREEN}✓ Directory structure created${NC}"

# Initialize database if needed
echo ""
echo "Checking database..."
if [ ! -f "data/stock_analyzer.db" ]; then
    echo "Database will be initialized on first run"
else
    echo -e "${GREEN}✓ Database exists${NC}"
fi

# Print summary
echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "To start the development server:"
echo ""
echo "  source venv/bin/activate"
echo "  uvicorn src.main:app --reload --host 127.0.0.1 --port 8000"
echo ""
echo "Or simply run:"
echo "  ./run.sh"
echo ""
echo "The application will be available at:"
echo -e "  ${GREEN}http://localhost:8000${NC}"
echo ""
echo "API Documentation:"
echo -e "  ${GREEN}http://localhost:8000/docs${NC}"
echo ""

# Check if we should start the server
if [ "$1" == "--run" ]; then
    echo "Starting development server..."
    echo ""
    uvicorn src.main:app --reload --host 127.0.0.1 --port 8000
fi
