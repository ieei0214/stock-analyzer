#!/bin/bash

# Stock Analyzer - Run Development Server

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "Virtual environment not found. Run ./init.sh first."
    exit 1
fi

# Load environment variables
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Default values
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}

echo "Starting Stock Analyzer..."
echo "Server: http://${HOST}:${PORT}"
echo "API Docs: http://${HOST}:${PORT}/docs"
echo ""

# Run the server
uvicorn src.main:app --reload --host ${HOST} --port ${PORT}
