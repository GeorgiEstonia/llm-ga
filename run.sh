#!/bin/bash

# Set script to exit on error
set -e

echo "Starting LLM-GA Flask server..."

# Check if virtual environment exists and activate it
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
elif [ -d "../venv" ]; then
    echo "Activating virtual environment from parent directory..."
    source ../venv/bin/activate
fi

# Check for .env file and load it
if [ -f ".env" ]; then
    echo "Loading environment variables from .env file..."
    set -a
    source .env
    set +a
elif [ -f "../.env" ]; then
    echo "Loading environment variables from parent directory .env file..."
    set -a
    source ../.env
    set +a
fi

# Check for required environment variables
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY environment variable is not set. Add it to .env file or export it."
    exit 1
fi

if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "ERROR: ANTHROPIC_API_KEY environment variable is not set. Add it to .env file or export it."
    exit 1
fi

# Set Flask variables
export FLASK_APP=api_server.py
export FLASK_ENV=production

# Get port from environment or use default
PORT=${PORT:-8080}
echo "Starting server on port $PORT..."

# Run the Flask server
python -m flask run --host=0.0.0.0 --port=$PORT 