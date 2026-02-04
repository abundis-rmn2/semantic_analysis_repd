#!/bin/bash

# Update and install system dependencies
echo "Updating system packages..."
sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    libatlas-base-dev \
    libgomp1 \
    pkg-config

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo "Installing Python dependencies (this may take a while on a Pi)..."
pip install -r requirements.txt

# Download Spacy model
echo "Downloading Spacy model..."
python3 -m spacy download es_core_news_sm

echo "Setup complete!"
echo "To run the project:"
echo "1. Copy config/.env.example to config/.env and add your API key."
echo "2. Activate the virtual environment: source venv/bin/activate"
echo "3. Run the dashboard: streamlit run dashboard_probable.py"
