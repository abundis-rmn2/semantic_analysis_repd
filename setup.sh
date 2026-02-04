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

# Clean pip cache
echo "Cleaning pip cache..."
pip cache purge

# Architecture Detection
ARCH=$(uname -m)
echo "Detected architecture: $ARCH"

if [[ "$ARCH" == "aarch64" || "$ARCH" == "armv7l" ]]; then
    echo "Configuring for ARM (Raspberry Pi)..."
    export OPENBLAS_CORETYPE=ARMV8
    export BLIS_ARCH="generic"
    echo 'export OPENBLAS_CORETYPE=ARMV8' >> venv/bin/activate
    echo 'export BLIS_ARCH="generic"' >> venv/bin/activate
else
    echo "Configuring for standard CPU (x86_64)..."
fi

# 1. Install NumPy first to pin version
echo "Installing NumPy..."
pip install "numpy<2"

# 2. Install Torch (CPU version) explicitly for compatibility
echo "Installing Torch (CPU-only)..."
pip install torch==2.2.0 --extra-index-url https://download.pytorch.org/whl/cpu

# 3. Install remaining requirements
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Download Spacy model
echo "Downloading Spacy model..."
python3 -m spacy download es_core_news_sm

echo "Setup complete!"
echo "------------------------------------------------"
echo "To run the project:"
echo "1. Configure your API key: nano config/.env"
echo "2. Activate virtualenv: source venv/bin/activate"
echo "3. Run analysis: python process_probable.py --sample 20"
echo "4. Run dashboard: streamlit run dashboard_probable.py"
