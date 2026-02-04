# Semantic Analysis REPD

This project is a standalone tool for analyzing violence in REPD (Registro Estatal de Personas Desaparecidas) data using LLM enrichment and semantic analysis.

## Project Structure
- `data/`: Contains the input CSV files (e.g., `sisovid.csv`).
- `config/`: Contains configuration files and API keys (`.env`).
- `results/`: Directory where analysis results and graphs are saved.
- `process_violence.py`: Main processing script.

## Installation & Setup

This project is optimized for CPU-only environments, including standard PCs and Raspberry Pi (ARM).

### 1. Requirements
- Python 3.10+
- DeepSeek API Key (configured in `config/.env`)
- **For Raspberry Pi**: Minimum 4GB RAM recommended (or 2GB Swap enabled).

### 2. Universal Setup
Run the automated setup script. It will detect your architecture (ARM or x86) and apply the necessary patches for architecture stability:
```bash
chmod +x setup.sh
./setup.sh
```

### 3. Memory Optimization (CRITICAL for Raspberry Pi)
If you are running on a Raspberry Pi and encounter "Killed" or "Illegal Instruction" errors, you likely need more Swap space:
```bash
# Set swap to 2GB
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile  # Change CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

### 4. Running the Project
```bash
# 1. Activate environment
source venv/bin/activate

# 2. Run analysis (example with sample)
python process_probable.py --sample 20

# 3. Launch Dashboard
streamlit run dashboard_probable.py
```

## Features
- **Deterministic Extraction**: Regex and keyword matching for fast signals.
- **LLM Enrichment**: DeepSeek API integration for structured extraction of weapons, perpetrators, and event explanations.
- **Incremental Saving**: Results are saved every 10 items to prevent data loss.
- **Graph Analysis**: Generates a network of related cases based on shared metadata and semantic similarity.
- **CPU Optimized**: Uses `numpy<2` and `torch-cpu` to ensure compatibility across architectures.
