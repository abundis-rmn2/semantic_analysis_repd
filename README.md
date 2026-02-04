# Semantic Analysis REPD

This project is a standalone tool for analyzing violence in REPD (Registro Estatal de Personas Desaparecidas) data using LLM enrichment and semantic analysis.

## Project Structure
- `data/`: Contains the input CSV files (e.g., `sisovid.csv`).
- `config/`: Contains configuration files and API keys (`.env`).
- `results/`: Directory where analysis results and graphs are saved.
- `process_violence.py`: Main processing script.

## Features
- **Deterministic Extraction**: Regex and keyword matching for fast signals.
- **LLM Enrichment**: DeepSeek API integration for structured extraction of weapons, perpetrators, and event explanations.
- **Incremental Saving**: Results are saved every 10 items to prevent data loss.
- **Graph Analysis**: Generates a network of related cases based on shared metadata and semantic similarity.

## Requirements
To install the necessary dependencies, run:
```bash
pip install -r requirements.txt
python3 -m spacy download es_core_news_sm
```

## Usage
1. Place your data in `data/sisovid.csv`.
2. Configure your `DEEPSEEK_API_KEY` in `config/.env`.
3. Run the analysis:
```bash
python3 process_violence.py
```

## Running on Raspberry Pi
To set up this project on a Raspberry Pi (Pi 4 or newer recommended):

1. **Install Dependencies**: Run the provided setup script:
   ```bash
   ./setup_pi.sh
   ```
2. **Configure Environment**:
   ```bash
   cp config/.env.example config/.env
   # Edit config/.env and add your DEEPSEEK_API_KEY
   ```
3. **Run the Analysis**:
   ```bash
   source venv/bin/activate
   # Example: process 20 cases to verify
   python process_probable.py --sample 20
   ```
4. **Run the Dashboard**:
   ```bash
   streamlit run dashboard_probable.py
   ```

Note: This setup uses **CPU-only** libraries and pins `numpy<2` to ensure stability on ARM architecture. Even so, the initial installation and first run may take some time.
