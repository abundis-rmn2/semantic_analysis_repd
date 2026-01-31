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
