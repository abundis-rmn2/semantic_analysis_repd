# Probable: Exploratory Disappearance Analysis Pipeline

Este módulo implementa un pipeline exploratorio diseñado para generar hipótesis plausibles sobre las desapariciones, permitiendo escenarios múltiples y no deterministas (más allá de la violencia).

## Componentes

- `process_probable.py`: Punto de entrada CLI.
- `probable_processor.py`: Lógica central y orquestación.
- `schemas.py`: Modelos de datos Pydantic para validación y estructura.
- `prompts.py`: Plantillas de prompts con ejemplos few-shot.
- `storage.py`: Gestión de persistencia (Parquet, CSV, SQLite, FAISS).
- `graph_utils.py`: Construcción de grafos de afinidad narrativa.
- `reusable/`: Utilidades compartidas.

## Uso

Para ejecutar el pipeline en una muestra:
```bash
python process_probable.py --input data/sisovid.csv --sample 10 --output-dir results
```

Para ejecutar sin llamadas a LLM (solo señales deterministas):
```bash
python process_probable.py --skip-llm
```

## Salidas

- `results/probable_analysis.parquet`: Datos completos con estructuras ricas.
- `results/probable_analysis.csv`: Exportación para revisión humana.
- `results/probable_graph.gml`: Grafo de conexiones entre casos.
- `results/provenance.db`: Logs de auditoría y versiones.
- `results/prompts_log.jsonl`: Registro detallado de interacciones con el LLM.

## Configuración

Asegúrate de tener un archivo `config/.env` con tu `DEEPSEEK_API_KEY`.
