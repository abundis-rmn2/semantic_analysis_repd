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

## Sistema de Pooling Dinámico (v1.2)

El pipeline implementa un sistema de retroalimentación iterativa consolidado en un solo archivo:

- **Narrative Taxonomy (`results/narrative_taxonomy.json`)**: 
    - **Keywords**: Almacena palabras clave extraídas normalizadas por familias semánticas.
    - **Scenarios**: Mantiene la taxonomía evolutiva de escenarios de desaparición.
    - El LLM actualiza este archivo cada batch, permitiendo que el sistema "aprenda" y use estas categorías en la siguiente pasada.

## Salidas Adicionales

- `results/narrative_taxonomy.json`: Diccionario unificado de inteligencia narrativa.
- `results/relationship_network.gml`: Red interactiva de casos, escenarios y palabras clave.
