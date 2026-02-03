import os
import json
import logging
import pandas as pd
import sqlite3
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class StorageManager:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Paths
        self.parquet_path = os.path.join(output_dir, "probable_analysis.parquet")
        self.csv_path = os.path.join(output_dir, "probable_analysis.csv")
        self.prompts_log_path = os.path.join(output_dir, "prompts_log.jsonl")
        self.db_path = os.path.join(output_dir, "provenance.db")
        
        self._init_db()

    def _init_db(self):
        """Initialize SQLite for provenance tracking."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS provenance (
                    id_original TEXT PRIMARY KEY,
                    input_hash TEXT,
                    prompt_id TEXT,
                    llm_response TEXT,
                    pipeline_version TEXT,
                    timestamp DATETIME
                )
            """)

    def save_results(self, results: List[Dict[str, Any]], append: bool = True):
        """Save results to CSV and Parquet."""
        df = pd.DataFrame(results)
        
        # Save Parquet (Rich types)
        if not append or not os.path.exists(self.parquet_path):
            df.to_parquet(self.parquet_path, index=False)
        else:
            existing_df = pd.read_parquet(self.parquet_path)
            pd.concat([existing_df, df]).to_parquet(self.parquet_path, index=False)
            
        # Save CSV (Human readable, might truncate complex fields)
        csv_df = df.copy()
        # Flatten complex structures for CSV better readability
        if 'scenarios' in csv_df.columns:
            csv_df['scenarios'] = csv_df['scenarios'].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x)
        if 'keywords' in csv_df.columns:
            csv_df['keywords'] = csv_df['keywords'].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
        
        if not append or not os.path.exists(self.csv_path):
            csv_df.to_csv(self.csv_path, index=False)
        else:
            csv_df.to_csv(self.csv_path, mode='a', header=False, index=False)

    def log_prompt(self, prompt_data: Dict[str, Any]):
        """Append LLM prompt/response to JSONL log."""
        prompt_data['timestamp'] = datetime.now().isoformat()
        with open(self.prompts_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(prompt_data, ensure_ascii=False) + '\n')

    def save_provenance(self, record_id: str, input_hash: str, prompt_id: str, response: str, version: str):
        """Store provenance in SQLite."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO provenance (id_original, input_hash, prompt_id, llm_response, pipeline_version, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (record_id, input_hash, prompt_id, response, version, datetime.now()))

    def save_faiss_index(self, embeddings, ids, index_name="embeddings.faiss"):
        """Save FAISS index if library is available."""
        try:
            import faiss
            import numpy as np
            
            d = embeddings.shape[1]
            index = faiss.IndexFlatL2(d)
            index.add(np.array(embeddings).astype('float32'))
            
            index_path = os.path.join(self.output_dir, index_name)
            faiss.write_index(index, index_path)
            
            # Save mapping of index position to id_original
            mapping_path = index_path + ".map.json"
            with open(mapping_path, 'w') as f:
                json.dump({i: id_ for i, id_ in enumerate(ids)}, f)
                
            logger.info(f"FAISS index saved to {index_path}")
        except ImportError:
            logger.warning("FAISS not installed. Skipping index save.")
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
