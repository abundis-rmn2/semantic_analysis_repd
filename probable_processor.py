import os
import json
import logging
import hashlib
import pandas as pd
from typing import List, Optional, Dict, Any
from datetime import datetime

# Reusable utils
from reusable.text_utils import normalize_text, extract_regex_slots
from schemas import EventAnalysis, ScenarioHypothesis, Evidence
from prompts import EXPLORATORY_SCENARIO_PROMPT, AMBIGUITY_DETECTOR_PROMPT
from storage import StorageManager

# External deps
from openai import OpenAI
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class ProbableProcessor:
    def __init__(self, output_dir: str = "results", model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.output_dir = output_dir
        self.storage = StorageManager(output_dir)
        self.pipeline_version = "1.0.0-probable"
        
        # Env / API
        env_path = os.path.join(os.path.dirname(__file__), "config", ".env")
        load_dotenv(env_path if os.path.exists(env_path) else None)
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com") if self.api_key else None
        
        # Models
        self.embedding_model = None
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(model_name)
            logger.info(f"Loaded embedding model: {model_name}")
        except Exception as e:
            logger.warning(f"Could not load embedding model: {e}")

    def signal_engineering(self, row: pd.Series) -> Dict[str, Any]:
        """Combine regex signals with keyword families."""
        text = str(row.get('descripcion_desaparicion', ''))
        norm_text = normalize_text(text)
        
        slots = extract_regex_slots(text)
        
        # New signal families (Signal engineering)
        signals = {
            "laboral": ["trabajo", "empleo", "puesto", "turno", "casa de empeño", "oferta", "vacante"],
            "migracion": ["cruzo", "frontera", "emigro", "migracion", "norte", "bolsa"],
            "institucional": ["hospital", "detenido", "agencia", "comisaria", "separos", "pgr", "fiscalia"],
        }
        
        keyword_hits = []
        for family, keywords in signals.items():
            hits = [kw for kw in keywords if kw in norm_text]
            if hits:
                keyword_hits.append({"family": family, "hits": hits})
                
        return {
            "temporal_slots": slots.get('date', []),
            "location_slots": slots.get('location_hint', []),
            "transport_slots": slots.get('transport', []),
            "keyword_hits": keyword_hits
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def call_llm_scenarios(self, text: str) -> Optional[List[ScenarioHypothesis]]:
        if not self.client: return None
        
        prompt = EXPLORATORY_SCENARIO_PROMPT.format(text=text)
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "Eres un analista de inteligencia criminal experto en desapariciones."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2
            )
            data = json.loads(response.choices[0].message.content)
            
            # Validation
            hypotheses = [ScenarioHypothesis(**s) for s in data.get('scenarios', [])]
            
            # Log for provenance
            self.storage.log_prompt({
                "type": "scenarios",
                "text": text,
                "response": data,
                "model": "deepseek-chat"
            })
            
            return hypotheses
        except Exception as e:
            logger.error(f"LLM Error (scenarios): {e}")
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def call_llm_ambiguity(self, text: str) -> Dict[str, Any]:
        if not self.client: return {"ambiguity_score": 0.5, "notes": "No LLM"}
        
        prompt = AMBIGUITY_DETECTOR_PROMPT.format(text=text)
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"LLM Error (ambiguity): {e}")
            return {"ambiguity_score": 1.0, "error": str(e)}

    def run_pipeline(self, df: pd.DataFrame, skip_llm: bool = False):
        all_results = []
        
        for idx, row in df.iterrows():
            id_orig = str(row.get('id_cedula_busqueda', idx))
            raw_text = str(row.get('descripcion_desaparicion', ''))
            norm_text = normalize_text(raw_text)
            input_hash = hashlib.sha256(raw_text.encode()).hexdigest()
            
            logger.info(f"Processing {id_orig}...")
            
            # A. Observables (Deterministic)
            observables = self.signal_engineering(row)
            
            # B. LLM Analysis
            scenarios = []
            ambiguity_data = {"ambiguity_score": 0.0}
            
            if not skip_llm and self.api_key:
                scenarios = self.call_llm_scenarios(raw_text) or []
                ambiguity_data = self.call_llm_ambiguity(raw_text)
            
            # C. Merge & Corroborate
            # (Heuristic: raise confidence if LLM + deterministic match)
            for scenario in scenarios:
                if scenario.scenario_label == 'laboral_forzada' and any(h['family'] == 'laboral' for h in observables['keyword_hits']):
                    scenario.scenario_confidence = 'alta'
            
            # D. Final Schema Mapping
            analysis = EventAnalysis(
                id_original=id_orig,
                raw_text=raw_text,
                text_norm=norm_text,
                date_dt=row.get('date_dt') if pd.notna(row.get('date_dt')) else None,
                municipio=str(row.get('municipio', 'DESCONOCIDO')),
                observables=observables,
                scenarios=scenarios,
                ambiguity_score=ambiguity_data.get('ambiguity_score', 0.0),
                llm_meta={"model": "deepseek-chat", "version": self.pipeline_version},
                provenance_id=f"{id_orig}_{input_hash[:8]}",
                evidence_snippets=ambiguity_data.get('missing_info', [])
            )
            
            dict_res = analysis.dict()
            if self.embedding_model:
                dict_res['embedding'] = self.embedding_model.encode(raw_text).tolist()
                
            all_results.append(dict_res)
            
            # Provenance log
            self.storage.save_provenance(id_orig, input_hash, "scenarios_v1", json.dumps([s.dict() for s in scenarios]), self.pipeline_version)

        # Batch save
        self.storage.save_results(all_results)
        
        # FAISS save
        if self.embedding_model:
            import numpy as np
            embeddings = np.array([r['embedding'] for r in all_results])
            ids = [r['id_original'] for r in all_results]
            self.storage.save_faiss_index(embeddings, ids)
            
        return all_results
