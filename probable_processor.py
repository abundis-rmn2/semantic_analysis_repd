import os
import json
import logging
import hashlib
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

# Reusable utils
from reusable.text_utils import normalize_text, extract_regex_slots
from schemas import EventAnalysis, ScenarioHypothesis, Evidence
from prompts import (
    EXPLORATORY_SCENARIO_PROMPT, 
    AMBIGUITY_DETECTOR_PROMPT,
    SCENARIO_NORMALIZER_PROMPT,
    KEYWORD_POOL_PROMPT
)
from storage import StorageManager
from clustering_utils import perform_clustering, calculate_cluster_entropy
from graph_utils import build_affinity_graph, build_bipartite_network, save_graph

# External deps
from openai import OpenAI
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class ProbableProcessor:
    def __init__(self, output_dir: str = "results", model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.output_dir = output_dir
        self.storage = StorageManager(output_dir)
        self.pipeline_version = "1.1.0-probable-pool"
        
        # Env / API
        env_path = os.path.join(os.path.dirname(__file__), "config", ".env")
        load_dotenv(env_path if os.path.exists(env_path) else None)
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com") if self.api_key else None
        
        # Taxonomy Initialization (Unified Pools)
        self.golden_path = os.path.join(os.path.dirname(__file__), "config", "golden_taxonomy.json")
        self.taxonomy_path = os.path.join(output_dir, "narrative_taxonomy.json")
        
        # 1. Load Golden (Reference)
        golden = self._load_pool(self.golden_path, {"keywords": {}, "scenarios": []})
        
        # 2. Load Dynamic (Results)
        dynamic = self._load_pool(self.taxonomy_path, {"keywords": {}, "scenarios": []})
        
        # Merge Scenarios (Golden takes priority)
        self.scenario_pool = golden.get("scenarios", [])
        existing_labels = {s['label'] for s in self.scenario_pool}
        for s in dynamic.get("scenarios", []):
            if s['label'] not in existing_labels:
                self.scenario_pool.append(s)
                
        # Merge Keywords
        self.keyword_pool = golden.get("keywords", {})
        for family, kws in dynamic.get("keywords", {}).items():
            if family in self.keyword_pool:
                self.keyword_pool[family] = list(set(self.keyword_pool[family] + kws))
            else:
                self.keyword_pool[family] = kws

        # Models
        self.embedding_model = None
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(model_name)
            logger.info(f"Loaded embedding model: {model_name}")
        except Exception as e:
            logger.warning(f"Could not load embedding model: {e}")

    def _load_pool(self, path: str, default: Any) -> Any:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading pool from {path}: {e}")
        return default

    def _save_pools(self):
        taxonomy = {
            "keywords": self.keyword_pool,
            "scenarios": self.scenario_pool
        }
        with open(self.taxonomy_path, 'w', encoding='utf-8') as f:
            json.dump(taxonomy, f, indent=4, ensure_ascii=False)
        logger.info(f"Taxonomy updated at {self.taxonomy_path}")

    def signal_engineering(self, row: pd.Series) -> Dict[str, Any]:
        """Combine regex signals with dynamic keyword families."""
        text = str(row.get('descripcion_desaparicion', ''))
        norm_text = normalize_text(text)
        slots = extract_regex_slots(text)
        
        keyword_hits = []
        for family, keywords in self.keyword_pool.items():
            hits = [kw for kw in keywords if kw.lower() in norm_text.lower()]
            if hits:
                keyword_hits.append({"family": family, "hits": list(set(hits))})
                
        return {
            "temporal_slots": slots.get('date', []),
            "location_slots": slots.get('location_hint', []),
            "transport_slots": slots.get('transport', []),
            "keyword_hits": keyword_hits
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def call_llm_scenarios(self, text: str) -> Tuple[Optional[List[ScenarioHypothesis]], List[str]]:
        if not self.client: return None, []
        
        pool_str = "\n".join([f"- {s['label']}: {s['description']}" for s in self.scenario_pool])
        prompt = EXPLORATORY_SCENARIO_PROMPT.format(text=text, scenario_pool=pool_str)
        
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
            
            hypotheses = [ScenarioHypothesis(**s) for s in data.get('scenarios', [])]
            discovered_keywords = data.get('discovered_keywords', [])
            
            return hypotheses, discovered_keywords
        except Exception as e:
            logger.error(f"LLM Error (scenarios): {e}")
            return None, []

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=2))
    def normalize_pools(self, emergent_scenarios: List[Dict], discovered_keywords: List[str]):
        """Update keyword and scenario pools using LLM normalization."""
        if not self.client: return

        # 1. Normalize Scenarios
        if emergent_scenarios:
            logger.info(f"Normalizing {len(emergent_scenarios)} emergent scenarios...")
            norm_prompt = SCENARIO_NORMALIZER_PROMPT.format(
                standard_pool=json.dumps(self.scenario_pool, indent=2),
                emergent_scenarios=json.dumps(emergent_scenarios, indent=2)
            )
            try:
                resp = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": norm_prompt}],
                    response_format={"type": "json_object"}
                )
                norm_data = json.loads(resp.choices[0].message.content)
                self.scenario_pool = norm_data.get('updated_pool', self.scenario_pool)
            except Exception as e:
                logger.error(f"Error normalizing scenarios: {e}")

        # 2. Update Keyword Pool
        if discovered_keywords:
            logger.info(f"Updating keyword pool with {len(discovered_keywords)} new keywords...")
            kw_prompt = KEYWORD_POOL_PROMPT.format(
                keyword_pool=json.dumps(self.keyword_pool, indent=2),
                new_keywords=json.dumps(list(set(discovered_keywords)), indent=2)
            )
            try:
                resp = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": kw_prompt}],
                    response_format={"type": "json_object"}
                )
                kw_data = json.loads(resp.choices[0].message.content)
                self.keyword_pool = kw_data.get('updated_keyword_pool', self.keyword_pool)
            except Exception as e:
                logger.error(f"Error updating keywords: {e}")

        self._save_pools()

    def _update_graphs(self, all_results: List[Dict]):
        """Build and save graphs based on current accumulated results."""
        if not all_results: return
        
        logger.info(f"Updating incremental graphs with {len(all_results)} cases...")
        res_df = pd.DataFrame(all_results)
        
        # 1. Narrative Affinity Graph
        G_affinity = build_affinity_graph(res_df)
        save_graph(G_affinity, os.path.join(self.output_dir, "probable_graph.gml"))
        
        # 2. Relationship Network
        G_rel = build_bipartite_network(res_df, keyword_pool=self.keyword_pool)
        save_graph(G_rel, os.path.join(self.output_dir, "relationship_network.gml"))

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

    def run_pipeline(self, df: pd.DataFrame, skip_llm: bool = False, batch_size: int = 10):
        all_results = []
        current_batch = []
        batch_emergent_scenarios = []
        batch_discovered_kws = []
        
        for i, (idx, row) in enumerate(df.iterrows()):
            id_orig = str(row.get('id_cedula_busqueda', idx))
            raw_text = str(row.get('descripcion_desaparicion', ''))
            norm_text = normalize_text(raw_text)
            input_hash = hashlib.sha256(raw_text.encode()).hexdigest()
            
            logger.info(f"[{i+1}/{len(df)}] Processing {id_orig}...")
            
            # A. Observables (Deterministic)
            observables = self.signal_engineering(row)
            
            # B. LLM Analysis
            scenarios = []
            ambiguity_data = {"ambiguity_score": 0.0}
            
            if not skip_llm and self.client:
                scenarios, discovered_kws = self.call_llm_scenarios(raw_text)
                scenarios = scenarios or []
                ambiguity_data = self.call_llm_ambiguity(raw_text)
                
                # Collect for pooling
                batch_discovered_kws.extend(discovered_kws)
                for s in scenarios:
                    if not any(sp['label'] == s.scenario_label for sp in self.scenario_pool):
                        batch_emergent_scenarios.append({
                            "label": s.scenario_label,
                            "notes": s.notes or ""
                        })
            
            # C. Merge & Corroborate
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
                keywords=list(set(discovered_kws)) if not skip_llm else [],
                ambiguity_score=ambiguity_data.get('ambiguity_score', 0.0),
                llm_meta={
                    "model": "deepseek-chat", 
                    "version": self.pipeline_version,
                    "discovered_keywords": discovered_kws if not skip_llm else []
                },
                provenance_id=f"{id_orig}_{input_hash[:8]}",
                evidence_snippets=ambiguity_data.get('missing_info', [])
            )
            
            dict_res = analysis.dict()
            if self.embedding_model:
                dict_res['embedding'] = self.embedding_model.encode(raw_text).tolist()
                
            all_results.append(dict_res)
            current_batch.append(dict_res)
            
            # Provenance log (per record)
            if not skip_llm:
                self.storage.save_provenance(
                    id_orig, 
                    input_hash, 
                    "scenarios_v1.1", 
                    json.dumps({"scenarios": [s.dict() for s in scenarios], "keywords": discovered_kws}), 
                    self.pipeline_version
                )

            # E. Incremental Save & Pool Update (Batch)
            if (i + 1) % batch_size == 0 or (i + 1) == len(df):
                logger.info(f"Saving incremental batch of {len(current_batch)} items...")
                self.storage.save_results(current_batch, append=True)
                
                # Normalize and update pools
                if not skip_llm:
                    self.normalize_pools(batch_emergent_scenarios, batch_discovered_kws)
                    batch_emergent_scenarios = []
                    batch_discovered_kws = []
                
                # Incremental Graph & Cluster Update
                if self.embedding_model and len(all_results) > 1:
                    import numpy as np
                    embeddings = np.array([r['embedding'] for r in all_results])
                    cluster_labels = perform_clustering(embeddings)
                    for idx_all, label in enumerate(cluster_labels):
                        all_results[idx_all]['cluster_label'] = int(label)
                    
                    # Update the CSV with the labels
                    self.storage.save_results(all_results, append=False)
                
                self._update_graphs(all_results)
                current_batch = []

        # Final FAISS save
        if self.embedding_model and all_results:
            import numpy as np
            embeddings = np.array([r['embedding'] for r in all_results])
            ids = [r['id_original'] for r in all_results]
            self.storage.save_faiss_index(embeddings, ids)
            
        return all_results
