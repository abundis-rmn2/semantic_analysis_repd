import pandas as pd
import numpy as np
import os
import re
import json
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

# Standard Baseline imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Advanced Parsing & Matching
try:
    import regex as re_adv
except ImportError:
    re_adv = re
from rapidfuzz import process, fuzz
from flashtext import KeywordProcessor
from dateutil import parser as date_parser

# Semantic & NLP
try:
    import spacy
    from sentence_transformers import SentenceTransformer
except ImportError:
    spacy = None
    SentenceTransformer = None

# Validation & LLM
from pydantic import BaseModel, Field, validator
from openai import OpenAI
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

# Graph Analysis
import networkx as nx

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Pydantic Schemas for LLM Control ---

class ViolenceEvent(BaseModel):
    is_violent: bool = Field(..., description="¿Hubo violencia directa en el evento?")
    violence_type: Optional[str] = Field(None, description="Tipo de violencia (Física, Psicológica, Armada, etc.)")
    weapons: List[str] = Field(default_factory=list, description="Armas mencionadas (ej. arma corta, machete, golpeadores)")
    perpetrators: Optional[str] = Field(None, description="Descripción de los perpetradores (ej. grupo armado, sujetos desconocidos)")
    injuries: Optional[str] = Field(None, description="Lesiones mencionadas")
    context_confidence: str = Field(..., description="Confianza contextual: alta, media, baja")
    explanation: str = Field(..., description="Resumen de 3-4 líneas del evento")
    text_cites: List[str] = Field(..., description="Citas textuales que respaldan la extracción (máximo 3)")

    @validator('context_confidence')
    def validate_confidence(cls, v):
        if v.lower() not in ['alta', 'media', 'baja']:
            return 'baja'
        return v.lower()

# --- Core Pipeline Class ---

class EnhancedViolenceProcessor:
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        # Search for .env in standalone config dir
        env_path = os.path.join(os.path.dirname(__file__), "config", ".env")
        if os.path.exists(env_path):
            load_dotenv(env_path)
            logger.info(f"Loaded .env from {env_path}")
        else:
            load_dotenv() # Fallback to standard
            logger.warning(f".env not found in {env_path}, searching in default locations.")
        
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com") if self.api_key else None
        
        if not self.client:
            logger.warning("DeepSeek API Key not found. LLM features will be disabled.")
        
        # Phase 1: Baseline Keywords (Immutable)
        self.baseline_keywords = [
            "arma", "camioneta", "llev", "golp", "secuestr", "fuerz", 
            "rehabil", "amenaz", "dinero", "organ", "discu", "drog", 
            "pistol", "empleo", "violen", "camionera", "central"
        ]
        
        # Phase 2: Keyword Processor (FlashText)
        self.keyword_processor = KeywordProcessor()
        for kw in self.baseline_keywords:
            self.keyword_processor.add_keyword(kw)
            
        # Semantic Model
        self.embedding_model = None
        if SentenceTransformer:
            try:
                self.embedding_model = SentenceTransformer(model_name)
                logger.info(f"Loaded embedding model: {model_name}")
            except Exception as e:
                logger.warning(f"Could not load embedding model: {e}")

        # NLP Model
        self.nlp = None
        if spacy:
            try:
                self.nlp = spacy.load("es_core_news_sm")
                logger.info("Loaded spaCy model: es_core_news_sm")
            except Exception as e:
                logger.warning(f"Could not load spaCy model: {e}")

    # --- 1. Preprocessing & Regex ---
    
    def extract_temporal_info(self, text: str) -> Dict[str, Any]:
        """Regex for dates, times, and patterns."""
        patterns = {
            "date": r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            "time": r'(\d{1,2}:\d{2})\s*(?:hrs|am|pm)?',
            "vehicle": r'(camioneta|vehículo|carro|auto)\s+([a-zA-Z\s]+)?'
        }
        extracted = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, str(text), re.IGNORECASE)
            extracted[key] = match.group(0) if match else None
        return extracted

    # --- 2. Hard Extraction (FlashText) ---
    
    def get_keyword_extracts(self, text: str) -> List[str]:
        return self.keyword_processor.extract_keywords(str(text))

    # --- 3. Fuzzy Signal Engineering ---
    
    def get_fuzzy_signals(self, text: str, target_list: List[str]) -> Dict[str, float]:
        signals = {}
        if not text: return signals
        for target in target_list:
            score = process.extractOne(target, [text], scorer=fuzz.partial_ratio)
            signals[f"fuzzy_{target}"] = score[1] if score else 0
        return signals

    # --- 5. LLM Integration ---

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def call_llm_for_slots(self, text: str) -> Optional[ViolenceEvent]:
        if not self.client:
            return None
            
        few_shot_context = """
Ejemplos de análisis:

Relato: "Se lo llevaron varios sujetos armados que llegaron en una camioneta blanca, lo golpearon y lo subieron a la fuerza."
Respuesta: {
    "is_violent": true,
    "violence_type": "Armada y Física",
    "weapons": ["armas de fuego", "camioneta", "golpes"],
    "perpetrators": "varios sujetos armados",
    "injuries": "golpes diversos",
    "context_confidence": "alta",
    "explanation": "Desaparición forzada ejecutada por grupo armado con uso de violencia física y vehículo.",
    "text_cites": ["sujetos armados", "lo golpearon", "subieron a la fuerza"]
}

Relato: "Salió de su casa a trabajar y ya no regresó. No tenía problemas con nadie."
Respuesta: {
    "is_violent": false,
    "violence_type": null,
    "weapons": [],
    "perpetrators": null,
    "injuries": null,
    "context_confidence": "baja",
    "explanation": "No se reportan indicios de violencia en el relato inicial.",
    "text_cites": ["ya no regresó"]
}
"""
        
        prompt = f"""Analiza el siguiente relato de desaparición y extrae información sobre violencia.
Responde estrictamente en formato JSON válido según el esquema solicitado.

{few_shot_context}

Nuevo Relato a analizar:
"{text}"

Esquema JSON esperado:
{{
    "is_violent": true/false,
    "violence_type": "string o null",
    "weapons": ["string"],
    "perpetrators": "string o null",
    "injuries": "string o null",
    "context_confidence": "alta/media/baja",
    "explanation": "Resumen de 3-4 líneas",
    "text_cites": ["Citas textuales del relato"]
}}
"""
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "Eres un asistente experto en análisis forense y criminológico que extrae información estructurada de testimonios en español."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            data = json.loads(response.choices[0].message.content)
            return ViolenceEvent(**data)
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            return None

    # --- 6. Scoring Engine ---
    
    def calculate_combined_score(self, row: Dict[str, Any]) -> float:
        score = 0
        # Deterministic Signals (Weight: 0.4)
        if row.get('keyword_count', 0) > 0:
            score += min(row['keyword_count'] * 0.1, 0.4)
            
        # LLM Signal (Weight: 0.4) - If LLM says violent and confidence is high
        if row.get('llm_is_violent'):
            conf_map = {'alta': 0.4, 'media': 0.2, 'baja': 0.1}
            score += conf_map.get(row.get('llm_confidence', 'baja'), 0)
            
        # Semantic/Fuzzy (Weight: 0.2)
        return min(score, 1.0)

    # --- Main Pipeline Execution ---

    def run_pipeline(self, input_csv: str, output_csv: str, sample_size: int = None, save_every: int = 10):
        logger.info(f"Starting pipeline for {input_csv}")
        if not os.path.exists(input_csv):
            logger.error(f"Input file not found: {input_csv}")
            return

        df = pd.read_csv(input_csv)
        
        if 'fecha_desaparicion' in df.columns:
            df['date_dt'] = pd.to_datetime(df['fecha_desaparicion'], errors='coerce')
        
        if sample_size:
            df = df.sample(min(sample_size, len(df))).copy()
            
        if os.path.exists(output_csv):
            os.remove(output_csv)
            
        all_results = []
        batch_results = []
        
        for i, (idx, row) in enumerate(df.iterrows()):
            text = row.get('descripcion_desaparicion', '')
            if pd.isna(text): text = ""
            
            temporal = self.extract_temporal_info(text)
            keywords = self.get_keyword_extracts(text)
            llm_result = self.call_llm_for_slots(text)
            
            res_row = {
                'id_original': row.get('id_cedula_busqueda', idx),
                'municipio': row.get('municipio', 'DESCONOCIDO'),
                'date_dt': row.get('date_dt'),
                'text': text,
                'found_keywords': ", ".join(keywords),
                'keyword_count': len(keywords),
                'detected_date': temporal.get('date'),
                'llm_is_violent': llm_result.is_violent if llm_result else False,
                'llm_violence_type': llm_result.violence_type if llm_result else None,
                'llm_weapons': ", ".join(llm_result.weapons) if llm_result else "",
                'llm_confidence': llm_result.context_confidence if llm_result else 'error',
                'llm_explanation': llm_result.explanation if llm_result else "Error en LLM",
                'llm_cites': "|".join(llm_result.text_cites) if llm_result else ""
            }
            
            if self.embedding_model:
                res_row['embedding'] = self.embedding_model.encode(text)
            
            res_row['final_score'] = self.calculate_combined_score(res_row)
            all_results.append(res_row)
            
            batch_row = {k: v for k, v in res_row.items() if k != 'embedding'}
            batch_results.append(batch_row)
            
            logger.info(f"Processed item {i+1}/{len(df)} (idx:{idx}) - Score: {res_row['final_score']:.2f}")
            
            if (i + 1) % save_every == 0:
                batch_df = pd.DataFrame(batch_results)
                batch_df.to_csv(output_csv, mode='a', header=not os.path.exists(output_csv), index=False)
                batch_results = []
                logger.info(f"--- Incremental save: {i + 1} items written to {output_csv} ---")

        if batch_results:
            batch_df = pd.DataFrame(batch_results)
            batch_df.to_csv(output_csv, mode='a', header=not os.path.exists(output_csv), index=False)
            
        logger.info(f"Pipeline complete. All results saved to {output_csv}")
        
        res_df = pd.DataFrame(all_results)
        self.build_relation_graph(res_df)

    def build_relation_graph(self, df: pd.DataFrame):
        """Build a graph of related cases based on weapons, violence type, and proximity."""
        G = nx.Graph()
        violent_df = df[df['llm_is_violent'] == True].reset_index()
        logger.info(f"Found {len(violent_df)} violent cases for graph nodes.")
        
        for idx, row in violent_df.iterrows():
            G.add_node(row['id_original'], 
                       municipio=row['municipio'], 
                       type=row['llm_violence_type'])

        from itertools import combinations
        logger.info("Generating candidate pairs for cross-matching...")
        for (i, row_a), (j, row_b) in combinations(violent_df.iterrows(), 2):
            m_match = row_a['municipio'] == row_b['municipio']
            d_prox = False
            if pd.notna(row_a['date_dt']) and pd.notna(row_b['date_dt']):
                d_prox = abs((row_a['date_dt'] - row_b['date_dt']).days) <= 7
            
            sem_sim = 0
            if 'embedding' in row_a and 'embedding' in row_b:
                from sklearn.metrics.pairwise import cosine_similarity
                sem_sim = cosine_similarity([row_a['embedding']], [row_b['embedding']])[0][0]

            weapons_a = set(w.strip().lower() for w in str(row_a['llm_weapons']).split(",") if w.strip())
            weapons_b = set(w.strip().lower() for w in str(row_b['llm_weapons']).split(",") if w.strip())
            shared_weapons = weapons_a.intersection(weapons_b)

            link_score = 0
            reasons = []
            if m_match: link_score += 0.3; reasons.append("mismo_municipio")
            if d_prox: link_score += 0.3; reasons.append("proximidad_temporal")
            if sem_sim > 0.8: link_score += 0.3; reasons.append(f"semantica_{sem_sim:.2f}")
            if shared_weapons: link_score += 0.4; reasons.append(f"armas_compartidas_{list(shared_weapons)}")

            if link_score >= 0.6:
                G.add_edge(row_a['id_original'], row_b['id_original'], 
                           weight=link_score, reasons=", ".join(reasons))
        
        logger.info(f"Generated graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
        if G.number_of_nodes() > 0:
            graph_path = os.path.join(os.path.dirname(__file__), "results", "violence_network.gml")
            nx.write_gml(G, graph_path)
            logger.info(f"Graph saved to {graph_path}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    processor = EnhancedViolenceProcessor()
    
    input_f = os.path.join(current_dir, "data", "sisovid.csv")
    output_f = os.path.join(current_dir, "results", "violence_analysis_results.csv")
    
    # Run with a smaller sample if needed, or None for full run
    processor.run_pipeline(input_f, output_f, sample_size=100)
