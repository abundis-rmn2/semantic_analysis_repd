import networkx as nx
import pandas as pd
import logging
import os
from typing import List, Dict, Any
from itertools import combinations

logger = logging.getLogger(__name__)

def build_affinity_graph(df: pd.DataFrame, similarity_threshold: float = 0.7) -> nx.Graph:
    """
    Build a graph of narrative affinities between cases.
    """
    G = nx.Graph()
    
    # Add nodes
    for _, row in df.iterrows():
        node_id = row['id_original']
        G.add_node(node_id, 
                   municipio=row.get('municipio', 'UNKNOWN'),
                   ambiguity=row.get('ambiguity_score', 0),
                   cluster=row.get('cluster_label', -1))

    # Add edges based on similarities
    logger.info("Generating affinity edges...")
    for (idx_a, row_a), (idx_b, row_b) in combinations(df.iterrows(), 2):
        reasons = []
        weight = 0
        
        # 1. Scenario Co-occurrence (High Confidence)
        scenarios_a = {s['scenario_label'] for s in row_a.get('scenarios', []) if s.get('scenario_confidence') == 'alta'}
        scenarios_b = {s['scenario_label'] for s in row_b.get('scenarios', []) if s.get('scenario_confidence') == 'alta'}
        shared_scenarios = scenarios_a.intersection(scenarios_b)
        if shared_scenarios:
            weight += 0.4
            reasons.append(f"shared_scenarios:{list(shared_scenarios)}")
            
        # 2. Shared Signals (from observables)
        obs_a = row_a.get('observables', {})
        obs_b = row_b.get('observables', {})
        
        # Temporal proximity (within 7 days if date_dt exists)
        if pd.notna(row_a.get('date_dt')) and pd.notna(row_b.get('date_dt')):
            diff_days = abs((row_a['date_dt'] - row_b['date_dt']).days)
            if diff_days <= 7:
                weight += 0.2
                reasons.append(f"temporal_proximity:{diff_days}d")
                
        # Shared Location (Municipio)
        if row_a.get('municipio') == row_b.get('municipio') and row_a.get('municipio') != 'DESCONOCIDO':
            weight += 0.2
            reasons.append("same_municipio")

        # 3. Embedding Similarity (Weak tie)
        if 'embedding' in row_a and 'embedding' in row_b:
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            emb_a = np.array(row_a['embedding']).reshape(1, -1)
            emb_b = np.array(row_b['embedding']).reshape(1, -1)
            sim = cosine_similarity(emb_a, emb_b)[0][0]
            if sim >= similarity_threshold:
                weight += 0.3 * sim
                reasons.append(f"embedding_sim:{sim:.2f}")

        # Final edge filter
        if weight >= 0.5:
            G.add_edge(row_a['id_original'], row_b['id_original'], 
                       weight=round(float(weight), 2), 
                       reasons=", ".join(reasons))

    return G

def save_graph(G: nx.Graph, output_path: str):
    """Save graph to GML."""
    try:
        nx.write_gml(G, output_path)
        logger.info(f"Graph saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving graph: {e}")
