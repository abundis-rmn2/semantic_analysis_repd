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

def build_bipartite_network(df: pd.DataFrame, keyword_pool: Dict[str, List[str]] = None) -> nx.Graph:
    """
    Build a bipartite (multi-mode) network connecting:
    Case -> Scenario
    Case -> Keyword -> Family
    """
    G = nx.Graph()
    
    # Pre-map keywords to families for efficiency
    kw_to_family = {}
    if keyword_pool:
        for family, kws in keyword_pool.items():
            for kw in kws:
                kw_to_family[kw.lower()] = family

    for _, row in df.iterrows():
        case_id = row['id_original']
        G.add_node(case_id, type='case', municipio=row.get('municipio'))
        
        # 1. Case -> Scenario
        scs = row.get('scenarios', [])
        if isinstance(scs, str):
            import json
            try: scs = json.loads(scs)
            except: scs = []
            
        for s in scs:
            if s.get('scenario_confidence') in ['alta', 'media']:
                scen_node = f"SCEN:{s['scenario_label']}"
                G.add_node(scen_node, type='scenario', label=s['scenario_label'])
                G.add_edge(case_id, scen_node, weight=1.0 if s['scenario_confidence'] == 'alta' else 0.5)

        # 2. Case -> Keyword -> Family
        kws = row.get('keywords', [])
        if isinstance(kws, str):
            import json
            try: kws = json.loads(kws)
            except: kws = []

        for kw in kws:
            kw_node = f"KW:{kw.lower()}"
            G.add_node(kw_node, type='keyword', label=kw.lower())
            G.add_edge(case_id, kw_node, weight=1.0)
            
            # Map to family if exists
            family = kw_to_family.get(kw.lower())
            if family:
                fam_node = f"FAM:{family}"
                G.add_node(fam_node, type='family', label=family)
                G.add_edge(kw_node, fam_node, weight=1.0)

    return G

def save_graph(G: nx.Graph, output_path: str):
    """Save graph to GML with string conversion for non-serializable fields."""
    try:
        # GML doesn't like lists/dicts in attributes
        G_serializable = G.copy()
        for n, data in G_serializable.nodes(data=True):
            for k, v in data.items():
                if isinstance(v, (list, dict)):
                    G_serializable.nodes[n][k] = str(v)
        
        nx.write_gml(G_serializable, output_path)
        logger.info(f"Graph saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving graph: {e}")
