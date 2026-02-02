import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def evaluate_cluster_fusion_candidates(df: pd.DataFrame, similarity_threshold: float = 0.85) -> List[Dict[str, Any]]:
    """
    Identify clusters that could be merged based on centroid similarity.
    """
    if 'cluster_label' not in df.columns or 'embedding' not in df.columns:
        return []

    # Calculate centroids
    centroids = {}
    valid_clusters = df[df['cluster_label'] != -1]
    
    for label, group in valid_clusters.groupby('cluster_label'):
        embs = np.array(group['embedding'].tolist())
        centroids[label] = np.mean(embs, axis=0)

    labels = list(centroids.keys())
    merges = []
    
    from sklearn.metrics.pairwise import cosine_similarity
    
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            li, lj = labels[i], labels[j]
            sim = cosine_similarity([centroids[li]], [centroids[lj]])[0][0]
            
            if sim >= similarity_threshold:
                merges.append({
                    "clusters": (li, lj),
                    "similarity": float(sim),
                    "action": "suggest_merge"
                })
                
    return merges

def eliminate_noisy_clusters(df: pd.DataFrame, min_density: float = 0.3) -> List[int]:
    """
    Identify clusters that are too sparse/noisy to be useful.
    (Simplified: small clusters or those with high internal variance)
    """
    to_eliminate = []
    if 'cluster_label' not in df.columns or 'embedding' not in df.columns:
        return to_eliminate

    valid_clusters = df[df['cluster_label'] != -1]
    
    for label, group in valid_clusters.groupby('cluster_label'):
        if len(group) < 2: continue # Handled by hdbscan anyway
        
        # Calculate average distance to centroid
        embs = np.array(group['embedding'].tolist())
        centroid = np.mean(embs, axis=0)
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(embs, [centroid])
        
        avg_sim = np.mean(similarities)
        if avg_sim < min_density:
            to_eliminate.append(int(label))
            
    return to_eliminate
