import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def perform_clustering(embeddings: np.ndarray, min_cluster_size: int = 3, min_samples: int = 1) -> np.ndarray:
    """
    Perform HDBSCAN clustering on embeddings.
    """
    if len(embeddings) < min_cluster_size:
        logger.warning("Not enough embeddings for clustering.")
        return np.array([-1] * len(embeddings))

    # Normalize embeddings
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings)

    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='euclidean')
    cluster_labels = clusterer.fit_predict(scaled_embeddings)
    
    unique, counts = np.unique(cluster_labels, return_counts=True)
    logger.info(f"Clustering finished: {dict(zip(unique, counts))}")
    
    return cluster_labels

def evaluate_temporal_cohorts(df: pd.DataFrame, time_window: str = '6M') -> List[Dict[str, Any]]:
    """
    Evaluate clusters by temporal cohorts (e.g., every 6 months).
    """
    if 'date_dt' not in df.columns or df['date_dt'].isnull().all():
        logger.warning("No temporal data available for cohort evaluation.")
        return []

    cohorts = []
    # Ensure date_dt is datetime
    df['date_dt'] = pd.to_datetime(df['date_dt'])
    
    # Group by time window
    for name, group in df.groupby(pd.Grouper(key='date_dt', freq=time_window)):
        if len(group) == 0: continue
        
        # Calculate dominant scenarios in this cohort
        all_scenarios = [s['scenario_label'] for sc_list in group['scenarios'] for s in sc_list if s['scenario_confidence'] == 'alta']
        if not all_scenarios: continue
        
        scenario_counts = pd.Series(all_scenarios).value_counts().to_dict()
        
        cohorts.append({
            "window_start": name,
            "count": len(group),
            "dominant_scenarios": scenario_counts
        })
        
    return cohorts

def calculate_cluster_entropy(cluster_labels: np.ndarray) -> float:
    """
    Calculate entropy of cluster assignments as a governance metric.
    Lower entropy means more concentrated/stable clusters.
    """
    from scipy.stats import entropy
    if len(cluster_labels) == 0: return 0.0
    
    _, counts = np.unique(cluster_labels, return_counts=True)
    return float(entropy(counts))
