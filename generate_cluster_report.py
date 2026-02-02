import pandas as pd
import json
import os
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_report(parquet_path: str, output_path: str):
    if not os.path.exists(parquet_path):
        logger.error(f"File not found: {parquet_path}")
        return

    logger.info(f"Loading data from {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    if 'cluster_label' not in df.columns:
        logger.error("No cluster_label found in data.")
        return

    report_data = []
    
    logger.info("Analyzing clusters...")
    for label, group in df.groupby('cluster_label'):
        # 1. Basic Stats
        size = len(group)
        
        # 2. Dominant Scenarios
        scenarios = []
        for sc_list in group['scenarios']:
            for s in sc_list:
                if s.get('scenario_confidence') == 'alta':
                    scenarios.append(s.get('scenario_label'))
        
        top_scenarios = Counter(scenarios).most_common(3)
        scenarios_str = ", ".join([f"{name} ({count})" for name, count in top_scenarios])
        
        # 3. Dominant Municipios
        top_municipios = group['municipio'].value_counts().head(3).to_dict()
        municipios_str = ", ".join([f"{m} ({c})" for m, c in top_municipios.items()])
        
        # 4. Representative Keywords (from observables)
        keywords = []
        for obs in group['observables']:
            for hit in obs.get('keyword_hits', []):
                keywords.extend(hit.get('hits', []))
        
        top_keywords = Counter(keywords).most_common(5)
        keywords_str = ", ".join([f"{k} ({c})" for k, c in top_keywords])
        
        # 5. Sample IDs
        sample_ids = ", ".join(group['id_original'].head(5).astype(str).tolist())
        
        report_data.append({
            "cluster_id": label,
            "size": size,
            "representative_scenarios": scenarios_str,
            "top_municipios": municipios_str,
            "top_keywords": keywords_str,
            "sample_ids": sample_ids,
            "percentage": round((size / len(df)) * 100, 2)
        })

    report_df = pd.DataFrame(report_data)
    # Sort by size descending
    report_df = report_df.sort_values(by="size", ascending=False)
    
    report_df.to_csv(output_path, index=False)
    logger.info(f"Report saved to {output_path}")
    
    # Print a summary to console
    print("\n--- Resumen de Clusters ---")
    print(report_df[['cluster_id', 'size', 'percentage', 'representative_scenarios']].to_string(index=False))

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parquet = os.path.join(current_dir, "results", "probable_analysis.parquet")
    output = os.path.join(current_dir, "results", "cluster_report.csv")
    generate_report(parquet, output)
