import streamlit as st
import pandas as pd
import json
import os
import networkx as nx
import streamlit.components.v1 as components
from pyvis.network import Network

st.set_page_config(page_title="Probable: Human Triage", layout="wide")

@st.cache_data
def load_data(file_path):
    if os.path.exists(file_path):
        return pd.read_parquet(file_path)
    return None

st.title("🔍 Probable: Triage de Desapariciones")
st.markdown("Revisión de hipótesis generadas por el pipeline exploratorio.")

# Sidebar for config
results_dir = st.sidebar.text_input("Directorio de Resultados", "results")
parquet_file = os.path.join(results_dir, "probable_analysis.parquet")

df = load_data(parquet_file)

if df is None:
    st.error(f"No se encontró el archivo: {parquet_file}")
else:
    
    # Filter controls
    st.sidebar.subheader("Filtros")
    min_ambiguity = st.sidebar.slider("Ambigüedad mínima", 0.0, 1.0, 0.0)
    
    # Cluster filter
    clusters = sorted(df['cluster_label'].unique()) if 'cluster_label' in df.columns else []
    selected_cluster = st.sidebar.selectbox("Filtrar por Cluster (-1 = Ruido)", ["Todos"] + clusters)
    
    # Scenario filter
    selected_scenario = st.sidebar.selectbox("Escenario (Alta Confianza)", ["Todos"] + sorted(list(set([s['scenario_label'] for sc_list in df['scenarios'] for s in sc_list if s['scenario_confidence'] == 'alta']))))

    # Apply filters
    filtered_df = df[df['ambiguity_score'] >= min_ambiguity]
    if selected_cluster != "Todos":
        filtered_df = filtered_df[filtered_df['cluster_label'] == selected_cluster]
    if selected_scenario != "Todos":
        filtered_df = filtered_df[filtered_df['scenarios'].apply(lambda x: any(s['scenario_label'] == selected_scenario and s['scenario_confidence'] == 'alta' for s in x))]

    st.write(f"Mostrando {len(filtered_df)} de {len(df)} registros.")

    # Cluster Analytics Tab
    tab1, tab2, tab3 = st.tabs(["Listado de Casos", "Analítica de Clusters", "Red de Afinidades"])
    
    with tab3:
        st.subheader("Visualización de la Red de Afinidades")
        if st.button("Cargar / Actualizar Red (Lento con >1k nodos)"):
            gml_file = os.path.join(results_dir, "probable_graph.gml")
            if not os.path.exists(gml_file):
                st.warning("No se ha generado el grafo de afinidades aún.")
            else:
                G = nx.read_gml(gml_file)
                if G.number_of_nodes() == 0:
                    st.info("El grafo está vacío.")
                else:
                    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white", notebook=False)
                    
                    # Add nodes with colors based on cluster
                    for node, data in G.nodes(data=True):
                        cluster_id = data.get('cluster', -1)
                        # Simple color palette
                        colors = ["#FF4B4B", "#1C83E1", "#00C0F2", "#F0F2F6", "#FFD21F", "#808080"]
                        color = colors[int(cluster_id) % len(colors)] if cluster_id != -1 else "#808080"
                        
                        label = f"ID: {node}\nCP: {data.get('municipio', 'N/A')}"
                        net.add_node(node, label=label, title=label, color=color)
                    
                    # Add edges
                    for source, target, data in G.edges(data=True):
                        weight = data.get('weight', 1.0)
                        reasons = data.get('reasons', '')
                        net.add_edge(source, target, value=weight, title=f"Peso: {weight}\nRazones: {reasons}")
                    
                    # Save and display
                    path = os.path.join(results_dir, "temp_graph.html")
                    net.save_graph(path)
                    
                    with open(path, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                        components.html(html_content, height=620)
                    
                    st.caption("Los colores representan clusters. Los grosores de las aristas indican fuerza de afinidad.")

    with tab2:
        st.subheader("Distribución de Escenarios por Cluster")
        if 'cluster_label' in df.columns:
            # Aggregate scenarios per cluster
            cluster_id = st.selectbox("Selecciona Cluster para detalle", clusters)
            cluster_data = df[df['cluster_label'] == cluster_id]
            
            sc_list = [s['scenario_label'] for sc_list in cluster_data['scenarios'] for s in sc_list if s['scenario_confidence'] == 'alta']
            if sc_list:
                counts = pd.Series(sc_list).value_counts()
                st.bar_chart(counts)
            else:
                st.write("No hay escenarios de alta confianza en este cluster.")
        
        st.subheader("Cohortes Temporales (Macroclases)")
        if 'date_dt' in df.columns:
            from clustering_utils import evaluate_temporal_cohorts
            cohorts = evaluate_temporal_cohorts(df)
            if cohorts:
                cohort_df = pd.DataFrame(cohorts)
                st.write(cohort_df)

    with tab1:
        # Pagination
        items_per_page = 20
        total_pages = (len(filtered_df) // items_per_page) + (1 if len(filtered_df) % items_per_page > 0 else 0)
        
        if total_pages > 1:
            page = st.number_input(f"Página (1-{total_pages})", min_value=1, max_value=total_pages, value=1)
        else:
            page = 1
            
        start_idx = (page - 1) * items_per_page
        end_idx = start_idx + items_per_page
        
        # Table view
        for idx, row in filtered_df.iloc[start_idx:end_idx].iterrows():
            with st.expander(f"Caso: {row['id_original']} | Municipio: {row['municipio']} | Ambigüedad: {row['ambiguity_score']:.2f}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("Relato")
                    st.write(row['raw_text'])
                    
                    st.subheader("Hipótesis")
                    for s in row['scenarios']:
                        color = "red" if s['scenario_confidence'] == 'alta' else "orange" if s['scenario_confidence'] == 'media' else "gray"
                        st.markdown(f"**:{color}[{s['scenario_label'].upper()}]** (Confianza: {s['scenario_confidence']})")
                        st.write(f"*Señales:* {', '.join(s['supporting_signals'])}")
                        st.info(f"Evidencia: {', '.join(s['text_cites'])}")
                        if s.get('notes'):
                            st.caption(f"Notas: {s['notes']}")
                
                with col2:
                    st.subheader("Observables")
                    st.json(row['observables'])
                    
                    st.subheader("Metadatos LLM")
                    st.json(row['llm_meta'])

    # Export for manual labeling
    if st.button("Preparar para Label Studio"):
        export_df = filtered_df[['id_original', 'raw_text', 'scenarios']].copy()
        export_path = os.path.join(results_dir, "label_studio_export.csv")
        export_df.to_csv(export_path, index=False)
        st.success(f"Exportado a {export_path}")
