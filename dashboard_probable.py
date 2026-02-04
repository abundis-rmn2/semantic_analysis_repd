import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
from datetime import datetime

# Page config
st.set_page_config(page_title="Probable: Exploratory Analysis Dashboard", layout="wide")

st.title("🔍 Probable: Análisis Exploratorio de Desapariciones")
st.markdown("### Pipeline de Descubrimiento Evolutivo de Escenarios y Palabras Clave")

# Data loading
RESULTS_DIR = "results"
CONFIG_DIR = "config"
CSV_PATH = os.path.join(RESULTS_DIR, "probable_analysis.csv")
TAXONOMY_PATH = os.path.join(RESULTS_DIR, "narrative_taxonomy.json")
GOLDEN_PATH = os.path.join(CONFIG_DIR, "golden_taxonomy.json")

def load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def load_data():
    if os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH)
            return df
        except:
            return None
    return None

def save_taxonomy(tax_data):
    with open(TAXONOMY_PATH, 'w', encoding='utf-8') as f:
        json.dump(tax_data, f, indent=4, ensure_ascii=False)

df = load_data()
golden_tax = load_json(GOLDEN_PATH) or {"keywords": {}, "scenarios": []}
dynamic_tax = load_json(TAXONOMY_PATH) or {"keywords": {}, "scenarios": []}

# Merge logic for display
scenario_pool = golden_tax.get("scenarios", [])
existing_labels = {s['label'] for s in scenario_pool}
for s in dynamic_tax.get("scenarios", []):
    if s['label'] not in existing_labels:
        scenario_pool.append(s)

keyword_pool = golden_tax.get("keywords", {})
for fam, kws in dynamic_tax.get("keywords", {}).items():
    if fam in keyword_pool:
        keyword_pool[fam] = list(set(keyword_pool[fam] + kws))
    else:
        keyword_pool[fam] = kws

if df is not None:
    # Sidebar Metrics & Management
    st.sidebar.header("📊 Métricas Globales")
    st.sidebar.metric("Total de Registros", len(df))
    st.sidebar.metric("Escenarios Activos", len(scenario_pool))
    
    st.sidebar.divider()
    st.sidebar.subheader("➕ Agregar Escenario Manual")
    with st.sidebar.form("add_scenario_form"):
        new_label = st.text_input("Etiqueta (snake_case)", placeholder="ej. reclutamiento_redes")
        new_desc = st.text_area("Descripción", placeholder="Describe las señales de este escenario...")
        if st.form_submit_button("Guardar Escenario"):
            if new_label and new_desc:
                # Basic cleaning
                clean_label = new_label.lower().replace(" ", "_")
                if not any(s['label'] == clean_label for s in scenario_pool):
                    # Add to dynamic
                    new_scen = {
                        "label": clean_label,
                        "description": new_desc,
                        "type": "manual"
                    }
                    dynamic_tax["scenarios"] = dynamic_tax.get("scenarios", []) + [new_scen]
                    save_taxonomy(dynamic_tax)
                    st.sidebar.success(f"¡Escenario '{clean_label}' agregado!")
                    st.rerun()
                else:
                    st.sidebar.error("Ese label ya existe.")
            else:
                st.sidebar.warning("Completa ambos campos.")

    # Layout: Top Row for Pools
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧬 Pool de Escenarios Evolutivos")
        if scenario_pool:
            scenario_df = pd.DataFrame(scenario_pool)
            st.dataframe(scenario_df, use_container_width=True)
        else:
            st.info("No se han descubierto escenarios aún.")

    with col2:
        st.subheader("🔑 Pool Dinámico de Palabras Clave")
        if keyword_pool:
            family = st.selectbox("Seleccionar Familia", list(keyword_pool.keys()))
            keywords = keyword_pool[family]
            st.write(f"**Palabras clave en {family}:**")
            st.write(", ".join(keywords))
        else:
            st.info("No se han descubierto palabras clave aún.")

    # Main Analysis Section
    st.divider()
    st.subheader("📄 Case Explorer")

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📄 Case Explorer", "🕸️ Graph Visualization", "🧬 Cluster Analysis"])

    with tab1:
        # Filters
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        municipios = ["Todos"] + sorted(df["municipio"].unique().tolist())
        with f_col1:
            sel_mun = st.selectbox("Filtrar por Municipio", municipios)
        
        with f_col3:
            sel_conf = st.multiselect("Filtrar por Confianza", ["alta", "media", "baja"], default=["alta"])

        # First, filter by Municipio and Confidence to determine available labels
        temp_df = df.copy()
        if sel_mun != "Todos":
            temp_df = temp_df[temp_df["municipio"] == sel_mun]
        
        available_labels = set()
        for s_str in temp_df["scenarios"].dropna():
            try:
                scs = json.loads(s_str)
                for s in scs:
                    if not sel_conf or s["scenario_confidence"] in sel_conf:
                        available_labels.add(s["scenario_label"])
            except:
                pass
        
        with f_col2:
            sel_scen = st.selectbox("Filtrar por Escenario", ["Todos"] + sorted(list(available_labels)))

        # Final filtering logic
        def match_criteria(s_str):
            try:
                scs = json.loads(s_str)
                for s in scs:
                    label_ok = (sel_scen == "Todos" or s["scenario_label"] == sel_scen)
                    conf_ok = (not sel_conf or s["scenario_confidence"] in sel_conf)
                    if label_ok and conf_ok:
                        return True
                return False
            except:
                return False

        filtered_df = temp_df[temp_df["scenarios"].apply(match_criteria)]

        st.write(f"Mostrando {len(filtered_df)} registros")
        
        for idx, row in filtered_df.iterrows():
            with st.expander(f"Caso {row['id_original']} - {row['municipio']}"):
                st.write("**Narrativa:**")
                st.write(row["raw_text"])
                
                scs = []
                try:
                    scs = json.loads(row["scenarios"])
                except:
                    pass
                
                if scs:
                    st.write("**Escenarios Identificados:**")
                    sc_cols = st.columns(len(scs))
                    for i, s in enumerate(scs):
                        with sc_cols[i]:
                            st.markdown(f"**{s['scenario_label']}**")
                            st.write(f"Confianza: {s['scenario_confidence']}")
                            st.write(f"Notas: {s.get('notes', 'N/A')}")
                
                st.write("**Metadata (LLM):**")
                st.json(row["llm_meta"])

    with tab2:
        import networkx as nx
        from pyvis.network import Network
        import streamlit.components.v1 as components

        st.subheader("Redes de Relación Interactivas")
        
        g_col1, g_col2 = st.columns([1, 3])
        
        with g_col1:
            graph_type = st.radio("Tipo de Grafo", ["Red de Relaciones (Bipartita)", "Afinidad Narrativa (Caso a Caso)"])
            
            gml_file = "relationship_network.gml" if "Relaciones" in graph_type else "probable_graph.gml"
            gml_path = os.path.join(RESULTS_DIR, gml_file)

            if os.path.exists(gml_path):
                G_raw = nx.read_gml(gml_path)
                
                st.write("---")
                st.write("**Filtros**")
                
                # Always show all possible project types
                available_types = ['case', 'scenario', 'keyword', 'family']
                
                # Wrapped in a form to avoid real-time re-rendering of the graph
                with st.form("graph_filters_form"):
                    sel_types = st.multiselect("Tipos de Nodo Visibles", available_types, default=[t for t in available_types if t in ['case', 'scenario']])
                    
                    muns = sorted(list(set([d.get('municipio', 'N/A') for n, d in G_raw.nodes(data=True) if d.get('municipio')])))
                    sel_g_muns = st.multiselect("Filtrar por Municipio (Grafo)", ["Todos"] + muns, default=["Todos"])

                    # 3. Filter by Confidence (for Scenario edges)
                    sel_g_conf = st.multiselect("Confianza Escenarios", ["alta", "media", "baja"], default=["alta"])
                    
                    search_node = st.text_input("Buscar Nodo (ID/Etiqueta)")
                    
                    apply_btn = st.form_submit_button("Aplicar Cambios")

                if apply_btn or 'graph_initialized' not in st.session_state:
                    st.session_state['graph_initialized'] = True
                    # The filtering logic will run below

                # Apply Filters to NetworkX
                G = G_raw.copy()
                
                # A. Filter edges by confidence (for case-scenario links)
                edges_to_remove = []
                for u, v, d in G.edges(data=True):
                    conf = d.get('confidence')
                    if conf: # Only apply to edges that have a confidence attribute
                        if not sel_g_conf or conf not in sel_g_conf:
                            edges_to_remove.append((u, v))
                G.remove_edges_from(edges_to_remove)

                # B. Filter nodes by type, municipio, and search
                nodes_to_remove = []
                for n, d in G.nodes(data=True):
                    type_ok = d.get('type', 'case') in sel_types
                    mun_ok = "Todos" in sel_g_muns or d.get('municipio') in sel_g_muns
                    search_ok = not search_node or search_node.lower() in str(n).lower() or search_node.lower() in str(d.get('label', '')).lower()
                    
                    if not (type_ok and mun_ok and search_ok):
                        nodes_to_remove.append(n)
                
                G.remove_nodes_from(nodes_to_remove)
                
                if st.checkbox("Ocultar Nodos Aislados", value=True):
                    G.remove_nodes_from(list(nx.isolates(G)))

        with g_col2:
            if os.path.exists(gml_path):
                try:
                    net = Network(height="750px", width="100%", bgcolor="#1a1a1a", font_color="white")
                    net.force_atlas_2based()
                    
                    type_colors = {
                        'case': "#FF4B4B",
                        'scenario': "#1C83E1",
                        'keyword': "#6FCF97",
                        'family': "#FFD21F",
                        'unknown': "#808080"
                    }

                    for node, data in G.nodes(data=True):
                        n_type = data.get('type', 'case')
                        color = type_colors.get(n_type, type_colors['unknown'])
                        label = data.get('label', node)
                        
                        title_text = f"TIPO: {n_type.upper()} | ID: {node}\n"
                        title_text += "-" * 30 + "\n"
                        
                        if n_type == 'case':
                            case_row = df[df['id_original'] == node]
                            if not case_row.empty:
                                narr = case_row.iloc[0]['raw_text']
                                title_text += f"NARRATIVA:\n{narr[:400]}...\n\n"
                                title_text += f"MUNICIPIO: {case_row.iloc[0]['municipio']}\n"
                        
                        elif n_type == 'scenario':
                            scen_info = next((s for s in scenario_pool if s['label'] == node.replace("SCEN:", "")), None)
                            if scen_info:
                                title_text += f"DESCRIPCIÓN:\n{scen_info.get('description', 'N/A')}\n"
                        
                        for k, v in data.items():
                            if k not in ['type', 'label', 'municipio']:
                                title_text += f"{k.upper()}: {v}\n"
                        
                        net.add_node(node, label=label, title=title_text, color=color, 
                                     size=25 if n_type in ['case', 'scenario'] else 15,
                                     borderWidth=2)

                    for source, target, data in G.edges(data=True):
                        net.add_edge(source, target, value=data.get('weight', 1.0), color="rgba(200, 200, 200, 0.3)")

                    net.set_options("""
                    var options = {
                      "physics": {
                        "forceAtlas2Based": {
                          "gravitationalConstant": -50,
                          "centralGravity": 0.01,
                          "springLength": 100,
                          "springConstant": 0.08
                        },
                        "maxVelocity": 50,
                        "solver": "forceAtlas2Based",
                        "timestep": 0.35,
                        "stabilization": { "iterations": 150 }
                      }
                    }
                    """)

                    path = os.path.abspath("temp_graph.html")
                    net.save_graph(path)
                    
                    with open(path, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                        custom_style = """
                        <style>
                        div.vis-tooltip {
                            max-width: 350px !important;
                            white-space: pre-wrap !important;
                            word-wrap: break-word !important;
                            background-color: #2c2c2c !important;
                            color: #ffffff !important;
                            padding: 12px !important;
                            border: 1px solid #444 !important;
                            border-radius: 8px !important;
                            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
                            font-size: 13px !important;
                            line-height: 1.5 !important;
                            box-shadow: 0 4px 12px rgba(0,0,0,0.5) !important;
                        }
                        </style>
                        """
                        html_content = html_content.replace("</head>", f"{custom_style}</head>")
                        components.html(html_content, height=850, scrolling=True)
                    
                except Exception as e:
                    st.error(f"Error renderizando el grafo: {e}")
            else:
                st.info(f"Archivo de grafo no encontrado: {gml_file}")

    with tab3:
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np

        st.subheader("🧬 Clusters Neurales (Grupos Semánticos)")
        st.markdown("""
        Esta vista agrupa casos por su **alma narrativa** (embeddings). 
        Permite identificar patrones emergentes que la taxonomía clásica podría ignorar.
        """)

        if 'cluster_label' in df.columns:
            clusters_df = df[df['cluster_label'] != -1]
            n_clusters = clusters_df['cluster_label'].nunique()
            
            if n_clusters > 0:
                # TF-IDF Calculation for Keyword Discrimination
                all_cluster_docs = []
                cluster_ids = sorted(clusters_df['cluster_label'].unique())
                
                for cid in cluster_ids:
                    c_rows = clusters_df[clusters_df['cluster_label'] == cid]
                    kws_list = []
                    for k in c_rows['keywords'].dropna():
                        if isinstance(k, str):
                            try: k = json.loads(k)
                            except: k = k.split(", ")
                        kws_list.extend([kw.replace(" ", "_") for kw in k]) # Use snake_case for TF-IDF
                    all_cluster_docs.append(" ".join(kws_list))

                vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
                tfidf_matrix = vectorizer.fit_transform(all_cluster_docs)
                feature_names = vectorizer.get_feature_names_out()

                cluster_id = st.select_slider("Seleccionar ID de Cluster", options=cluster_ids)
                c_idx = cluster_ids.index(cluster_id)
                c_data = clusters_df[clusters_df['cluster_label'] == cluster_id]
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.metric("Casos en este Cluster", len(c_data))
                    
                    # Scenario Distribution (Priority)
                    all_scs = []
                    for s_str in c_data['scenarios'].dropna():
                        try:
                            scs = json.loads(s_str)
                            all_scs.extend([s['scenario_label'] for s in scs if s['scenario_confidence'] == 'alta'])
                        except: pass
                    
                    if all_scs:
                        st.write("**Distribución de Escenarios:**")
                        sc_counts = pd.Series(all_scs).value_counts()
                        st.bar_chart(sc_counts)
                    else:
                        st.info("Sin escenarios con confianza suficiente.")
                
                with col_b:
                    # Discriminating Keywords (TF-IDF)
                    st.write("**🔑 Palabras Clave Discriminantes (TF-IDF):**")
                    st.caption("Palabras que describen la singularidad de este cluster frente a los demás.")
                    
                    row_data = tfidf_matrix.getrow(c_idx).toarray()[0]
                    top_kw_indices = row_data.argsort()[-15:][::-1]
                    top_kws = {feature_names[i]: row_data[i] for i in top_kw_indices if row_data[i] > 0}
                    
                    if top_kws:
                        kw_df = pd.DataFrame(list(top_kws.items()), columns=['Keyword', 'Score']).sort_values('Score', ascending=False)
                        fig = px.bar(kw_df, x='Score', y='Keyword', orientation='h', color='Score', 
                                     color_continuous_scale='Viridis', title="Top TF-IDF Keywords")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No se hallaron palabras clave discriminantes.")

                st.divider()
                st.subheader("📖 Elementos Clave del Relato")
                st.markdown("Basado en el análisis de divergencia, estos son los elementos que definen este grupo:")
                
                # Show top 5 keywords as bullet points
                top_5 = list(top_kws.keys())[:5]
                if top_5:
                    cols = st.columns(len(top_5))
                    for i, kw in enumerate(top_5):
                        cols[i].info(f"**{kw.replace('_', ' ').upper()}**")

                st.write("---")
                st.write(f"**Explorador de Casos del Cluster {cluster_id}:**")
                st.dataframe(c_data[['id_original', 'municipio', 'raw_text']], use_container_width=True)
            else:
                st.info("No se han detectado clusters distintos aún.")
        else:
            st.warning("Datos de clustering no encontrados. Ejecute el pipeline con embeddings habilitados.")

else:
    st.warning("No se encontraron datos en el directorio de resultados. Ejecute el pipeline primero.")
    st.code("python process_probable.py --sample 50 --batch-size 10")
