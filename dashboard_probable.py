import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
from datetime import datetime

# Page config
st.set_page_config(page_title="Probable: Exploratory Analysis Dashboard", layout="wide")

st.title("🔍 Probable: Exploratory Disappearance Analysis")
st.markdown("### Evolutionary Scenario & Keyword Discovery Pipeline")

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
    st.sidebar.header("📊 Global Metrics")
    st.sidebar.metric("Total Records", len(df))
    st.sidebar.metric("Active Scenarios", len(scenario_pool))
    
    st.sidebar.divider()
    st.sidebar.subheader("➕ Add Manual Scenario")
    with st.sidebar.form("add_scenario_form"):
        new_label = st.text_input("Label (snake_case)", placeholder="ej. reclutamiento_redes")
        new_desc = st.text_area("Description", placeholder="Describe las señales de este escenario...")
        if st.form_submit_button("Save Scenario"):
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
                    st.sidebar.success(f"Scenario '{clean_label}' added!")
                    st.rerun()
                else:
                    st.sidebar.error("Ese label ya existe.")
            else:
                st.sidebar.warning("Completa ambos campos.")

    # Layout: Top Row for Pools
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧬 Evolutionary Scenario Pool")
        if scenario_pool:
            scenario_df = pd.DataFrame(scenario_pool)
            st.dataframe(scenario_df, use_container_width=True)
        else:
            st.info("No scenarios discovered yet.")

    with col2:
        st.subheader("🔑 Dynamic Keyword Pool")
        if keyword_pool:
            family = st.selectbox("Select Family", list(keyword_pool.keys()))
            keywords = keyword_pool[family]
            st.write(f"**Keywords in {family}:**")
            st.write(", ".join(keywords))
        else:
            st.info("No keywords discovered yet.")

    # Main Analysis Section
    st.divider()
    st.subheader("📄 Case Explorer")

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📄 Case Explorer", "🕸️ Graph Visualization", "🧬 Cluster Analysis"])

    with tab1:
        # Filters
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        municipios = ["All"] + sorted(df["municipio"].unique().tolist())
        with f_col1:
            sel_mun = st.selectbox("Filter by Municipio", municipios)
        
        with f_col3:
            sel_conf = st.multiselect("Filter by Confidence", ["alta", "media", "baja"], default=["alta"])

        # First, filter by Municipio and Confidence to determine available labels
        temp_df = df.copy()
        if sel_mun != "All":
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
            # Only show labels that have at least 1 record given the other filters
            sel_scen = st.selectbox("Filter by Scenario", ["All"] + sorted(list(available_labels)))

        # Final filtering logic
        def match_criteria(s_str):
            try:
                scs = json.loads(s_str)
                # A record matches if it has AT LEAST one scenario that satisfies BOTH filters
                for s in scs:
                    label_ok = (sel_scen == "All" or s["scenario_label"] == sel_scen)
                    conf_ok = (not sel_conf or s["scenario_confidence"] in sel_conf)
                    if label_ok and conf_ok:
                        return True
                return False
            except:
                return False

        filtered_df = temp_df[temp_df["scenarios"].apply(match_criteria)]

        st.write(f"Showing {len(filtered_df)} records")
        
        # Custom display for cases
        for idx, row in filtered_df.iterrows():
            with st.expander(f"Case {row['id_original']} - {row['municipio']}"):
                st.write("**Narrative:**")
                st.write(row["raw_text"])
                
                scs = []
                try:
                    scs = json.loads(row["scenarios"])
                except:
                    pass
                
                if scs:
                    st.write("**Identified Scenarios:**")
                    sc_cols = st.columns(len(scs))
                    for i, s in enumerate(scs):
                        with sc_cols[i]:
                            st.markdown(f"**{s['scenario_label']}**")
                            st.write(f"Confidence: {s['scenario_confidence']}")
                            st.write(f"Notes: {s.get('notes', 'N/A')}")
                
                st.write("**Metadata:**")
                st.json(row["llm_meta"])

    with tab2:
        import networkx as nx
        from pyvis.network import Network
        import streamlit.components.v1 as components

        st.subheader("Interactive Relationship Networks")
        
        g_col1, g_col2 = st.columns([1, 3])
        
        with g_col1:
            graph_type = st.radio("Select Graph Type", ["Relationship Network (Bipartite)", "Narrative Affinity (Case-to-Case)"])
            
            gml_file = "relationship_network.gml" if "Bipartite" in graph_type else "probable_graph.gml"
            gml_path = os.path.join(RESULTS_DIR, gml_file)

            if os.path.exists(gml_path):
                # Load graph
                G_raw = nx.read_gml(gml_path)
                
                # Dynamic Filters
                st.write("---")
                st.write("**Filters**")
                
                # 1. Filter by Type (for Bipartite)
                available_types = sorted(list(set(nx.get_node_attributes(G_raw, 'type').values())))
                if not available_types: available_types = ['case']
                
                sel_types = st.multiselect("Visible Node Types", available_types, default=available_types)
                
                # 2. Filter by Municipio (if exists)
                muns = sorted(list(set([d.get('municipio', 'N/A') for n, d in G_raw.nodes(data=True) if d.get('municipio')])))
                sel_g_muns = st.multiselect("Filter by Municipio", ["All"] + muns, default=["All"])
                
                # Search
                search_node = st.text_input("Search Node (ID/Label)")

                # Apply Filters to NetworkX
                nodes_to_keep = []
                for n, d in G_raw.nodes(data=True):
                    type_ok = d.get('type', 'case') in sel_types
                    mun_ok = "All" in sel_g_muns or d.get('municipio') in sel_g_muns
                    search_ok = not search_node or search_node.lower() in str(n).lower() or search_node.lower() in str(d.get('label', '')).lower()
                    
                    if type_ok and mun_ok and search_ok:
                        nodes_to_keep.append(n)
                
                G = G_raw.subgraph(nodes_to_keep).copy()
                
                # Remove isolated nodes if requested
                if st.checkbox("Hide Isolated Nodes", value=True):
                    G.remove_nodes_from(list(nx.isolates(G)))

        with g_col2:
            if os.path.exists(gml_path):
                try:
                    # Setup PyVis
                    net = Network(height="750px", width="100%", bgcolor="#1a1a1a", font_color="white")
                    net.force_atlas_2based()
                    
                    # Colors
                    type_colors = {
                        'case': "#FF4B4B",     # Red
                        'scenario': "#1C83E1", # Blue
                        'keyword': "#6FCF97",  # Green
                        'family': "#FFD21F",   # Yellow
                        'unknown': "#808080"
                    }

                    # Add nodes with enriched data
                    for node, data in G.nodes(data=True):
                        n_type = data.get('type', 'case')
                        color = type_colors.get(n_type, type_colors['unknown'])
                        label = data.get('label', node)
                        
                        # BUILD ENRICHED TOOLTIP (Title)
                        title_text = f"TYPE: {n_type.upper()} | ID: {node}\n"
                        title_text += "-" * 30 + "\n"
                        
                        if n_type == 'case':
                            # Try to find narrative in main df
                            case_row = df[df['id_original'] == node]
                            if not case_row.empty:
                                narr = case_row.iloc[0]['raw_text']
                                title_text += f"NARRATIVA:\n{narr[:400]}...\n\n"
                                title_text += f"MUNICIPIO: {case_row.iloc[0]['municipio']}\n"
                        
                        elif n_type == 'scenario':
                            # Try to find description in scenario pool
                            scen_info = next((s for s in scenario_pool if s['label'] == node.replace("SCEN:", "")), None)
                            if scen_info:
                                title_text += f"DESCRIPCIÓN:\n{scen_info.get('description', 'N/A')}\n"
                        
                        # Add other metadata
                        for k, v in data.items():
                            if k not in ['type', 'label', 'municipio']:
                                title_text += f"{k.upper()}: {v}\n"
                        
                        net.add_node(node, label=label, title=title_text, color=color, 
                                     size=25 if n_type in ['case', 'scenario'] else 15,
                                     borderWidth=2)

                    # Add edges
                    for source, target, data in G.edges(data=True):
                        net.add_edge(source, target, value=data.get('weight', 1.0), color="rgba(200, 200, 200, 0.3)")

                    # Options for interaction
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

                    # Save and embed
                    path = os.path.abspath("temp_graph.html")
                    net.save_graph(path)
                    
                    with open(path, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                        
                        # INJECT CUSTOM CSS FOR TOOLTIPS
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
                        # Insert style before closing head tag
                        html_content = html_content.replace("</head>", f"{custom_style}</head>")
                        
                        components.html(html_content, height=850, scrolling=True)
                    
                except Exception as e:
                    st.error(f"Error rendering graph: {e}")
            else:
                st.info(f"Graph file not found: {gml_file}. Please run the pipeline first.")

    with tab3:
        st.subheader("🧬 Neural Clusters (Semantic Groups)")
        st.markdown("""
        Esta vista agrupa casos por su **alma narrativa** (embeddings). 
        Los clusters son generados automáticamente por el algoritmo HDBSCAN.
        """)

        if 'cluster_label' in df.columns:
            # Filter out noise (-1)
            clusters_df = df[df['cluster_label'] != -1]
            n_clusters = clusters_df['cluster_label'].nunique()
            
            if n_clusters > 0:
                cluster_id = st.select_slider("Select Cluster ID", options=sorted(clusters_df['cluster_label'].unique()))
                
                c_data = clusters_df[clusters_df['cluster_label'] == cluster_id]
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.metric("Cases in Cluster", len(c_data))
                    
                    # Dominant Scenarios in Cluster
                    all_scs = []
                    for s_str in c_data['scenarios'].dropna():
                        try:
                            scs = json.loads(s_str)
                            all_scs.extend([s['scenario_label'] for s in scs if s['scenario_confidence'] in ['alta', 'media']])
                        except: pass
                    
                    if all_scs:
                        st.write("**Dominant Scenarios:**")
                        sc_counts = pd.Series(all_scs).value_counts()
                        st.bar_chart(sc_counts)
                
                with col_b:
                    # Dominant Keywords in Cluster
                    all_kws = []
                    for kws in c_data['keywords'].dropna():
                        if isinstance(kws, str):
                            try: kws = json.loads(kws)
                            except: kws = kws.split(", ")
                        all_kws.extend(kws)
                    
                    if all_kws:
                        st.write("**Top Keywords in Cluster:**")
                        kw_counts = pd.Series(all_kws).value_counts().head(10)
                        fig = px.pie(names=kw_counts.index, values=kw_counts.values, hole=0.4)
                        st.plotly_chart(fig, use_container_width=True)

                st.write("---")
                st.write(f"**Cases in Cluster {cluster_id}:**")
                st.dataframe(c_data[['id_original', 'municipio', 'raw_text']], use_container_width=True)
            else:
                st.info("No distinct clusters detected yet. The current results might be too diverse or small.")
        else:
            st.warning("Clustering data not found. Please run the pipeline with embeddings enabled.")

else:
    st.warning("No data found in results directory. Run the pipeline first.")
    st.code("python process_probable.py --sample 50 --batch-size 10")
