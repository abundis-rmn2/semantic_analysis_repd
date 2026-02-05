import streamlit as st
import pandas as pd
import json
import os
import networkx as nx
import plotly.express as px
from pyvis.network import Network
import streamlit.components.v1 as components
from datetime import datetime
import time
import logging

# Setup Logger for Debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("probable_dashboard")

# Page config
st.set_page_config(page_title="Probable: Exploratory Analysis Dashboard", layout="wide")

def nx_to_sigma_json(G: nx.Graph, df: pd.DataFrame = None) -> dict:
    """Convierte un NetworkX graph a la estructura que espera Sigma.js."""
    # Create lookup dictionaries for better metadata enrichment
    id_to_text = {}
    id_to_mun = {}
    if df is not None:
        id_to_text = dict(zip(df['id_original'].astype(str), df['raw_text']))
        id_to_mun = dict(zip(df['id_original'].astype(str), df['municipio']))

    # Calculate degrees for sizing (importance)
    # Using G.in_degree if directed, else G.degree
    if G.is_directed():
        degrees = dict(G.in_degree())
    else:
        degrees = dict(G.degree())

    nodes = []
    for n, data in G.nodes(data=True):
        n_type = data.get("type", "unknown")
        meta = {**data}
        deg = degrees.get(n, 1)
        
        # Determine Label (Only scenario/blue nodes have labels now)
        if n_type == "scenario":
            label = data.get("label", str(n))
            # Blue nodes size by degree (reduced to half as requested)
            size = 4 + (deg * 0.75)
        elif n_type == "case":
            label = "" 
            size = 8
        else:
            label = "" # No label for Keywords or Families either
            size = 4 + (deg * 0.5)

        # Enrich based on type
        if n_type == "case":
            meta["raw_text"] = id_to_text.get(str(n), data.get("raw_text", ""))
            meta["municipio"] = id_to_mun.get(str(n), data.get("municipio", ""))
        elif n_type == "scenario":
            label_clean = str(n).replace("SCEN:", "")
            meta["description"] = data.get("description", f"Patrón detectado: {label_clean}")
        elif n_type == "keyword":
            label_clean = str(n).replace("KW:", "")
            meta["description"] = f"Término clave extraído: {label_clean}"
        elif n_type == "family":
            label_clean = str(n).replace("FAM:", "")
            meta["description"] = f"Agrupación temática: {label_clean}"

        nodes.append({
            "id": str(n),
            "label": label,
            "x": data.get("x", 0.0),
            "y": data.get("y", 0.0),
            "size": size,
            "color": {
                "border": "#ffffff",
                "background": {
                    "case": "#FF4B4B",
                    "scenario": "#1C83E1",
                    "keyword": "#6FCF97",
                    "family": "#FFD21F",
                    "unknown": "#808080"
                }.get(n_type, "#808080")
            },
            "type": n_type,
            "metadata": meta
        })
    
    edges = [
        {
            "id": f"{source}-{target}",
            "source": str(source),
            "target": str(target),
            "size": data.get("weight", 1.0),
            "color": "#ccc",
            "type": "arrow" if data.get("directed", False) else "line"
        }
        for source, target, data in G.edges(data=True)
    ]
    return {"nodes": nodes, "edges": edges}

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

@st.cache_data
def load_graph_cached(path):
    start = time.time()
    if os.path.exists(path):
        G = nx.read_gml(path)
        logger.info(f"PERF: Grafo cargado desde {path} - Duración: {time.time() - start:.4f}s")
        return G
    return nx.Graph()

def save_taxonomy(tax_data):
    with open(TAXONOMY_PATH, 'w', encoding='utf-8') as f:
        json.dump(tax_data, f, indent=4, ensure_ascii=False)

# Sidebar Metrics & Management
st.sidebar.header("🛠️ Configuración de Sistema")

def log_debug(message, duration=None):
    time_suffix = f" ({duration:.4f}s)" if duration is not None else ""
    full_message = f"{message}{time_suffix}"
    logger.info(full_message)

# Data loading with Performance Tracking
start_time = time.time()

with st.status("Cargando base de conocimientos...", expanded=False) as status:
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
    
    status.update(label=f"Datos listos", state="complete", expanded=False)

log_debug(f"Carga de datos completada: {len(df) if df is not None else 0} filas", duration=time.time() - start_time)

if df is not None:
    # Sidebar Metrics & Management
    st.sidebar.header("📊 Métricas Globales")
    st.sidebar.metric("Total de Registros", len(df))
    st.sidebar.metric("Escenarios Activos", len(scenario_pool))
    
    # --- LAZY LOADING NAVIGATION ---
    st.sidebar.divider()
    menu = st.sidebar.radio(
        "📂 Seleccionar Herramienta", 
        ["ℹ️ Inicio", "📄 Explorador de Casos", "🌐 Visualización JS", "🐍 Visualización Python", "🧬 Análisis de Clusters"],
        help="Los datos se cargarán solo al seleccionar la sección."
    )
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

    st.divider()
    st.subheader("📄 Case Explorer")

    if menu == "ℹ️ Inicio":
        st.markdown(f"""
        ## 🚨 Proyecto Probable: Inteligencia Semántica
        Bienvenido al sistema de análisis exploratorio para la crisis de desapariciones. 
        Este sistema utiliza IA para procesar narrativas y descubrir patrones ocultos.
        
        ### 🛠️ Herramientas disponibles:
        1.  **📄 Case Explorer:** Análisis de texto y metadata por caso.
        2.  **🕸️ Graph Visualization:** Redes de relaciones (Carga archivos GML pesados).
        3.  **🧬 Cluster Analysis:** Agrupación semántica por IA.
        
        ---
        **Estado Global:**
        *   Sistema Operativo: Listo
        """)
        
        if df is None:
            st.error("Archivo `probable_analysis.csv` no detectado.")
        else:
            st.success("Conexión con base de datos establecida. Selecciona una herramienta en el menú de la izquierda.")

    elif menu == "📄 Explorador de Casos":
        # Filters
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        municipios = ["Todos"] + sorted(df["municipio"].unique().tolist())
        with f_col1:
            sel_mun = st.selectbox("Filtrar por Municipio", municipios)
        
        with f_col3:
            sel_conf = st.multiselect("Filtrar por Confianza", ["alta", "media", "baja"], default=["alta"])

        # Search Bar
        with f_col4:
            search_query = st.text_input("🔍 Buscar en narrativa/ID", placeholder="ej. camioneta blanca...")

        # Optimized filtering logic
        # 1. Filter by Municipio
        temp_df = df.copy()
        if sel_mun != "Todos":
            temp_df = temp_df[temp_df["municipio"] == sel_mun]
        
        # 2. Pre-parse scenarios for fast filtering (Caching would be better but this is 500 rows)
        def get_filtered_scenarios(s_str):
            try:
                scs = json.loads(s_str)
                return [s for s in scs if not sel_conf or s["scenario_confidence"] in sel_conf]
            except:
                return []
        
        temp_df['parsed_scs'] = temp_df['scenarios'].apply(get_filtered_scenarios)
        
        # Determine available labels based on current filters
        available_labels = sorted(list(set([s['scenario_label'] for scs in temp_df['parsed_scs'] for s in scs])))
        
        with f_col2:
            sel_scen = st.selectbox("Filtrar por Escenario", ["Todos"] + available_labels)

        # 3. Apply Final Filters
        def final_filter(row):
            # Scenario check
            if sel_scen != "Todos":
                if not any(s['scenario_label'] == sel_scen for s in row['parsed_scs']):
                    return False
            else:
                if not row['parsed_scs'] and sel_conf: # If confidence filtered but no scenarios match
                    return False
            
            # Search check
            if search_query:
                q = search_query.lower()
                text_match = q in str(row['raw_text']).lower()
                id_match = q in str(row['id_original']).lower()
                if not (text_match or id_match):
                    return False
            
            return True

        filtered_df = temp_df[temp_df.apply(final_filter, axis=1)]

        # --- PAGINATION ---
        st.write(f"Mostrando {len(filtered_df)} registros")
        items_per_page = 20
        total_pages = max(1, (len(filtered_df) - 1) // items_per_page + 1)
        
        p_col1, p_col2 = st.columns([1, 4])
        with p_col1:
            page_num = st.number_input("Página", min_value=1, max_value=total_pages, step=1, value=1)
        
        start_idx = (page_num - 1) * items_per_page
        end_idx = start_idx + items_per_page
        page_df = filtered_df.iloc[start_idx:end_idx]
        
        for idx, row in page_df.iterrows():
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

    elif menu == "🌐 Visualización JS":
        st.subheader("Redes de Relación Interactivas – Sigma.js")

        g_col1, g_col2 = st.columns([1, 3])

        with g_col1:
            graph_type = st.radio("Tipo de Grafo", ["Red de Relaciones (Bipartita)", "Afinidad Narrativa (Caso a Caso)"])
            gml_file = "relationship_network.gml" if "Relaciones" in graph_type else "probable_graph.gml"
            gml_path = os.path.join(RESULTS_DIR, gml_file)

            if os.path.exists(gml_path):
                g_start = time.time()
                
                with st.status(f"Procesando red: {graph_type}...", expanded=False) as g_status:
                    G_raw = load_graph_cached(gml_path)
                    
                    available_types = ['case', 'scenario', 'keyword', 'family']
                    
                    muns = sorted(list(set([d.get('municipio', 'N/A') for n, d in G_raw.nodes(data=True) if d.get('municipio')])))
                    with st.form("graph_filters_form"):
                        sel_types = st.multiselect("Tipos de Nodo Visibles", available_types, default=[t for t in available_types if t in ['case', 'scenario']])
                        sel_g_muns = st.multiselect("Filtrar por Municipio (Grafo)", ["Todos"] + muns, default=["Todos"])
                        sel_g_conf = st.multiselect("Confianza Escenarios", ["alta", "media", "baja"], default=["alta"])
                        search_node = st.text_input("Buscar Nodo (ID/Etiqueta)")
                        apply_btn = st.form_submit_button("Aplicar Cambios")
                    
                    if apply_btn or 'graph_initialized' not in st.session_state:
                        st.session_state['graph_initialized'] = True
                    
                    G = G_raw.copy()
                    
                    # Edge confidence filter
                    edges_to_remove = []
                    for u, v, d in G.edges(data=True):
                        conf = d.get('confidence')
                        if conf and (not sel_g_conf or conf not in sel_g_conf):
                            edges_to_remove.append((u, v))
                    G.remove_edges_from(edges_to_remove)
                    
                    # Node filters
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
                    
                    g_status.update(label=f"Red procesada", state="complete")
                    log_debug(f"Renderizado final: {len(G.nodes)} nodos, {len(G.edges)} aristas", duration=time.time() - g_start)

        with g_col2:
            if os.path.exists(gml_path):
                try:
                    # Pass the dataframe to enrich case descriptions
                    sigma_data = nx_to_sigma_json(G, df=df)
                    sigma_json = json.dumps(sigma_data)
                    sigma_html = f"""
<!doctype html>
<html lang="es">
<head>
    <meta charset="utf-8">
    <style>
        #container {{ width: 100vw; height: 800px; background-color: #0d0d0d; margin: 0; padding: 0; }}
        #loader {{ 
            position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); 
            color: #888; font-family: sans-serif; text-align: center; z-index: 10;
        }}
        .tooltip {{
            position: absolute; display: none; background: rgba(25, 25, 25, 0.95); color: #ddd;
            padding: 12px; border-radius: 6px; font-size: 12px; z-index: 1000; border: 1px solid #444;
            pointer-events: auto; max-width: 320px; box-shadow: 0 4px 25px rgba(0,0,0,0.8);
            font-family: sans-serif; transition: opacity 0.2s;
        }}
        .tooltip .close-btn {{
            float: right; cursor: pointer; color: #ff4b4b; font-weight: bold; margin-left: 10px;
        }}
        #controls {{
            position: absolute; top: 20px; left: 20px; z-index: 2000;
            display: flex; gap: 8px; align-items: center;
        }}
        .btn {{
            background: rgba(28, 131, 225, 0.2); 
            color: #1C83E1; 
            border: 1px solid #1C83E1; 
            padding: 8px 14px;
            border-radius: 20px; 
            cursor: pointer; 
            font-family: 'Inter', sans-serif; 
            font-size: 11px;
            font-weight: 600; 
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            backdrop-filter: blur(5px);
        }}
        .btn:hover {{ 
            background: #1C83E1; 
            color: white; 
            box-shadow: 0 0 15px rgba(28, 131, 225, 0.5);
            transform: translateY(-2px); 
        }}
        .btn.paused, .btn.active {{ 
            background: rgba(255, 75, 75, 0.2); 
            color: #FF4B4B; 
            border-color: #FF4B4B; 
        }}
        .btn.paused:hover, .btn.active:hover {{
            background: #FF4B4B;
            color: white;
            box-shadow: 0 0 15px rgba(255, 75, 75, 0.5);
        }}
        .btn.secondary {{ 
            background: rgba(255, 255, 255, 0.05); 
            color: #ccc; 
            border-color: #444; 
        }}
        .btn.secondary:hover {{
            background: #444;
            color: white;
        }}

        #settings-modal {{
            position: absolute; top: 65px; left: 20px; z-index: 2000;
            background: rgba(30, 30, 30, 0.95); padding: 15px; border-radius: 8px;
            border: 1px solid #444; color: #eee; font-family: sans-serif; font-size: 12px;
            display: none; width: 200px; box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        }}
        .setting-group {{ margin-bottom: 12px; }}
        .setting-group label {{ display: block; margin-bottom: 5px; color: #aaa; }}
        .setting-group input[type="number"] {{ 
            width: 100%; background: #000; border: 1px solid #666; color: #fff; 
            padding: 4px; border-radius: 3px; 
        }}
        .setting-row {{ 
            display: flex; align-items: center; gap: 8px; margin-bottom: 8px; 
            font-size: 11px; color: #ccc;
        }}
        .setting-row input {{ cursor: pointer; }}
        #performance-info {{ 
            font-size: 9px; color: #888; background: #1a1a1a; 
            padding: 5px; border-radius: 4px; margin-top: 5px;
            border: 1px solid #333;
        }}
    </style>
</head>
<body>
    <div id="loader">Configurando red desde GitHub...</div>
    <div id="controls">
        <button id="layout-btn" class="btn">⏸ Pausar Red</button>
        <button id="noverlap-btn" class="btn secondary" title="Separa nodos encimados sin romper la estructura.">⚓ Anticolisión</button>
        <button id="louvain-btn" class="btn secondary" title="Agrupa matemáticamente por conexiones (colores).">🧬 Detectar Grupos</button>
        <button id="settings-btn" class="btn secondary">⚙️ Ajustes</button>
    </div>

    <div id="settings-modal" style="width: 250px; max-height: 80vh; overflow-y: auto;">
        <div style="font-weight: bold; margin-bottom: 5px; border-bottom: 1px solid #444; padding-bottom: 5px;">Ajustes ForceAtlas2</div>
        <div id="dynamic-status" style="font-size: 10px; color: #6FCF97; margin-bottom: 10px; font-style: italic;">
            ✨ Escalado Dinámico por Densidad: Activo
        </div>
        <div class="setting-group">
            <label title="Cuánta fuerza de separación deseas. Valores altos (ej. 100,000) dispersan mucho la red.">Fuerza de Repulsión (Escalado)</label>
            <input type="number" id="set-scaling" step="1000" value="120000">
        </div>
        <div class="setting-group">
            <label title="Atrae los nodos al centro. Evita que las islas se alejen flotando.">Gravedad (Atracción Central)</label>
            <input type="number" id="set-gravity" step="0.1" value="2.0">
        </div>
        <div class="setting-group">
            <label title="Umbral para la optimización Barnes-Hut (0 a 1).">Barnes Hut Theta</label>
            <input type="number" id="set-theta" step="0.1" value="0.5" min="0" max="1">
        </div>
        <div class="setting-group">
            <label title="Cuánto 'balanceo' permites. Mayor a 1 es desalentado. Menor da más precisión pero menos velocidad.">Jitter Tolerance</label>
            <input type="number" id="set-jitter" step="0.1" value="1.0">
        </div>
        <div class="setting-group">
            <label title="Influencia del peso de las aristas. 0 es sin influencia, 1 es normal.">Influencia Peso Aristas</label>
            <input type="number" id="set-edgeWeight" step="0.1" value="0.0">
        </div>
        <div class="setting-row">
            <input type="checkbox" id="set-adjustSizes" checked> <label for="set-adjustSizes" title="Toma en cuenta el tamaño de los nodos para evitar solapamientos.">Ajustar Tamaños</label>
        </div>
        <div class="setting-row">
            <input type="checkbox" id="set-outboundAttraction"> <label for="set-outboundAttraction" title="Distribuye la atracción por las aristas salientes. Los hubs atraen menos y son empujados a los bordes.">Disuadir Hubs (Outbound)</label>
        </div>
        <div class="setting-row">
            <input type="checkbox" id="set-linLog"> <label for="set-linLog" title="Escalado logarítmico para las fuerzas. Produce clusters más compactos.">Modo LinLog</label>
        </div>
        <div class="setting-row">
            <input type="checkbox" id="set-barnesHut" checked> <label for="set-barnesHut" title="Optimización para reducir complejidad de N^2 a N*log(N). Recomendado para redes grandes.">Optimización Barnes-Hut</label>
        </div>
        <div class="setting-row">
            <input type="checkbox" id="set-strongGravity"> <label for="set-strongGravity" title="Una vista con gravedad mucho más fuerte hacia el centro.">Modo Gravedad Fuerte</label>
        </div>
        <div id="performance-info">
            🚀 <b>Modo Turbo:</b> Estás usando una repulsión masiva. Se recomienda mantener <b>Barnes-Hut Activo</b> para evitar que el navegador se bloquee con 4,700 nodos.
        </div>
        <button id="apply-settings" class="btn" style="width:100%; margin-top: 10px;">Aplicar Parámetros</button>
    </div>

    <div id="container"></div>
    <div id="tooltip" class="tooltip"></div>
    <script>
        const data = {sigma_json};
        let pinnedNode = null;
        
        function loadScript(src) {{
            return new Promise((resolve, reject) => {{
                const s = document.createElement('script');
                s.src = src;
                s.onload = resolve;
                s.onerror = () => reject(new Error("Error cargando " + src));
                document.head.appendChild(s);
            }});
        }}

        async function init() {{
            const loader = document.getElementById('loader');
            const container = document.getElementById('container');
            const tooltip = document.getElementById('tooltip');
            const settingsModal = document.getElementById('settings-modal');
            const layoutBtn = document.getElementById('layout-btn');

            try {{
                await loadScript("https://data.abundis.com.mx/vista/includes/js-networks/graphology.js");
                await loadScript("https://data.abundis.com.mx/vista/includes/js-networks/graphology-library.js");
                await loadScript("https://data.abundis.com.mx/vista/includes/js-networks/sigma.js");
                
                const GraphLib = window.graphology || window.Graphology;
                const Library = window.graphologyLibrary;
                const graph = new (GraphLib.Graph || GraphLib)();
                
                if (!data.nodes || data.nodes.length === 0) {{
                    loader.innerHTML = "No hay datos para mostrar.";
                    return;
                }}

                data.nodes.forEach(node => {{
                    if (!graph.hasNode(node.id)) {{
                        const baseColor = node.color?.background || "#1C83E1";
                        graph.addNode(node.id, {{
                            label: node.label || node.id,
                            x: Math.random(), y: Math.random(),
                            size: node.size || 5,
                            color: baseColor,
                            originalColor: baseColor, // Store for toggle
                            category: node.type,
                            metadata: node.metadata || {{}}
                        }});
                    }}
                }});
                
                if (data.edges) {{
                    data.edges.forEach(edge => {{
                        if (graph.hasNode(edge.source) && graph.hasNode(edge.target)) {{
                            graph.addEdge(edge.source, edge.target, {{ size: 1, color: "#222" }});
                        }}
                    }});
                }}

                // --- Layout Logic ---
                let fa2Settings = {{}};
                let layoutInstance = null;
                let isPaused = false;
                let isManualLoop = false;

                if (Library && Library.layoutForceAtlas2) {{
                    const FA2 = Library.layoutForceAtlas2;
                    const nodeCount = graph.order;
                    const isTurbo = nodeCount > 500;
                    
                    const initialGravity = isTurbo ? 2 : 0.01;
                    const initialScaling = isTurbo ? 120000 : 20000;
                    const initialTheta = isTurbo ? 0.8 : 0.5;

                    fa2Settings = {{
                        gravity: initialGravity,
                        scalingRatio: initialScaling,
                        adjustSizes: true,
                        outboundAttractionDistribution: true,
                        linLogMode: false,
                        barnesHutOptimize: false,
                        barnesHutTheta: initialTheta,
                        jitterTolerance: 1.0,
                        edgeWeightInfluence: 0.0,
                        strongGravityMode: false
                    }};

                    // Sync UI
                    document.getElementById('set-gravity').value = initialGravity;
                    document.getElementById('set-scaling').value = initialScaling;
                    document.getElementById('set-theta').value = initialTheta;
                    document.getElementById('set-jitter').value = 1.0;
                    document.getElementById('set-edgeWeight').value = 0.0;
                    document.getElementById('set-outboundAttraction').checked = true;
                    document.getElementById('set-linLog').checked = false;
                    document.getElementById('set-adjustSizes').checked = true;
                    document.getElementById('set-barnesHut').checked = true; // Enabled by default for performance
                    
                    const statusEl = document.getElementById('dynamic-status');
                    if (isTurbo) {{
                        statusEl.innerHTML = "🚀 Escalado Turbo: Activo (" + nodeCount + " nodos)";
                        statusEl.style.color = "#6FCF97";
                    }} else {{
                        statusEl.innerHTML = "✨ Escalado Dinámico: Modo Precisión";
                        statusEl.style.color = "#1C83E1";
                    }}

                    function startLayout() {{
                        try {{
                            const Constructor = FA2.FA2Layout || (Library && Library.FA2Layout);
                            if (Constructor) {{
                                layoutInstance = new Constructor(graph, {{ settings: fa2Settings }});
                                layoutInstance.start();
                            }} else {{
                                isManualLoop = true;
                                runStep();
                            }}
                        }} catch(e) {{
                            isManualLoop = true;
                            runStep();
                        }}
                    }}

                    function runStep() {{
                        if (isPaused || !isManualLoop) return;
                        const iters = graph.order > 1000 ? 1 : 2;
                        FA2.assign(graph, {{ iterations: iters, settings: fa2Settings }});
                        requestAnimationFrame(runStep);
                    }}

                    window.toggleLayout = () => {{
                        isPaused = !isPaused;
                        if (layoutInstance) {{
                            if (layoutInstance.isRunning()) layoutInstance.stop();
                            else layoutInstance.start();
                        }}
                        layoutBtn.innerHTML = isPaused ? "▶️ Reanudar Red" : "⏸ Pausar Red";
                        layoutBtn.classList.toggle('paused', isPaused);
                    }};

                    layoutBtn.addEventListener('click', window.toggleLayout);
                    if (graph.order > 0) startLayout();
                }}

                // --- NEW: Noverlap & Louvain ---
                document.getElementById('noverlap-btn').addEventListener('click', () => {{
                    if (Library && Library.layoutNoverlap) {{
                        const statusEl = document.getElementById('dynamic-status');
                        statusEl.innerHTML = "⚓ Ejecutando Anticolisión...";
                        Library.layoutNoverlap.assign(graph, {{
                            maxIterations: 50,
                            settings: {{ ratio: 1.2, margin: 2 }}
                        }});
                        statusEl.innerHTML = "✅ Espaciado Optimizado";
                    }}
                }});

                let isLouvainActive = false;
                document.getElementById('louvain-btn').addEventListener('click', () => {{
                    if (Library && Library.communitiesLouvain) {{
                        isLouvainActive = !isLouvainActive;
                        const btn = document.getElementById('louvain-btn');
                        
                        if (isLouvainActive) {{
                            const communities = Library.communitiesLouvain(graph);
                            const colors = ["#82ca9d", "#8884d8", "#ffc658", "#ff8042", "#0088fe", "#00c49f", "#ffbb28", "#ff8042", "#a4de6c", "#d0ed57"];
                            graph.forEachNode((node, attr) => {{
                                const commId = communities[node];
                                graph.setNodeAttribute(node, "color", colors[commId % colors.length]);
                            }});
                            document.getElementById('dynamic-status').innerHTML = "🧬 Clusters Detectados (Louvain)";
                            btn.innerHTML = "🔙 Restaurar Colores";
                            btn.classList.add('active');
                        }} else {{
                            graph.forEachNode((node, attr) => {{
                                graph.setNodeAttribute(node, "color", attr.originalColor);
                            }});
                            document.getElementById('dynamic-status').innerHTML = "✨ Colores Originales";
                            btn.innerHTML = "🧬 Detectar Grupos";
                            btn.classList.remove('active');
                        }}
                    }}
                }});

                // --- Settings Modal ---
                document.getElementById('settings-btn').addEventListener('click', () => {{
                    settingsModal.style.display = (settingsModal.style.display === 'block') ? 'none' : 'block';
                }});

                document.getElementById('apply-settings').addEventListener('click', () => {{
                    fa2Settings = {{
                        gravity: parseFloat(document.getElementById('set-gravity').value),
                        scalingRatio: parseFloat(document.getElementById('set-scaling').value),
                        barnesHutTheta: parseFloat(document.getElementById('set-theta').value),
                        jitterTolerance: parseFloat(document.getElementById('set-jitter').value),
                        edgeWeightInfluence: parseFloat(document.getElementById('set-edgeWeight').value),
                        adjustSizes: document.getElementById('set-adjustSizes').checked,
                        outboundAttractionDistribution: document.getElementById('set-outboundAttraction').checked,
                        linLogMode: document.getElementById('set-linLog').checked,
                        barnesHutOptimize: document.getElementById('set-barnesHut').checked,
                        strongGravityMode: document.getElementById('set-strongGravity').checked
                    }};
                    
                    if (layoutInstance) layoutInstance.stop();
                    isManualLoop = false;
                    isPaused = false;
                    layoutBtn.innerHTML = "⏸ Pausar Red";
                    layoutBtn.classList.remove('paused');
                    
                    setTimeout(() => {{
                        const FA2 = Library.layoutForceAtlas2;
                        const Constructor = FA2.FA2Layout || (Library && Library.FA2Layout);
                        if (Constructor) {{
                            layoutInstance = new Constructor(graph, {{ settings: fa2Settings }});
                            layoutInstance.start();
                        }} else {{
                            isManualLoop = true;
                            runStep();
                        }}
                    }}, 50);
                    settingsModal.style.display = 'none';
                }});

                // --- Renderer ---
                const renderer = new Sigma(graph, container, {{
                    renderEdgeLabels: false,
                    labelFont: "Inter, sans-serif",
                    labelSize: 11,
                    labelWeight: "600",
                    labelColor: {{ color: "#ccc" }},
                    defaultNodeType: "circle",
                    defaultEdgeType: "arrow",
                    labelDensity: 0.07,
                    labelGridCellSize: 60,
                    labelRenderedSizeThreshold: 8
                }});

                function updateTooltip(nodeId) {{
                    const attr = graph.getNodeAttributes(nodeId);
                    const meta = attr.metadata || {{}};
                    const desc = meta.raw_text || meta.description || "";
                    
                    let html = `<span class="close-btn" onclick="window.closePinned()">✖</span>`;
                    html += `<strong>${{attr.label}}</strong><br/><small style='color:#1C83E1'>${{attr.category.toUpperCase()}}</small>`;
                    if (meta.municipio) html += `<br/><i style='font-size:11px; color:#888'>📍 ${{meta.municipio}}</i>`;
                    if (desc) {{
                        html += `<hr style='border:0;border-top:1px solid #444;margin:8px 0'/>
                                <div style='max-height:180px; overflow-y:auto; line-height:1.4; font-size:11px; color:#bbb'>
                                    ${{desc}}
                                </div>`;
                    }}
                    tooltip.innerHTML = html;
                    tooltip.style.display = 'block';
                }}

                window.closePinned = () => {{ pinnedNode = null; tooltip.style.display = 'none'; }};
                
                renderer.on("enterNode", ({{ node }}) => {{ 
                    if (!pinnedNode) {{ updateTooltip(node); tooltip.style.pointerEvents = 'none'; }}
                }});
                renderer.on("leaveNode", () => {{ 
                    if (!pinnedNode) tooltip.style.display = 'none'; 
                }});
                renderer.on("clickNode", ({{ node }}) => {{
                    pinnedNode = node; updateTooltip(node); tooltip.style.pointerEvents = 'auto';
                }});
                renderer.on("clickStage", () => window.closePinned());
                
                window.addEventListener('mousemove', e => {{
                    if (!pinnedNode) {{
                        tooltip.style.left = (e.clientX + 15) + 'px';
                        tooltip.style.top = (e.clientY + 15) + 'px';
                    }}
                }});

                loader.style.display = 'none';

            }} catch (err) {{
                console.error("Sigma Init Error:", err);
                loader.innerHTML = `<div style="color:#ff4b4b">Error: ${{err.message}}</div>`;
            }}
        }}

        init();
    </script>
</body>
</html>
"""

                    components.html(sigma_html, height=850, scrolling=True)
                except Exception as e:
                    st.error(f"Error renderizando Sigma.js: {e}")
            else:
                st.info(f"Archivo de grafo no encontrado: {gml_file}")

    elif menu == "🐍 Visualización Python":
        st.subheader("Redes de Relación Interactivas – PyVis (Legacy)")
        
        g_col1, g_col2 = st.columns([1, 3])
        
        with g_col1:
            graph_type = st.radio("Tipo de Grafo (Python)", ["Red de Relaciones (Bipartita)", "Afinidad Narrativa (Caso a Caso)"])
            gml_file = "relationship_network.gml" if "Relaciones" in graph_type else "probable_graph.gml"
            gml_path = os.path.join(RESULTS_DIR, gml_file)

            if os.path.exists(gml_path):
                G_raw = load_graph_cached(gml_path)
                
                st.write("---")
                st.write("**Filtros**")
                
                available_types = ['case', 'scenario', 'keyword', 'family']
                
                with st.form("graph_filters_form_python"):
                    sel_types = st.multiselect("Tipos de Nodo Visibles", available_types, default=[t for t in available_types if t in ['case', 'scenario']])
                    muns = sorted(list(set([d.get('municipio', 'N/A') for n, d in G_raw.nodes(data=True) if d.get('municipio')])))
                    sel_g_muns = st.multiselect("Filtrar por Municipio (Grafo)", ["Todos"] + muns, default=["Todos"])
                    sel_g_conf = st.multiselect("Confianza Escenarios", ["alta", "media", "baja"], default=["alta"])
                    search_node = st.text_input("Buscar Nodo (ID/Etiqueta)")
                    apply_btn = st.form_submit_button("Aplicar Cambios")

                if apply_btn or 'graph_initialized_py' not in st.session_state:
                    st.session_state['graph_initialized_py'] = True

                # Apply Filters
                G = G_raw.copy()
                edges_to_remove = []
                for u, v, d in G.edges(data=True):
                    conf = d.get('confidence')
                    if conf and (not sel_g_conf or conf not in sel_g_conf):
                        edges_to_remove.append((u, v))
                G.remove_edges_from(edges_to_remove)

                nodes_to_remove = []
                for n, d in G.nodes(data=True):
                    type_ok = d.get('type', 'case') in sel_types
                    mun_ok = "Todos" in sel_g_muns or d.get('municipio') in sel_g_muns
                    search_ok = not search_node or search_node.lower() in str(n).lower() or search_node.lower() in str(d.get('label', '')).lower()
                    if not (type_ok and mun_ok and search_ok):
                        nodes_to_remove.append(n)
                G.remove_nodes_from(nodes_to_remove)
                
                if st.checkbox("Ocultar Nodos Aislados", value=True, key="py_isolates"):
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
                    st.error(f"Error renderizando el grafo con PyVis: {e}")
            else:
                st.info(f"Archivo de grafo no encontrado: {gml_file}")

    elif menu == "🧬 Análisis de Clusters":
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
