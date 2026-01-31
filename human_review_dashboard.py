import streamlit as st
import pandas as pd
import json
import os

st.set_page_config(page_title="Probable: Human Triage", layout="wide")

st.title("🔍 Probable: Triage de Desapariciones")
st.markdown("Revisión de hipótesis generadas por el pipeline exploratorio.")

# Sidebar for config
results_dir = st.sidebar.text_input("Directorio de Resultados", "results")
parquet_file = os.path.join(results_dir, "probable_analysis.parquet")

if not os.path.exists(parquet_file):
    st.error(f"No se encontró el archivo: {parquet_file}")
else:
    df = pd.read_parquet(parquet_file)
    
    # Filter controls
    st.sidebar.subheader("Filtros")
    min_ambiguity = st.sidebar.slider("Ambigüedad mínima", 0.0, 1.0, 0.0)
    selected_scenario = st.sidebar.selectbox("Escenario (Alta Confianza)", ["Todos"] + sorted(list(set([s['scenario_label'] for sc_list in df['scenarios'] for s in sc_list if s['scenario_confidence'] == 'alta']))))

    # Apply filters
    filtered_df = df[df['ambiguity_score'] >= min_ambiguity]
    if selected_scenario != "Todos":
        filtered_df = filtered_df[filtered_df['scenarios'].apply(lambda x: any(s['scenario_label'] == selected_scenario and s['scenario_confidence'] == 'alta' for s in x))]

    st.write(f"Mostrando {len(filtered_df)} de {len(df)} registros.")

    # Table view
    for idx, row in filtered_df.iterrows():
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
