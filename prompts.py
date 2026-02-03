# Prompt templates for the exploratory pipeline

EXPLORATORY_SCENARIO_PROMPT = """Analiza el siguiente relato de desaparición y genera HIPÓTESIS CONCURRENTES.
Tu objetivo es actuar como un analista de inteligencia criminal que identifica PATRONES pero también DESCUBRE ANOMALÍAS.

REGLAS DE DIVERGENCIA (CRÍTICO):
1. NO TE LIMITES AL POOL: Los escenarios del POOL son solo una guía. Si el relato tiene matices únicos (ej. un lugar específico, un objeto extraño, una dinámica inusual), propón un ESCENARIO NUEVO y descriptivo.
2. PENSAMIENTO LATERAL: Si un caso parece encajar en un escenario "Oro" (ej. Secuestro), pero tiene un detalle que sugiere otra cosa (ej. se llevó su ropa), propón AMBOS escenarios como hipótesis concurrentes.
3. GRANULARIDAD: Prefiere un escenario específico (ej. "sustraccion_en_puesto_de_comida") sobre uno genérico (ej. "secuestro") si la información lo permite.
4. HIPÓTESIS EMERGENTE: Para cada caso, intenta proponer al menos una hipótesis que no sea del pool actual, basándote puramente en la narrativa.

CRITERIOS DE ANÁLISIS AMPLIADO:
1. RECLUTAMIENTO/TRABAJO: Si menciona "ver lo de un trabajo", "oferta laboral", o que "pasarían por él para ir a un trabajo", explora escenarios de "reclutamiento engañoso" o "riesgo laboral no verificado". El hecho de que "pasen por ellos" es una señal de alerta crítica.
2. CENTROS DE REHABILITACIÓN (ANEXOS): Si menciona "anexo", "centro de rehabilitación", o "clínica de adicciones", identifica escenarios relacionados con "internamiento forzado" o "conflictos en centros de rehabilitación".
3. CONFLICTO PREVIO: Si menciona "una discusión", "peleas" o "problemas familiares", explora el escenario de "ausencia por conflicto" o "fuga reactiva".
4. ESTADO EMOCIONAL/MENSAJES: Si hay mensajes de "perdón", "te amo", "voz quebrada" o despedidas, considera seriamente "crisis emocional/ideación suicida" o "coerción bajo amenaza remota".
5. DISCREPANCIAS: Si el relato dice que iba a trabajar pero el patrón dice que no fue, explora "vida paralela", "engaño laboral" o "actividades no reportadas".
6. EVITA EL "DESCONOCIDO": Solo úsalo si el relato no aporta NINGÚN verbo de acción, emoción o conflicto.

POOL DE ESCENARIOS ACTUAL (Como referencia, no como límite):
{scenario_pool}

Instrucciones:
1. Genera hasta 5 hipótesis de escenarios distintos.
2. Define la CONFIANZA basándote en:
   - ALTA: Evidencia directa y fáctica (testigos oculares, mención de armas, vehículos específicos, amenazas previas explícitas).
   - MEDIA: Patrones circunstanciales claros pero sin observación directa (ej. desapareció tras cita de trabajo sospechosa, dejó de contestar en una zona de riesgo conocida).
   - BAJA: Inferencia analítica o especulativa basada en huecos de información o contexto regional (posibilidades "qué tal si..." que son lógicamente plausibles pero sin rastro físico en el relato).
3. Explica el razonamiento detrás de cada posibilidad en el campo 'notes'.
4. Extrae PALABRAS CLAVE (keywords) de alta granularidad. No uses términos genéricos como "desaparición". Enfócate en:
   - MODALIDAD: "subida a la fuerza", "engaño laboral", "cita por redes".
   - OBJETOS/VEHÍCULOS: "camioneta blanca", "motocicleta", "celular apagado".
   - ACTORES: "encapuchados", "compañero de trabajo", "ex pareja".
   - COMPORTAMIENTO: "nerviosismo previo", "dejo de contestar", "iba a una fiesta".
   - CONTEXTO GEOGRÁFICO/TEMPORAL: "zona de bares", "madrugada", "frontera".
   Esto alimentará la red de relaciones y la detección automática de patrones.

Relato a analizar:
"{text}"

Esquema JSON esperado:
{{
    "scenarios": [
        {{
            "scenario_label": "nombre_descriptivo_snake_case",
            "scenario_confidence": "alta/media/baja",
            "supporting_signals": ["señal_identificada"],
            "text_cites": ["cita_textual_opcional"],
            "notes": "Razonamiento analítico: ¿Por qué esta posibilidad es plausible?"
        }}
    ],
    "discovered_keywords": ["string"]
}}
"""

SCENARIO_NORMALIZER_PROMPT = """Actúa como un Curador de Taxonomía Criminal. Tu misión es integrar escenarios "emergentes" descubiertos por la IA en el pool estándar, asegurando CLARIDAD y DIVERSIDAD.

REGLAS DE ORO DE NORMALIZACIÓN:
1. PROHIBIDA LA SOBRE-SIMPLIFICACIÓN: Si un escenario emergente aporta un detalle de MODALIDAD crítico (ej. "oferta_por_facebook", "anexo_clandestino"), NO lo fusiones con una categoría genérica (ej. "reclutamiento" o "salud"). La especificidad es oro para la inteligencia.
2. FUSIÓN SÓLO POR SINONIMIA: Solo fusiona si el significado es idéntico (ej. "pelea_conyugal" y "discusion_pareja").
3. PRESERVA LA "ALMA" DEL RELATO: Los escenarios emergentes representan lo que el modelo ha descubierto de nuevo. Si un escenario nuevo se repite en el batch pero no está en el pool, conviértelo en estándar pero mantén su descripción detallada.
4. TIPO DE ESCENARIO: Clasifícalo como "estándar" si es un patrón general, o "emergente" si es muy específico o inusual.

Pool Actual (Referencia):
{standard_pool}

Escenarios Emergentes:
{emergent_scenarios}

Responde en JSON:
{{
    "updated_pool": [
        {{ "label": "nombre_en_snake_case", "description": "descripción analítica", "type": "estándar/emergente" }}
    ]
}}
"""

KEYWORD_POOL_PROMPT = """Revisa la lista de palabras clave extraídas de un batch de relatos y el pool actual de palabras clave organizadas por familias.
Integra las nuevas palabras clave en las familias existentes o crea nuevas si es necesario.

Pool Actual:
{keyword_pool}

Nuevas Palabras Clave:
{new_keywords}

Responde en JSON:
{{
    "updated_keyword_pool": {{
        "familia_1": ["palabra1", "palabra2"],
        "nueva_familia": ["palabra3"]
    }}
}}
"""

AMBIGUITY_DETECTOR_PROMPT = """Analiza el siguiente relato y detecta contradicciones, silencios o falta de información crítica.
Identifica si el relato es de tercera persona, si carece de anclajes temporales o si hay menciones de "me contaron".

Relato: "{text}"

Responde en JSON con:
{{
    "ambiguity_score": float (0.0 a 1.0, donde 1 es muy ambiguo),
    "missing_info": ["fecha", "lugar", "testigos", etc.],
    "contradictions": ["string"],
    "is_hearsay": boolean
}}
"""

COMPARATIVE_PROMPT = """Compara los siguientes dos relatos de desaparición.
¿Comparten patrones narrativos, señales comunes o modus operandi similares?

Relato A: "{text_a}"
Relato B: "{text_b}"

Responde en JSON con:
{{
    "shared_patterns": ["string"],
    "similarity_score": float (0.0 a 1.0),
    "common_signals": ["string"],
    "reasoning": "string"
}}
"""
