# Prompt templates for the exploratory pipeline

EXPLORATORY_SCENARIO_PROMPT = """Analiza el siguiente relato de desaparición y genera hipótesis plausibles utilizando ÚNICAMENTE las siguientes MACROCLASES:
- coerción_armada (desaparición forzada, presencia de armas/vehículos)
- ausencia_voluntaria (partida por voluntad propia, sin violencia)
- laboral_forzada (engaño laboral, reclutamiento)
- migración (traslado fronterizo o intención de emigrar)
- internamiento_institucional (detención por autoridad, hospital, anexo)
- conflicto_familiar (discusión previa, problemas domésticos)
- accidente (vial, laboral, natural)
- desconocido (sin información suficiente)

Instrucciones:
1. Identifica hasta 4 escenarios posibles de la lista anterior.
2. Para cada escenario, asigna una confianza (alta, media, baja).
3. Proporciona citas textuales exactas que respalden cada escenario.
4. No infieras identidades ni afirmes causalidad definitiva.
5. Responde estrictamente en formato JSON.

Ejemplos:

Relato: "Se lo llevaron varios sujetos armados que llegaron en una camioneta blanca, lo golpearon y lo subieron a la fuerza."
Respuesta: {{
    "scenarios": [
        {{
            "scenario_label": "coerción_armada",
            "scenario_confidence": "alta",
            "supporting_signals": ["uso de fuerza", "vehículo sospechoso", "sujetos armados"],
            "text_cites": ["sujetos armados", "lo golpearon", "subieron a la fuerza"],
            "notes": "Escenario típico de desaparición forzada por grupo criminal."
        }}
    ]
}}

Relato: "Salió de su casa tras una discusión con sus padres, diciendo que ya no quería estar ahí. Se llevó una mochila con ropa."
Respuesta: {{
    "scenarios": [
        {{
            "scenario_label": "ausencia_voluntaria",
            "scenario_confidence": "alta",
            "supporting_signals": ["discusión previa", "preparación de equipaje", "intención declarada"],
            "text_cites": ["ya no quería estar ahí", "se llevó una mochila con ropa"],
            "notes": "Los indicadores sugieren una partida por voluntad propia tras conflicto familiar."
        }}
    ]
}}

Relato: "Le ofrecieron un trabajo muy bien pagado en otra ciudad por Facebook. Se fue con un contacto que no conocíamos y ya no responde el celular."
Respuesta: {{
    "scenarios": [
        {{
            "scenario_label": "laboral_forzada_o_tata",
            "scenario_confidence": "media",
            "supporting_signals": ["oferta laboral sospechosa", "redes sociales", "traslado con desconocidos"],
            "text_cites": ["trabajo muy bien pagado", "contacto que no conocíamos"],
            "notes": "Posible enganche mediante engaño para explotación o reclutamiento."
        }},
        {{
            "scenario_label": "migración_irregular",
            "scenario_confidence": "baja",
            "supporting_signals": ["traslado a otra ciudad"],
            "text_cites": ["se fue con un contacto"],
            "notes": "Menos probable pero el traslado podría ser parte de un intento migratorio."
        }}
    ]
}}

Nuevo Relato a analizar:
"{text}"

Esquema JSON esperado:
{{
    "scenarios": [
        {{
            "scenario_label": "string",
            "scenario_confidence": "alta/media/baja",
            "supporting_signals": ["string"],
            "text_cites": ["string"],
            "notes": "string"
        }}
    ]
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
