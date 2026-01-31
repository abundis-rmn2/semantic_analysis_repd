import pytest
import os
import pandas as pd
from schemas import EventAnalysis, ScenarioHypothesis
from reusable.text_utils import normalize_text, extract_temporal_info

def test_normalization():
    assert normalize_text("Hola, ¿cómo estás?") == "hola como estas"
    assert normalize_text("   Muchos   espacios   ") == "muchos espacios"

def test_regex_extraction():
    text = "El 12/05/2023 en una camioneta blanca en la calle Juárez"
    slots = extract_temporal_info(text)
    assert "12/05/2023" in slots['date']
    assert any("camioneta" in t for t in slots['transport'])
    assert any("calle" in l for l in slots['location_hint'])

def test_schema_validation():
    data = {
        "id_original": "123",
        "raw_text": "text",
        "text_norm": "text",
        "municipio": "Guadalajara",
        "scenarios": [
            {
                "scenario_label": "test",
                "scenario_confidence": "ALTA",
                "supporting_signals": ["sig1"]
            }
        ],
        "provenance_id": "hash123"
    }
    analysis = EventAnalysis(**data)
    assert analysis.scenarios[0].scenario_confidence == "alta"

if __name__ == "__main__":
    # Minimal smoke test
    print("Running minimal smoke tests...")
    test_normalization()
    test_regex_extraction()
    test_schema_validation()
    print("Tests passed!")
