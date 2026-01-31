from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime

class Evidence(BaseModel):
    source_span: str = Field(..., description="Exact text from the source")
    offset_start: Optional[int] = None
    offset_end: Optional[int] = None
    extraction_method: str = Field(..., description="Method used: regex, keyword, or llm")

class ScenarioHypothesis(BaseModel):
    scenario_label: str = Field(..., description="Open string label for the scenario (e.g., coercion_armada)")
    scenario_confidence: str = Field(..., description="alta, media, or baja")
    supporting_signals: List[str] = Field(default_factory=list, description="List of signals supporting this scenario")
    text_cites: List[str] = Field(default_factory=list, description="Exact phrases supporting this scenario")
    notes: Optional[str] = None

    @validator('scenario_confidence')
    def validate_confidence(cls, v):
        if v.lower() not in ['alta', 'media', 'baja']:
            return 'baja'
        return v.lower()

class EventAnalysis(BaseModel):
    id_original: str
    raw_text: str
    text_norm: str
    date_dt: Optional[datetime] = None
    municipio: str
    observables: Dict[str, Any] = Field(
        default_factory=lambda: {
            "temporal_slots": [],
            "location_slots": [],
            "transport_slots": [],
            "actor_slots": [],
            "keyword_hits": []
        }
    )
    scenarios: List[ScenarioHypothesis] = Field(default_factory=list)
    ambiguity_score: float = Field(0.0, ge=0.0, le=1.0)
    evidence_snippets: List[str] = Field(default_factory=list)
    llm_meta: Dict[str, Any] = Field(default_factory=dict)
    provenance_id: str
