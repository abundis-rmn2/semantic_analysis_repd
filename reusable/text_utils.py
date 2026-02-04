import re
import unicodedata
from typing import Dict, Any, List

def normalize_text(text: str) -> str:
    """Basic normalization: lower, strip accents, collapse spaces."""
    if not text:
        return ""
    text = str(text).lower()
    # Remove accents
    text = "".join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    # Remove special chars but keep spaces and alphanumeric
    text = re.sub(r'[^\w\s]', ' ', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_temporal_info(text: str) -> Dict[str, Any]:
    """Regex for dates, times, and patterns (generalized from process_violence.py)."""
    patterns = {
        "date": r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        "time": r'(\d{1,2}:\d{2})\s*(?:hrs|am|pm)?',
        "transport": r'(camioneta|vehiculo|carro|auto|uber|taxi|moto|motocicleta|camion)\s+([a-zA-Z\s]+)?',
        "location_hint": r'(calle|colonia|avenida|cruce|esquina|andador|carretera)\s+([a-zA-Z\s]+)?'
    }
    extracted = {}
    for key, pattern in patterns.items():
        # Find all matches
        matches = re.findall(pattern, str(text), re.IGNORECASE)
        if matches:
            # For extraction, we might just want the first or a list
            extracted[key] = [" ".join(m) if isinstance(m, tuple) else m for m in matches]
        else:
            extracted[key] = []
    return extracted

def extract_regex_slots(text: str) -> Dict[str, List[str]]:
    """Extract common slots using regex."""
    # This can be expanded with more patterns from signal engineering
    return extract_temporal_info(text)
