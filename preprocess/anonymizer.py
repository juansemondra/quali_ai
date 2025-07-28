# preprocess/anonymizer.py
import importlib
import spacy
from spacy.cli import download as spacy_download

_nlp = None

def get_nlp():
    """
    Carga el modelo 'en_core_web_sm' importándolo directamente,
    descargándolo si no está instalado.
    """
    global _nlp
    if _nlp is None:
        try:
            # Primero, intenta importar el paquete de modelo
            model_pkg = importlib.import_module("en_core_web_sm")
            _nlp = model_pkg.load()
        except (ImportError, OSError):
            # Si no está instalado, descárgalo y vuelve a importar
            spacy_download("en_core_web_sm")
            model_pkg = importlib.import_module("en_core_web_sm")
            _nlp = model_pkg.load()
    return _nlp

def anonymize_text(text: str) -> str:
    """
    Reemplaza entidades PERSON, ORG y GPE por tokens genéricos.
    """
    nlp = get_nlp()
    doc = nlp(text)
    anonymized = text
    for ent in reversed(doc.ents):
        if ent.label_ in ("PERSON", "ORG", "GPE"):
            anonymized = (
                anonymized[:ent.start_char] +
                f"<{ent.label_.lower()}>" +
                anonymized[ent.end_char:]
            )
    return anonymized