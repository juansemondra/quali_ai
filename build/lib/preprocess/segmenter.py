# preprocess/segmenter.py
from typing import List
from preprocess.anonymizer import get_nlp

def sentence_segment(text: str) -> List[str]:
    """
    Retorna lista de oraciones usando el componente de sentenciamiento de spaCy.
    """
    nlp = get_nlp()
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def paragraph_segment(text: str) -> List[str]:
    """
    Divide el texto en párrafos usando dobles saltos de línea.
    """
    paras = [p.strip() for p in text.split('\n\n') if p.strip()]
    return paras