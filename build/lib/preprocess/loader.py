# preprocess/loader.py
import os
import pandas as pd
from PyPDF2 import PdfReader

def load_raw(path: str):
    """
    Lee archivos .csv (debe tener columna 'text'), .txt o .pdf y devuelve
    un DataFrame con la columna 'text'.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
        if 'text' not in df.columns:
            raise ValueError(f"CSV {path} no contiene columna 'text'")
        return df[['text']]
    elif ext == ".txt":
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        return pd.DataFrame({'text': [text]})
    elif ext == ".pdf":
        reader = PdfReader(path)
        text_pages = []
        for page in reader.pages:
            text_pages.append(page.extract_text() or "")
        full_text = "\n\n".join(text_pages)
        return pd.DataFrame({'text': [full_text]})
    else:
        raise ValueError(f"Formato no soportado: {ext}")