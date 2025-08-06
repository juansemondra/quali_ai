# preprocess/cleaner.py
import re
import pandas as pd

def clean_text(text: str) -> str:
    """
    Aplica transformaciones básicas:
    - Quita HTML tags
    - Elimina URLs
    - Normaliza espacios y case-folding
    """
    # 1. Quitar tags HTML
    text = re.sub(r'<[^>]+>', ' ', text)
    # 2. Quitar URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # 3. Quitar emojis y símbolos no ASCII
    text = re.sub(r'[^\x00-\x7F]+',' ', text)
    # 4. Minusculas y limpieza de espacios
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def load_and_clean(path: str) -> pd.DataFrame:
    """
    Lee un CSV con columna 'text', aplica clean_text y retorna DataFrame.
    """
    df = pd.read_csv(path)
    df['cleaned'] = df['text'].apply(clean_text)
    return df