# storage.py
# Módulo para almacenamiento y exportación de resultados

import os
import pandas as pd
import json

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

def save_to_csv(df: pd.DataFrame, path: str) -> None:
    """
    Guarda un DataFrame en CSV creando el directorio si no existe.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding='utf-8')

def save_to_json(df: pd.DataFrame, path: str) -> None:
    """
    Guarda un DataFrame en JSON (orientación 'records'), creando el directorio.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    records = df.to_dict(orient='records')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

def build_faiss_index(embeddings, index_path: str, metric: str = 'cosine'):
    """
    Construye y guarda un índice FAISS a partir de un array de embeddings.

    :param embeddings: numpy array de forma (n_samples, dim)
    :param index_path: ruta donde guardar el archivo del índice
    :param metric: 'l2' o 'cosine'
    """
    if not FAISS_AVAILABLE:
        raise ImportError("faiss no está instalado. Instala faiss-cpu o faiss-gpu para usar esta función.")

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    dim = embeddings.shape[1]
    if metric == 'l2':
        idx = faiss.IndexFlatL2(dim)
    else:
        # para cosine, normalizamos y usamos IndexFlatIP
        faiss.normalize_L2(embeddings)
        idx = faiss.IndexFlatIP(dim)

    idx.add(embeddings)
    faiss.write_index(idx, index_path)

    return idx

# Ejemplo de uso:
if __name__ == '__main__':
    # Simula un DataFrame
    df = pd.DataFrame([
        {'fragment': 'Texto 1', 'topic': 0},
        {'fragment': 'Texto 2', 'topic': 1},
    ])
    save_to_csv(df, 'data/processed/results.csv')
    save_to_json(df, 'data/processed/results.json')

    # Simula embeddings
    import numpy as np
    emb = np.random.random((2, 768)).astype('float32')
    if FAISS_AVAILABLE:
        build_faiss_index(emb, 'data/processed/fragments.index')
    else:
        print('FAISS no disponible, se omitió índice.')