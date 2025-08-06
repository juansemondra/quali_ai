# main.py

import os
import pandas as pd

from preprocess.loader import load_raw
from preprocess.cleaner import clean_text
from preprocess.anonymizer import anonymize_text
from preprocess.segmenter import sentence_segment

from orchestrator import LLMOrchestrator
from nlp_engine import NLPEngine
import storage

# Directorios de entrada y salida
DATA_DIR = "data/raw"
OUT_DIR  = "data/processed"


def process_all() -> pd.DataFrame:
    """
    Carga, limpia, anonimiza, segmenta y analiza con LLM (códigos & sentimiento).
    Retorna DataFrame con columnas: file, fragment, codes, sentiment.
    """
    orch = LLMOrchestrator(model_name="deepseek-r1:8b", use_ollama=True)
    records = []

    for fname in sorted(os.listdir(DATA_DIR)):
        path = os.path.join(DATA_DIR, fname)
        try:
            df = load_raw(path)
        except ValueError:
            continue  # salta formatos no soportados

        df["cleaned"]    = df["text"].apply(clean_text)
        df["anonymized"] = df["cleaned"].apply(anonymize_text)

        for _, row in df.iterrows():
            fragments = sentence_segment(row["anonymized"])
            for frag in fragments:
                analysis = orch.extract_codes_and_sentiment(frag)
                records.append({
                    "file":      fname,
                    "fragment":  frag,
                    "codes":     analysis.get("codes"),
                    "sentiment": analysis.get("sentiment"),
                })

    return pd.DataFrame(records)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Paso 1: fragmentos, códigos y sentimiento
    df_frag = process_all()

    if df_frag.empty:
        print("⚠️ No se encontraron fragmentos para procesar. Verifica tu carpeta data/raw.")
        return

    # Ahora sí, la columna 'fragment' existe
    fragments = df_frag["fragment"].tolist()

    # Inicializamos columnas por defecto para NLP
    df_frag["topic"]         = None
    df_frag["probability"]   = None
    df_frag["representation"] = None
    df_frag["cluster"]       = None

    # Sólo si hay al menos dos fragmentos, aplicamos motor NLP
    if len(fragments) >= 2:
        engine = NLPEngine(n_clusters=10)

        # Paso 2: tópicos y clusters
        topic_info   = engine.extract_topics(fragments)
        cluster_ids  = engine.cluster_embeddings(fragments)

        df_frag["topic"]         = [info["topic"]         for info in topic_info]
        df_frag["probability"]   = [info["probability"]   for info in topic_info]
        df_frag["representation"] = [info["representation"] for info in topic_info]
        df_frag["cluster"]       = cluster_ids

        # Paso opcional 3: construir índice FAISS de embeddings
        try:
            embeddings = engine.embedder.encode(fragments, convert_to_tensor=False)
            index_path = os.path.join(OUT_DIR, "fragments.index")
            storage.build_faiss_index(embeddings, index_path)
            print(f"✅ FAISS index creado en {index_path}")
        except Exception as e:
            print(f"⚠️ No se pudo crear FAISS index: {e}")
    else:
        print("⚠️ Sólo hay 1 fragmento: se omite topic modeling, clustering y FAISS index.")

    # Paso final: exportación
    csv_path  = os.path.join(OUT_DIR, "analysis.csv")
    json_path = os.path.join(OUT_DIR, "analysis.json")

    storage.save_to_csv(df_frag, csv_path)
    storage.save_to_json(df_frag, json_path)

    print(f"✅ Procesados {len(df_frag)} fragmentos.")
    print(f"   → CSV exportado en: {csv_path}")
    print(f"   → JSON exportado en: {json_path}")

if __name__ == "__main__":
    main()