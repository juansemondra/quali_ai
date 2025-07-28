# main.py
import os
import pandas as pd

from preprocess.loader import load_raw
from preprocess.cleaner import clean_text
from preprocess.anonymizer import anonymize_text
from preprocess.segmenter import sentence_segment

DATA_DIR = "data/raw"
OUT_DIR  = "data/processed"

def process_all():
    results = []
    for fname in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, fname)
        try:
            df = load_raw(path)
        except ValueError:
            continue  # salta archivos no soportados
        # Limpieza y anonimización
        df['cleaned'] = df['text'].apply(clean_text)
        df['anonymized'] = df['cleaned'].apply(anonymize_text)

        # Segmentación en oraciones
        for _, row in df.iterrows():
            for sent in sentence_segment(row['anonymized']):
                results.append({
                    "file": fname,
                    "sentence": sent
                })
    return pd.DataFrame(results)

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    df_out = process_all()
    out_path = os.path.join(OUT_DIR, "sentences.csv")
    df_out.to_csv(out_path, index=False, encoding='utf-8')
    print(f"Procesadas {len(df_out)} oraciones. Salida en {out_path}")