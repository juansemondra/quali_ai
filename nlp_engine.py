# nlp_engine.py
# Motor de Análisis NLP: topic modeling, clustering de embeddings y análisis de sentimiento

from typing import List, Dict, Any
import pandas as pd

# Topic Modeling
from bertopic import BERTopic

# Embeddings + Clustering
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# Sentiment Analysis
from transformers import pipeline

class NLPEngine:
    def __init__(
        self,
        topic_model: BERTopic = None,
        embed_model_name: str = "all-mpnet-base-v2",
        n_clusters: int = 10,
        sentiment_model: str = "nlptown/bert-base-multilingual-uncased-sentiment"
    ):
        # Inicializar BERTopic
        self.topic_model = topic_model or BERTopic()
        # Inicializar modelo de embeddings
        self.embedder = SentenceTransformer(embed_model_name)
        self.n_clusters = n_clusters
        # Pipeline de sentimiento
        self.sentiment = pipeline("sentiment-analysis", model=sentiment_model)

    def extract_topics(self, documents: List[str]) -> List[Dict[str, Any]]:
        """
        Aplica BERTopic y retorna lista de dicts:
        { 'topic': int, 'probability': float, 'representation': List[str] }
        """
        topics, probs = self.topic_model.fit_transform(documents)
        reps = self.topic_model.get_topic_info()
        results = []
        for topic, prob in zip(topics, probs):
            info = self.topic_model.get_topic(topic)
            results.append({
                "topic": topic,
                "probability": float(prob),
                "representation": [t[0] for t in info]
            })
        return results

    def cluster_embeddings(self, documents: List[str]) -> List[int]:
        """
        Genera embeddings y aplica KMeans para agrupar documentos.
        Retorna lista de etiquetas de cluster.
        """
        embeddings = self.embedder.encode(documents, convert_to_tensor=False)
        kmeans = KMeans(n_clusters=self.n_clusters)
        labels = kmeans.fit_predict(embeddings)
        return labels.tolist()

    def analyze_sentiments(self, documents: List[str]) -> List[Dict[str, Any]]:
        """
        Aplica pipeline de sentimiento y retorna lista de resultados:
        { 'label': str, 'score': float }
        """
        return self.sentiment(documents, batch_size=16)

# Ejemplo de uso
if __name__ == "__main__":
    docs = [
        "Me sentí apoyado en la escuela, pero a veces había discriminación.",
        "Tuve experiencias muy positivas con algunos profesores."
    ]
    engine = NLPEngine(n_clusters=2)

    # Topics
    top_res = engine.extract_topics(docs)
    print("Topics:", top_res)

    # Clusters
    clusters = engine.cluster_embeddings(docs)
    print("Clusters:", clusters)

    # Sentiments
    senti = engine.analyze_sentiments(docs)
    print("Sentiments:", senti)
