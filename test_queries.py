"""Quick test: run sample queries against the sloka engine."""
import json
import chromadb
from chromadb.utils import embedding_functions

DB_DIR = "vector_db"

ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_collection(name="slokas", embedding_function=ef)

test_queries = [
    "I am suffering in career growth, nothing is working despite hard work",
    "I need inner peace, my mind is always restless and anxious",
    "My relationship is falling apart and I feel heartbroken",
    "I am struggling with health issues and feel weak",
    "I feel lost and don't know what my purpose in life is",
    "I am addicted to social media and can't focus",
]

for query in test_queries:
    print(f"\n{'='*60}")
    print(f"  QUERY: {query}")
    print(f"{'='*60}")

    results = collection.query(query_texts=[query], n_results=3, include=["metadatas", "distances"])

    for i, (meta, dist) in enumerate(zip(results["metadatas"][0], results["distances"][0]), 1):
        score = round(1 - dist, 3)
        print(f"\n  [{i}] {meta['id']} | Score: {score} | Path: {meta['path']}")
        print(f"  Sanskrit: {meta['sanskrit'][:80]}...")
        print(f"  Meaning: {meta['english_meaning'][:100]}...")
        print(f"  Why: {meta['when_to_use'][:100]}...")
