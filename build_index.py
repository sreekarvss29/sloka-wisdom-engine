"""
Build the ChromaDB vector index from the sloka dataset.
Run this once before using the query engine.
"""
import json
import chromadb
from chromadb.utils import embedding_functions

DATA_DIR = "data"
DB_DIR = "vector_db"

def load_slokas():
    """Load all sloka data from JSON files."""
    slokas = []
    sources = ["bhagavad_gita.json"]

    for source_file in sources:
        path = f"{DATA_DIR}/{source_file}"
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                source_name = data["source"]
                paths_info = data.get("paths", {})
                for sloka in data["slokas"]:
                    sloka["source"] = source_name
                    sloka["path_description"] = paths_info.get(sloka["path"], "")
                    slokas.append(sloka)
        except FileNotFoundError:
            print(f"Warning: {path} not found, skipping.")

    return slokas


def build_document_text(sloka):
    """
    Build a rich text document for embedding.
    This combines multiple fields so the vector search can match on
    life themes, meaning, and usage context.
    """
    themes = ", ".join(sloka["life_themes"])
    return (
        f"Life themes: {themes}. "
        f"When to use: {sloka['when_to_use']} "
        f"Meaning: {sloka['english_meaning']} "
        f"Path: {sloka['path']} — {sloka['path_description']} "
        f"Difficulty: {sloka['difficulty']}."
    )


def build_index():
    """Build the ChromaDB collection with sloka embeddings."""
    slokas = load_slokas()
    print(f"Loaded {len(slokas)} slokas.")

    # Use the default sentence-transformer for embeddings (runs locally, no API key needed)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    client = chromadb.PersistentClient(path=DB_DIR)

    # Delete existing collection if it exists, then recreate
    try:
        client.delete_collection("slokas")
    except Exception:
        pass

    collection = client.create_collection(
        name="slokas",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )

    # Prepare batch data
    ids = []
    documents = []
    metadatas = []

    for sloka in slokas:
        ids.append(sloka["id"])
        documents.append(build_document_text(sloka))
        metadatas.append({
            "id": sloka["id"],
            "source": sloka["source"],
            "chapter": sloka.get("chapter", 0),
            "verse": sloka.get("verse", 0),
            "sanskrit": sloka["sanskrit"],
            "transliteration": sloka["transliteration"],
            "english_meaning": sloka["english_meaning"],
            "path": sloka["path"],
            "path_description": sloka["path_description"],
            "life_themes": json.dumps(sloka["life_themes"]),
            "when_to_use": sloka["when_to_use"],
            "difficulty": sloka["difficulty"],
        })

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"Indexed {len(ids)} slokas into ChromaDB at '{DB_DIR}'.")
    print("Done! You can now run: python query.py")


if __name__ == "__main__":
    build_index()
