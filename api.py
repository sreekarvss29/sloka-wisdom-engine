"""
Sloka Wisdom Engine — FastAPI Backend
Serves sloka recommendations via REST API.
"""
import json
import os
from contextlib import asynccontextmanager

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

load_dotenv()

DB_DIR = "vector_db"
TOP_K = 5

# Global collection reference
collection = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global collection
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(name="slokas", embedding_function=ef)
    print(f"Loaded collection with {collection.count()} slokas.")
    yield


app = FastAPI(
    title="Sloka Wisdom Engine",
    description="Get ancient Sanskrit wisdom for modern life situations",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

AVAILABLE_PATHS = {
    "karma_yoga": "Path of Selfless Action (career, duty, purpose)",
    "bhakti_yoga": "Path of Devotion (relationships, love, faith)",
    "jnana_yoga": "Path of Knowledge (clarity, wisdom, self-awareness)",
    "dhyana_yoga": "Path of Meditation (peace, anxiety, focus)",
    "sankhya_yoga": "Path of Wisdom (grief, loss, resilience)",
    "raja_yoga": "Path of Discipline (health, habits, self-control)",
}


def _retrieve(user_query: str, top_k: int = TOP_K, path_filter: str = None):
    where_filter = None
    if path_filter:
        where_filter = {"path": path_filter}

    results = collection.query(
        query_texts=[user_query],
        n_results=top_k,
        where=where_filter,
        include=["metadatas", "distances"],
    )

    slokas = []
    for meta, distance in zip(results["metadatas"][0], results["distances"][0]):
        sloka = dict(meta)
        sloka["life_themes"] = json.loads(sloka["life_themes"])
        sloka["relevance_score"] = round(1 - distance, 3)
        slokas.append(sloka)

    return slokas


@app.get("/api/search")
def search_slokas(
    q: str = Query(..., description="Your life situation or question"),
    path: str = Query(None, description="Filter by yoga path"),
    top_k: int = Query(TOP_K, ge=1, le=20, description="Number of results"),
):
    """Search for relevant slokas based on a life situation."""
    slokas = _retrieve(q, top_k=top_k, path_filter=path)
    sources = list(dict.fromkeys(s["source"] for s in slokas))
    return {
        "query": q,
        "path_filter": path,
        "count": len(slokas),
        "sources": sources,
        "slokas": slokas,
    }


@app.get("/api/paths")
def get_paths():
    """Get available yoga paths for filtering."""
    return {"paths": AVAILABLE_PATHS}


@app.get("/api/stats")
def get_stats():
    """Get index statistics."""
    count = collection.count()
    return {"total_slokas": count}


# Serve static frontend
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    def serve_frontend():
        return FileResponse(os.path.join(static_dir, "index.html"))
