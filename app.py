#!/usr/bin/env python3
"""IPC Semantic Search API + static frontend"""

import json
import os
from pathlib import Path
from typing import List

import numpy as np
import redis
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent / "data"
EMBEDDED_PATH = DATA_DIR / "embedded_ipc.json"
STATIC_DIR = Path(__file__).parent / "static"

# Redis connection (change host/port/password as needed)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

REDIS_KEY = "lawappre:embedded_ipc"  # key used to store the data

app = FastAPI(title="IPC Semantic Search", version="1.0.0")


def load_laws_from_json():
    """Fallback: load from the original JSON file"""
    with open(EMBEDDED_PATH) as f:
        return json.load(f)


def get_laws():
    """
    Try to load from Redis first.
    If not present, load from JSON and cache it in Redis.
    """
    cached = redis_client.get(REDIS_KEY)
    if cached:
        print("Loaded embeddings from Redis cache")
        return json.loads(cached)

    print("Redis cache miss → loading from JSON and caching...")
    laws = load_laws_from_json()

    # Store in Redis (no expiry, or set an expiry if you want)
    redis_client.set(REDIS_KEY, json.dumps(laws))
    # Optional: expire after 7 days
    # redis_client.setex(REDIS_KEY, 60 * 60 * 24 * 7, json.dumps(laws))

    print(f"Cached {len(laws)} sections in Redis")
    return laws


# ---------- Load data & model at startup ----------
print("Loading embeddings...")
LAWS = get_laws()

EMBEDDINGS = np.array([law["embedding"] for law in LAWS], dtype=np.float32)
NORMS = np.linalg.norm(EMBEDDINGS, axis=1, keepdims=True)
NORMS[NORMS == 0] = 1.0
EMBEDDINGS_NORM = EMBEDDINGS / NORMS

print(f"Loaded {len(LAWS)} sections. Loading query model...")
MODEL = SentenceTransformer("all-MiniLM-L6-v2")
print("Ready.")


class SearchRequest(BaseModel):
    query: str


class SearchResult(BaseModel):
    similarity: float
    section: str
    title: str
    description: str


def cosine_search(query: str, top_n: int = 5) -> List[SearchResult]:
    q_emb = MODEL.encode([query], convert_to_numpy=True)[0].astype(np.float32)
    q_norm = np.linalg.norm(q_emb)
    if q_norm == 0:
        q_norm = 1.0
    q_emb = q_emb / q_norm

    scores = EMBEDDINGS_NORM @ q_emb
    top_idx = np.argsort(scores)[::-1][:top_n]

    results = []
    for i in top_idx:
        law = LAWS[i]
        results.append(
            SearchResult(
                similarity=float(scores[i]),
                section=law["section"],
                title=law["title"],
                description=law["description"],
            )
        )
    return results


@app.post("/search", response_model=List[SearchResult])
def search(body: SearchRequest):
    return cosine_search(body.query.strip(), top_n=5)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "sections": len(LAWS),
        "cache": "redis" if redis_client.exists(REDIS_KEY) else "json",
    }


# Optional: endpoint to force refresh the cache
@app.post("/refresh-cache")
def refresh_cache():
    laws = load_laws_from_json()
    redis_client.set(REDIS_KEY, json.dumps(laws))
    return {"message": f"Cache refreshed with {len(laws)} sections"}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
