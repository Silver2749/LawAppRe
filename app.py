#!/usr/bin/env python3
"""LawAppRe – IPC Semantic Search using InsightLaw API"""

import os
from typing import List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# InsightLaw API
INSIGHTLAW_BASE = "https://insightlaw.in"
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = FastAPI(title="LawAppRe – IPC Search via InsightLaw", version="2.0.0")


class SearchRequest(BaseModel):
    query: str
    top_n: Optional[int] = 5
    corpus: Optional[str] = "ipc"  # "ipc" or "bns"


class SearchResult(BaseModel):
    section: str
    title: str
    description: str
    score: float
    corpus: str


async def search_insightlaw(
    query: str, corpus: str = "ipc", top_n: int = 5
) -> List[SearchResult]:
    """Call InsightLaw search API"""
    if corpus not in ("ipc", "bns"):
        corpus = "ipc"

    url = f"{INSIGHTLAW_BASE}/api/{corpus}/search"
    params = {"q": query.strip()}

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as e:
            raise HTTPException(
                status_code=502, detail=f"InsightLaw API error: {str(e)}"
            )

    results = []
    for item in data.get("results", [])[:top_n]:
        preview = item.get("preview", {})
        description = preview.get("en") or preview.get("hi") or preview.get("ml") or ""

        results.append(
            SearchResult(
                section=str(item.get("section", "")),
                title=item.get("title_en") or item.get("title", ""),
                description=description,
                score=float(item.get("score", 0)),
                corpus=data.get("corpus", corpus.upper()),
            )
        )
    return results


@app.post("/search", response_model=List[SearchResult])
async def search(body: SearchRequest):
    if not body.query or not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    return await search_insightlaw(body.query, body.corpus or "ipc", body.top_n or 5)


@app.get("/section/{corpus}/{number}")
async def get_section(corpus: str, number: str):
    """Fetch a single section by number (ipc or bns)"""
    if corpus not in ("ipc", "bns"):
        raise HTTPException(status_code=400, detail="corpus must be 'ipc' or 'bns'")

    url = f"{INSIGHTLAW_BASE}/api/{corpus}/section/{number}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(url)
        if resp.status_code == 404:
            raise HTTPException(status_code=404, detail="Section not found")
        resp.raise_for_status()
        return resp.json()


@app.get("/health")
async def health():
    """Check both our service and InsightLaw"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{INSIGHTLAW_BASE}/api/health")
            insight_status = (
                r.json() if r.status_code == 200 else {"status": "unavailable"}
            )
    except Exception:
        insight_status = {"status": "unavailable"}

    return {
        "status": "ok",
        "backend": "InsightLaw API",
        "insightlaw": insight_status,
    }


# Serve frontend
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
