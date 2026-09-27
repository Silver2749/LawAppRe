#!/usr/bin/env python3
"""Precompute embeddings for all IPC sections and save to embedded_ipc.json"""

import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np

DATA_DIR = Path(__file__).parent / "data"
IPC_PATH = DATA_DIR / "ipc.json"
OUT_PATH = DATA_DIR / "embedded_ipc.json"

def main():
    print("Loading IPC data...")
    with open(IPC_PATH) as f:
        laws = json.load(f)
    print(f"Loaded {len(laws)} sections")

    print("Loading model all-MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    documents = []
    for law in laws:
        section = str(law["Section"])
        doc = (
            f"Chapter: {law['chapter_title']}\n\n"
            f"Section: {section}\n\n"
            f"Title: {law['section_title']}\n\n"
            f"Description:\n{law['section_desc']}"
        )
        documents.append(doc)

    print("Encoding documents (this may take a minute)...")
    embeddings = model.encode(documents, show_progress_bar=True, convert_to_numpy=True)

    embedded = []
    for i, law in enumerate(laws):
        embedded.append({
            "section": str(law["Section"]),
            "title": law["section_title"],
            "description": law["section_desc"],
            "embedding": embeddings[i].tolist(),
        })

    print(f"Saving {len(embedded)} embeddings to {OUT_PATH}...")
    with open(OUT_PATH, "w") as f:
        json.dump(embedded, f)

    print("Done.")

if __name__ == "__main__":
    main()
