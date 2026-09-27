# IPC Semantic Search

Describe an activity or offense in plain language and find the closest matching **Indian Penal Code (IPC)** sections using semantic embeddings (`all-MiniLM-L6-v2`).

Same core idea as the original Rust project (embed sections → cosine similarity → top matches), packaged as a ready-to-deploy web app.

## Features

- Semantic search over ~575 IPC sections
- Clean, responsive dark UI
- Example queries for quick testing
- REST API: `POST /search` with `{"query": "..."}`
- Health check: `GET /health`

## Quick start (local)

```bash
# 1. Create venv (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install deps
pip install -r requirements.txt

# 3. Generate embeddings (one-time, ~1–2 min)
python generate_embeddings.py

# 4. Run the server
python app.py
# → http://localhost:8080
```

## Docker

```bash
docker build -t ipc-search .
docker run -p 8080:8080 ipc-search
```

## Deploy

Works on any platform that can run a Python web process or Docker:

| Platform | Notes |
|----------|-------|
| **Render** | Web Service · Docker or native Python · set start command `python app.py` |
| **Railway** | Deploy from GitHub · auto-detects Dockerfile |
| **Fly.io** | `fly launch` then `fly deploy` |
| **Hugging Face Spaces** | Docker Space |
| **VPS** | Docker or systemd + uvicorn |

Set the `PORT` environment variable if the platform requires a specific port (defaults to 8080).

## API

```bash
curl -X POST http://localhost:8080/search \
  -H "Content-Type: application/json" \
  -d '{"query": "someone steals my phone"}'
```

Response:

```json
[
  {
    "similarity": 0.72,
    "section": "379",
    "title": "Punishment for theft",
    "description": "..."
  }
]
```

## Data

IPC sections from [civictech-India/Indian-Law-Penal-Code-Json](https://github.com/civictech-India/Indian-Law-Penal-Code-Json).

Place your own `ipc.json` in `data/` (same schema) and re-run `generate_embeddings.py` if needed.

## Disclaimer

This is **not legal advice**. Results are approximate semantic matches only.
