FROM python:3.12-slim

WORKDIR /app

# System deps for sentence-transformers / torch CPU
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the model so first request is fast
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY data/ data/
COPY static/ static/
COPY app.py .
COPY generate_embeddings.py .

# Generate embeddings if not already present (build-time)
RUN if [ ! -f data/embedded_ipc.json ]; then python generate_embeddings.py; fi

ENV PORT=8080
EXPOSE 8080

CMD ["python", "app.py"]
