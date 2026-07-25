import os
from pathlib import Path

OPENROUTER_API_KEY: str = os.environ.get("OPENROUTER_API_KEY", "")
MODEL: str = os.environ.get("PRECICE_AI_MODEL", "mistralai/mistral-7b-instruct:free")
HOST: str = os.environ.get("PRECICE_AI_HOST", "127.0.0.1")
PORT: int = int(os.environ.get("PRECICE_AI_PORT", "7860"))
EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_COLLECTION: str = "precice_docs"
INGEST_INTERVAL_HOURS: int = 1
MAX_RETRIEVAL_CHUNKS: int = 6
MAX_SESSION_MESSAGES: int = 20
