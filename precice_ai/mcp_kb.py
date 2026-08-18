from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_KB_STORE_DIR = Path.home() / ".precice-ai" / "kb_store"
VECTOR_PREFIX = "kb-embeddings-"
VECTOR_SUFFIX = ".npz"


def get_kb_store_dir() -> Path:
    env = os.environ.get("PRECICE_KB_STORE_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_KB_STORE_DIR


def get_lexical_kb_path() -> Path:
    return get_kb_store_dir() / "knowledge_base.json"


def get_vector_paths() -> dict[str, Path]:
    kb_dir = get_kb_store_dir()
    paths: dict[str, Path] = {}
    if not kb_dir.exists():
        return paths
    for path in sorted(kb_dir.glob(f"{VECTOR_PREFIX}*{VECTOR_SUFFIX}")):
        category = path.name.removeprefix(VECTOR_PREFIX).removesuffix(VECTOR_SUFFIX)
        if category:
            paths[category] = path
    return paths


def has_local_kb() -> bool:
    return bool(get_vector_paths()) or get_lexical_kb_path().exists()


def query_local_kb(question: str, top_k: int = 5) -> dict[str, Any]:
    payload = _read_lexical_kb()
    if not payload:
        return {
            "status": "error",
            "message": f"No lexical KB found at {get_lexical_kb_path()}",
        }

    documents = payload.get("documents", [])
    if not isinstance(documents, list) or not documents:
        return {"status": "error", "message": "Local lexical KB has no documents."}

    query_terms = _tokenize(question)
    if not query_terms:
        return {"status": "error", "message": "Query is empty after tokenization."}

    scored: list[tuple[float, dict[str, Any]]] = []
    for doc in documents:
        if not isinstance(doc, dict):
            continue
        text = f"{doc.get('title', '')}\n{doc.get('content', '')}"
        score = _bm25_like_score(query_terms, _tokenize(text))
        if score > 0:
            scored.append((score, doc))

    scored.sort(key=lambda item: item[0], reverse=True)
    results = []
    for score, doc in scored[:top_k]:
        results.append(
            {
                "score": round(score, 4),
                "source": str(doc.get("source", "unknown")),
                "title": str(doc.get("title", "")),
                "url": str(doc.get("url", "")),
                "snippet": _snippet_for_terms(str(doc.get("content", "")), query_terms),
            }
        )

    return {
        "status": "ok",
        "updated_at": payload.get("updated_at", "unknown"),
        "results": results,
    }


def local_kb_status() -> dict[str, Any]:
    kb_dir = get_kb_store_dir()
    vector_paths = get_vector_paths()
    lexical_path = get_lexical_kb_path()

    categories: dict[str, Any] = {}
    total_chunks = 0
    latest_updated: str | None = None
    for category, path in vector_paths.items():
        chunk_count = _read_npz_chunk_count(path)
        updated_at = _mtime_to_iso(path)
        latest_updated = _max_iso(latest_updated, updated_at)
        if chunk_count is not None:
            total_chunks += chunk_count
        categories[category] = {
            "path": str(path),
            "size_mb": round(path.stat().st_size / 1_048_576, 2),
            "chunk_count": chunk_count,
            "updated_at": updated_at,
        }

    lexical_documents = 0
    lexical_updated = None
    payload = _read_lexical_kb()
    if payload:
        documents = payload.get("documents", [])
        if isinstance(documents, list):
            lexical_documents = len(documents)
        lexical_updated_raw = payload.get("updated_at")
        if isinstance(lexical_updated_raw, str):
            lexical_updated = lexical_updated_raw
            latest_updated = _max_iso(latest_updated, lexical_updated_raw)
    elif lexical_path.exists():
        lexical_updated = _mtime_to_iso(lexical_path)
        latest_updated = _max_iso(latest_updated, lexical_updated)

    ready = bool(categories) or lexical_documents > 0
    return {
        "state": "ready" if ready else "starting",
        "kb_dir": str(kb_dir),
        "backend": "mcp-local-kb",
        "vector_categories": len(categories),
        "chunk_count": total_chunks,
        "source_count": lexical_documents,
        "last_updated": latest_updated,
        "categories": categories,
        "lexical_kb": {
            "path": str(lexical_path),
            "exists": lexical_path.exists(),
            "documents": lexical_documents,
            "updated_at": lexical_updated,
        },
    }


def _read_npz_chunk_count(path: Path) -> int | None:
    try:
        import numpy as np
    except ImportError:
        return None

    try:
        with np.load(path, allow_pickle=True) as data:
            chunks = data["chunks"].tolist()
            if isinstance(chunks, str):
                return len(json.loads(chunks))
            if hasattr(chunks, "__len__"):
                return len(chunks)
    except Exception:
        return None
    return None


def _read_lexical_kb() -> dict[str, Any] | None:
    path = get_lexical_kb_path()
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]{2,}", text.lower())


def _bm25_like_score(query_terms: list[str], doc_terms: list[str]) -> float:
    if not query_terms or not doc_terms:
        return 0.0
    doc_len = len(doc_terms)
    if doc_len == 0:
        return 0.0
    tf: dict[str, int] = {}
    for term in doc_terms:
        tf[term] = tf.get(term, 0) + 1
    score = 0.0
    for term in query_terms:
        freq = tf.get(term, 0)
        if freq == 0:
            continue
        score += (freq / (freq + 1.2 * (0.25 + 0.75 * (doc_len / 1000.0)))) * (1.0 + math.log1p(freq))
    return score


def _snippet_for_terms(content: str, terms: list[str], size: int = 320) -> str:
    lowered = content.lower()
    for term in terms:
        idx = lowered.find(term)
        if idx >= 0:
            start = max(0, idx - size // 3)
            end = min(len(content), start + size)
            return content[start:end].strip()
    return content[:size].strip()


def _mtime_to_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _max_iso(left: str | None, right: str | None) -> str | None:
    if not left:
        return right
    if not right:
        return left
    try:
        left_dt = datetime.fromisoformat(left.replace("Z", "+00:00"))
        right_dt = datetime.fromisoformat(right.replace("Z", "+00:00"))
    except ValueError:
        return right or left
    return left if left_dt >= right_dt else right
