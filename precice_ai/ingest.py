from __future__ import annotations

from datetime import datetime, timezone

from precice_ai.mcp_kb import local_kb_status


status = {
    "state": "starting",
    "message": "Checking local preCICE MCP knowledge base...",
    "chunk_count": 0,
    "source_count": 0,
    "last_updated": None,
    "backend": "mcp-local-kb",
    "kb_dir": None,
    "category_count": 0,
}


def run_ingestion() -> None:
    details = local_kb_status()
    status["backend"] = str(details.get("backend", "mcp-local-kb"))
    status["kb_dir"] = details.get("kb_dir")
    status["category_count"] = int(details.get("vector_categories", 0) or 0)
    status["chunk_count"] = int(details.get("chunk_count", 0) or 0)
    status["source_count"] = int(details.get("source_count", 0) or 0)
    status["last_updated"] = details.get("last_updated") or _now_iso()

    if details.get("state") == "ready":
        status["state"] = "ready"
        status["message"] = (
            f"Ready - connected to local MCP KB with "
            f"{status['category_count']} vector categories, "
            f"{status['chunk_count']} chunks, and {status['source_count']} lexical sources."
        )
        return

    status["state"] = "starting"
    status["message"] = (
        "Local MCP KB not found yet. Expected vector .npz assets or knowledge_base.json "
        f"in {status['kb_dir']}."
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


run_ingestion()
