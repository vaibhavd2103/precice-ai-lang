import hashlib
import logging
import time
from datetime import datetime, timezone
from typing import List

import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from precice_ai.vectorstore import get_vectorstore

logger = logging.getLogger(__name__)

DOCS_URLS = [
    "https://precice.org/configuration-overview.html",
    "https://precice.org/configuration-coupling-scheme.html",
    "https://precice.org/configuration-coupling-scheme-two-participant.html",
    "https://precice.org/configuration-mapping.html",
    "https://precice.org/configuration-participants.html",
    "https://precice.org/configuration-mesh.html",
    "https://precice.org/configuration-data.html",
    "https://precice.org/configuration-action.html",
    "https://precice.org/configuration-logging.html",
    "https://precice.org/couple-your-code-api.html",
    "https://precice.org/couple-your-code-preparing-your-solver.html",
    "https://precice.org/couple-your-code-mesh-and-data-access.html",
    "https://precice.org/couple-your-code-initialization.html",
    "https://precice.org/couple-your-code-implicit-coupling.html",
    "https://precice.org/couple-your-code-time-step-sizes.html",
    "https://precice.org/installation-overview.html",
    "https://precice.org/installation-packages.html",
    "https://precice.org/installation-source-cmake.html",
    "https://precice.org/installation-spack.html",
    "https://precice.org/tutorials-overview.html",
    "https://precice.org/tutorials-flow-over-heated-plate.html",
    "https://precice.org/tutorials-perpendicular-flap.html",
    "https://precice.org/tutorials-elastic-tube-1d.html",
    "https://precice.org/troubleshooting.html",
    "https://precice.org/faq.html",
    "https://precice.org/tooling-precice-config-visualizer.html",
    "https://precice.org/tooling-aste.html",
]

status = {
    "state": "starting",
    "message": "",
    "chunk_count": 0,
    "source_count": 0,
    "last_updated": None,
}


def _fetch_text(url: str, timeout: int = 15) -> str:
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "preCICE-AI/0.1"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        # Remove script/style noise
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)
    except Exception as e:
        logger.warning("Failed to fetch %s: %s", url, e)
        return ""


def _scrape_docs() -> List[Document]:
    docs = []
    for url in DOCS_URLS:
        text = _fetch_text(url)
        if text:
            docs.append(Document(page_content=text, metadata={"source": url, "type": "docs"}))
            logger.info("Fetched docs page: %s (%d chars)", url, len(text))
    return docs


def _scrape_forum() -> List[Document]:
    docs = []
    base = "https://precice.discourse.group"
    try:
        resp = requests.get(f"{base}/latest.json", timeout=15, headers={"User-Agent": "preCICE-AI/0.1"})
        resp.raise_for_status()
        topics = resp.json().get("topic_list", {}).get("topics", [])[:50]
    except Exception as e:
        logger.warning("Failed to fetch forum topics: %s", e)
        return docs

    for topic in topics:
        tid = topic.get("id")
        title = topic.get("title", "")
        url = f"{base}/t/{tid}"
        try:
            time.sleep(0.3)
            tr = requests.get(f"{url}.json", timeout=15, headers={"User-Agent": "preCICE-AI/0.1"})
            tr.raise_for_status()
            posts = tr.json().get("post_stream", {}).get("posts", [])[:5]
            text_parts = []
            for post in posts:
                raw = post.get("cooked", "") or post.get("raw", "")
                soup = BeautifulSoup(raw, "html.parser")
                text_parts.append(soup.get_text(separator="\n", strip=True))
            combined = f"{title}\n\n" + "\n\n---\n\n".join(text_parts)
            if combined.strip():
                docs.append(Document(
                    page_content=combined,
                    metadata={"source": url, "title": title, "type": "forum"},
                ))
        except Exception as e:
            logger.warning("Failed to fetch forum topic %s: %s", tid, e)

    logger.info("Fetched %d forum topics", len(docs))
    return docs


def run_ingestion() -> None:
    global status
    status["state"] = "ingesting"
    status["message"] = "Fetching documentation and forum posts..."
    status["last_updated"] = datetime.now(timezone.utc).isoformat()

    try:
        all_docs = []
        all_docs.extend(_scrape_docs())
        all_docs.extend(_scrape_forum())

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=120,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(all_docs)

        vs = get_vectorstore()
        ids = []
        texts = []
        metadatas = []
        for i, chunk in enumerate(chunks):
            source = chunk.metadata.get("source", "unknown")
            doc_id = hashlib.md5(f"{source}::{i}".encode()).hexdigest()
            ids.append(doc_id)
            texts.append(chunk.page_content)
            metadatas.append(chunk.metadata)

        if texts:
            vs.add_texts(texts=texts, metadatas=metadatas, ids=ids)

        status["state"] = "ready"
        status["chunk_count"] = len(chunks)
        status["source_count"] = len(all_docs)
        status["message"] = f"Ready — {len(chunks)} chunks from {len(all_docs)} sources"
        status["last_updated"] = datetime.now(timezone.utc).isoformat()
        logger.info("Ingestion complete: %d chunks from %d sources", len(chunks), len(all_docs))

    except Exception as e:
        status["state"] = "error"
        status["message"] = str(e)
        status["last_updated"] = datetime.now(timezone.utc).isoformat()
        logger.error("Ingestion failed: %s", e)
