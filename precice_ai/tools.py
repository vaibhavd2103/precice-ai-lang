import os
import subprocess
import urllib.parse
from pathlib import Path
from typing import TYPE_CHECKING

import requests
from langchain_core.tools import tool

from precice_ai import config
from precice_ai.vectorstore import get_vectorstore


@tool
def search_precice_docs(query: str) -> str:
    """
    Search the preCICE documentation, FAQ, and forum for information.
    Use this for any question about preCICE concepts, configuration, API,
    installation, tutorials, or troubleshooting.
    Returns relevant text chunks with source URLs.
    """
    try:
        vs = get_vectorstore()
        if vs._collection.count() == 0:
            return "The knowledge base is still being indexed. Please try again in a moment."
        retriever = vs.as_retriever(
            search_type="mmr",
            search_kwargs={"k": config.MAX_RETRIEVAL_CHUNKS, "fetch_k": 20, "lambda_mult": 0.6},
        )
        results = retriever.invoke(query)
        if not results:
            return "No relevant documentation found for that query."
        parts = []
        for doc in results:
            source = doc.metadata.get("source", "unknown")
            parts.append(f"[Source: {source}]\n{doc.page_content}\n---")
        return "\n\n".join(parts)
    except Exception as e:
        return f"Error searching docs: {e}"


@tool
def list_project_files(working_dir: str, extension_filter: str = "") -> str:
    """
    List all files in the user's simulation project directory.
    working_dir: absolute path to the project folder.
    extension_filter: optional file extension to filter by, e.g. ".xml" or ".py".
    Returns a tree-style listing of files with sizes.
    """
    try:
        base = Path(working_dir).resolve()
        if not base.exists() or not base.is_dir():
            return f"Directory not found: {working_dir}"

        lines = [f"{base}/"]
        for root, dirs, files in os.walk(base):
            # Skip hidden/build dirs
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("__pycache__", "node_modules")]
            root_path = Path(root)
            # Security: only yield paths inside base
            try:
                rel_root = root_path.resolve().relative_to(base)
            except ValueError:
                continue
            depth = len(rel_root.parts)
            indent = "  " * depth
            if rel_root.parts:
                lines.append(f"{indent}{root_path.name}/")
            for fname in sorted(files):
                if extension_filter and not fname.endswith(extension_filter):
                    continue
                fpath = root_path / fname
                try:
                    size = fpath.stat().st_size
                    file_indent = "  " * (depth + 1)
                    lines.append(f"{file_indent}{fname}  ({size:,} bytes)")
                except OSError:
                    pass
        return "\n".join(lines) if lines else "No files found."
    except Exception as e:
        return f"Error listing files: {e}"


@tool
def read_project_file(file_path: str, working_dir: str) -> str:
    """
    Read the contents of a file in the user's project directory.
    Use this to read precice-config.xml, solver scripts, log files,
    CMakeLists.txt, or any other project file the user asks about.
    file_path: path relative to working_dir, or absolute path inside working_dir.
    working_dir: absolute path to the project folder (provided by session context).
    """
    try:
        base = Path(working_dir).resolve()
        candidate = Path(file_path)
        if not candidate.is_absolute():
            candidate = base / candidate
        resolved = candidate.resolve()

        if not resolved.is_relative_to(base):
            return f"Access denied: {file_path} is outside the project directory."
        if not resolved.exists():
            return f"File not found: {file_path}"
        if not resolved.is_file():
            return f"Not a file: {file_path}"

        MAX_BYTES = 50 * 1024
        try:
            data = resolved.read_bytes()
        except OSError as e:
            return f"Error reading file: {e}"

        truncated = len(data) > MAX_BYTES
        data = data[:MAX_BYTES]
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return f"Binary file — cannot display: {file_path}"

        result = f"# {resolved}\n\n{text}"
        if truncated:
            result += f"\n\n[Truncated — showing first 50 KB of file]"
        return result
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def write_project_file(file_path: str, content: str, working_dir: str) -> str:
    """
    Write or overwrite a file in the user's project directory.
    Use this to create or modify precice-config.xml, scripts, or config files.
    IMPORTANT: Only call this when the user explicitly asks to create or modify a file.
    Always show the user what you are about to write before calling this tool.
    file_path: path relative to working_dir.
    content: the complete file content to write.
    working_dir: absolute path to the project folder.
    Returns confirmation with the absolute path written.
    """
    try:
        base = Path(working_dir).resolve()
        candidate = Path(file_path)
        if not candidate.is_absolute():
            candidate = base / candidate
        resolved = candidate.resolve()

        if not resolved.is_relative_to(base):
            return f"Access denied: {file_path} is outside the project directory."

        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return f"Written {len(content)} chars to {resolved}"
    except Exception as e:
        return f"Error writing file: {e}"


@tool
def validate_precice_config(config_path: str, working_dir: str) -> str:
    """
    Validate a preCICE XML configuration file using the preCICE config checker.
    Runs `precice-tools check <config_path>` and returns any errors or warnings.
    Use this when the user has a config file and wants to check it for errors.
    config_path: path to the precice-config.xml file, relative to working_dir.
    working_dir: absolute path to the project folder.
    """
    try:
        base = Path(working_dir).resolve()
        candidate = Path(config_path)
        if not candidate.is_absolute():
            candidate = base / candidate
        resolved = candidate.resolve()

        if not resolved.is_relative_to(base):
            return f"Access denied: {config_path} is outside the project directory."
        if not resolved.exists():
            return f"Config file not found: {config_path}"

        result = subprocess.run(
            ["precice-tools", "check", str(resolved)],
            capture_output=True, text=True, timeout=30,
        )
        output = (result.stdout + result.stderr).strip()
        if result.returncode == 0 and not output:
            return "Config is valid."
        return output or "Config is valid."
    except FileNotFoundError:
        return (
            "precice-tools not found. Install preCICE to enable config validation. "
            "See https://precice.org/installation-overview.html"
        )
    except subprocess.TimeoutExpired:
        return "Validation timed out after 30 seconds."
    except Exception as e:
        return f"Error validating config: {e}"


@tool
def search_forum_live(query: str) -> str:
    """
    Search the preCICE Discourse forum in real-time for recent discussions.
    Use this for very recent questions, bugs, or topics not yet in the local index.
    query: search terms to look for on the forum.
    Returns up to 5 relevant topic titles, summaries, and URLs.
    """
    try:
        url = f"https://precice.discourse.group/search.json?q={urllib.parse.quote(query)}"
        resp = requests.get(url, timeout=15, headers={"User-Agent": "preCICE-AI/0.1"})
        resp.raise_for_status()
        data = resp.json()
        topics = data.get("topics", [])[:5]
        if not topics:
            return "No forum results found for that query."
        parts = []
        for t in topics:
            title = t.get("title", "")
            slug = t.get("slug", "")
            tid = t.get("id", "")
            excerpt = t.get("excerpt", "")
            forum_url = f"https://precice.discourse.group/t/{slug}/{tid}"
            parts.append(f"**{title}**\n{excerpt}\n{forum_url}")
        return "\n\n---\n\n".join(parts)
    except Exception as e:
        return f"Error searching forum: {e}"


@tool
def read_attached_file(filename: str, session_id: str) -> str:
    """
    Read the content of a file attached by the user in the chat interface.
    Use this when the user says they have attached a file or asks about an uploaded file.
    filename: the name of the file as uploaded.
    session_id: the current session ID.
    Returns the file content as text.
    """
    try:
        # Import here to avoid circular import at module level
        from precice_ai.conversation import ATTACHMENT_STORE
        session_files = ATTACHMENT_STORE.get(session_id, {})
        if filename not in session_files:
            available = list(session_files.keys())
            return (
                f"File '{filename}' not found in session attachments. "
                f"Available: {available or 'none'}"
            )
        return session_files[filename]
    except Exception as e:
        return f"Error reading attached file: {e}"


ALL_TOOLS = [
    search_precice_docs,
    list_project_files,
    read_project_file,
    write_project_file,
    validate_precice_config,
    search_forum_live,
    read_attached_file,
]
