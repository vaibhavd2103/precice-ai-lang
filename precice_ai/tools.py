import os
import asyncio
import importlib.util
import subprocess
import sys
import urllib.parse
from pathlib import Path
from typing import Any

import requests
from langchain_core.tools import tool

from precice_ai import config
from precice_ai.mcp_kb import query_local_kb
from precice_ai.logger import log_event

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
except ImportError:
    MultiServerMCPClient = None


@tool
def search_precice_docs(query: str) -> str:
    """
    Search the preCICE documentation, FAQ, and forum for information.
    Use this for any question about preCICE concepts, configuration, API,
    installation, tutorials, or troubleshooting.
    Returns relevant text chunks with source URLs.
    """
    try:
        result = query_local_kb(query, top_k=config.MAX_RETRIEVAL_CHUNKS)
        if result.get("status") != "ok":
            message = result.get("message", "unknown error")
            return f"The local MCP knowledge base is not ready: {message}"

        results = result.get("results", [])
        if not results:
            return "No relevant documentation found for that query."

        parts = []
        for doc in results:
            source = doc.get("url") or doc.get("source", "unknown")
            title = doc.get("title", "")
            snippet = doc.get("snippet", "")
            score = doc.get("score", "")
            header = f"[Source: {source}]"
            if title:
                header += f"\n[Title: {title}]"
            if score != "":
                header += f"\n[Score: {score}]"
            parts.append(f"{header}\n{snippet}\n---")
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

LOCAL_TOOLS = [
    search_precice_docs,
    list_project_files,
    read_project_file,
    write_project_file,
    validate_precice_config,
    search_forum_live,
    read_attached_file,
]

_mcp_client = None
_mcp_tools: list[Any] = []
_tool_status: dict[str, Any] = {
    "initialized": False,
    "local_tools": [tool_.name for tool_ in LOCAL_TOOLS],
    "mcp_tools": [],
    "startup_tools": [tool_.name for tool_ in LOCAL_TOOLS],
    "mcp_server_path": None,
    "mcp_python": None,
    "mcp_connected": False,
    "mcp_error": None,
}


def _default_mcp_server_path() -> Path:
    return (Path(__file__).resolve().parents[2] / "precice-ai" / "server.py").resolve()


def _get_mcp_server_path() -> Path:
    configured = os.environ.get("PRECICE_AI_MCP_SERVER")
    if configured:
        return Path(configured).expanduser().resolve()
    return _default_mcp_server_path()


def _get_mcp_python() -> str:
    return os.environ.get("PRECICE_AI_MCP_PYTHON", sys.executable)


def _check_python_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _preflight_mcp_server(server_path: Path, python_bin: str) -> str | None:
    required_modules = [
        ("dotenv", "python-dotenv"),
        ("mcp", "mcp"),
        ("fastmcp", "fastmcp"),
        ("httpx", "httpx"),
        ("lxml", "lxml"),
        ("numpy", "numpy"),
        ("openai", "openai"),
    ]
    missing = [package for module, package in required_modules if not _check_python_module(module)]
    if missing and Path(python_bin).resolve() == Path(sys.executable).resolve():
        packages = ", ".join(missing)
        return (
            f"MCP server cannot start with {python_bin}: missing Python packages in the current "
            f"environment: {packages}. Install them or set PRECICE_AI_MCP_PYTHON to an interpreter "
            "where the sibling `precice-ai` repo is installed."
        )

    try:
        probe = subprocess.run(
            [python_bin, str(server_path)],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(server_path.parent),
        )
    except FileNotFoundError:
        return f"MCP Python interpreter not found: {python_bin}"
    except subprocess.TimeoutExpired:
        return None
    except Exception as exc:
        return f"Failed to probe MCP server startup: {exc}"

    stderr = (probe.stderr or "").strip()
    stdout = (probe.stdout or "").strip()
    if probe.returncode != 0:
        detail = stderr or stdout or f"exit code {probe.returncode}"
        return f"MCP server failed to start with {python_bin}: {detail}"
    return None


def get_all_tools() -> list[Any]:
    # The sibling MCP server is the source of truth for preCICE knowledge.
    # The old local index is opt-in, so a missing MCP connection cannot make
    # the model believe it queried the category-wise vector KB.
    allow_fallback = os.environ.get("PRECICE_AI_ALLOW_LOCAL_SEARCH_FALLBACK", "false").lower() in {
        "1", "true", "yes",
    }
    local_tools = [
        tool_
        for tool_ in LOCAL_TOOLS
        if tool_ is not search_precice_docs
        or (_tool_status["mcp_connected"] is False and allow_fallback)
    ]
    return [*local_tools, *_mcp_tools]


def get_mcp_tool(name: str) -> Any | None:
    """Return a loaded MCP tool by name, if the MCP server is connected."""
    return next((tool_ for tool_ in _mcp_tools if tool_.name == name), None)


def get_available_tool_names() -> list[str]:
    return [tool_.name for tool_ in get_all_tools()]


def get_tool_status() -> dict[str, Any]:
    return {
        **_tool_status,
        "mcp_tools": list(_tool_status["mcp_tools"]),
        "startup_tools": list(_tool_status["startup_tools"]),
        "available_count": len(get_all_tools()),
    }


async def initialize_tools() -> dict[str, Any]:
    global _mcp_client, _mcp_tools

    _mcp_tools = []
    _tool_status["initialized"] = True
    _tool_status["mcp_connected"] = False
    _tool_status["mcp_error"] = None

    server_path = _get_mcp_server_path()
    python_bin = _get_mcp_python()
    _tool_status["mcp_server_path"] = str(server_path)
    _tool_status["mcp_python"] = python_bin

    if MultiServerMCPClient is None:
        _tool_status["mcp_error"] = "langchain_mcp_adapters is not installed"
    elif not server_path.exists():
        _tool_status["mcp_error"] = f"MCP server not found at {server_path}"
    else:
        preflight_error = _preflight_mcp_server(server_path, python_bin)
        if preflight_error:
            _tool_status["mcp_error"] = preflight_error
            _tool_status["mcp_tools"] = []
            _tool_status["startup_tools"] = get_available_tool_names()
            log_event(
                "mcp_startup",
                connected=False,
                tools=[],
                error=preflight_error,
                server_path=_tool_status["mcp_server_path"],
            )
            return get_tool_status()
        try:
            # The sibling MCP server reads OPENROUTER_API_KEY for embedding
            # queries, while this app accepts PRECICE_AI_API_KEY as well.
            # Pass both names to the child process explicitly.
            child_env = dict(os.environ)
            if config.LLM_API_KEY:
                child_env.setdefault("OPENROUTER_API_KEY", config.LLM_API_KEY)
            child_env.setdefault("EMBEDDING_BASE_URL", "https://openrouter.ai/api/v1")
            child_env.setdefault("EMBEDDING_MODEL", "openai/text-embedding-3-small")
            _mcp_client = MultiServerMCPClient(
                {
                    "precice_ai_mcp": {
                        "transport": "stdio",
                        "command": python_bin,
                        # Run the sibling package as a module so its package
                        # imports and .env resolution are deterministic.
                        "args": ["-m", "precice_ai.server"],
                        "env": child_env,
                        "cwd": str(server_path.parent),
                    }
                }
            )
            if hasattr(_mcp_client, "get_tools"):
                _mcp_tools = await asyncio.wait_for(_mcp_client.get_tools(), timeout=20)
            elif hasattr(_mcp_client, "list_tools"):
                _mcp_tools = await asyncio.wait_for(_mcp_client.list_tools(), timeout=20)
            else:
                raise RuntimeError("MCP client does not expose get_tools() or list_tools().")
            _tool_status["mcp_connected"] = True
        except Exception as exc:
            _mcp_tools = []
            if isinstance(exc, asyncio.TimeoutError):
                _tool_status["mcp_error"] = "Timed out after 20 seconds while loading MCP tools"
            else:
                _tool_status["mcp_error"] = str(exc)

    _tool_status["mcp_tools"] = [tool_.name for tool_ in _mcp_tools]
    _tool_status["startup_tools"] = get_available_tool_names()
    log_event(
        "mcp_startup",
        connected=_tool_status["mcp_connected"],
        tools=_tool_status["mcp_tools"],
        error=_tool_status["mcp_error"],
        server_path=_tool_status["mcp_server_path"],
    )
    return get_tool_status()


ALL_TOOLS = LOCAL_TOOLS
