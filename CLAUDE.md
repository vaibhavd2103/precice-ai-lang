# preCICE AI — Claude Code Project Brief

You are building **preCICE AI**: a local agentic assistant for the preCICE multiphysics
coupling library. Users run it on their machine, open a browser, select a simulation
project folder, and chat with an LLM agent that can read/write their local files,
search preCICE documentation, query the Discourse forum, and validate their configs.

Read this entire file before writing a single line of code. Follow every constraint
exactly. When in doubt, re-read the relevant section.

---

## 1. What you are building

A Python package called `precice-ai` that:

1. Is installed with `pip install -e .`
2. Is started with `precice-ai` (one command, no arguments required)
3. Opens `http://localhost:7860` automatically in the user's browser
4. Shows a chat interface where the user can:
   - Select a working directory (their simulation project folder)
   - Attach files as extra context (drag-and-drop or file picker)
   - Chat with an LLM agent in a streaming interface
   - See tool calls the agent makes (read file, search docs, etc.) shown inline
   - See source citations (docs page or forum thread) under each answer
5. Ingests preCICE documentation, FAQ, and Discourse forum on startup and
   re-ingests every hour automatically (background thread, no user action needed)
6. Never writes anything to a database — all sessions are in-memory only

---

## 2. Technology stack — use exactly these, no substitutions

| Layer               | Library                            | Notes                                    |
| ------------------- | ---------------------------------- | ---------------------------------------- |
| LLM                 | `langchain-openai` ChatOpenAI      | OpenRouter base URL, free models         |
| Agent orchestration | `langgraph`                        | ReAct loop with tool nodes               |
| RAG / chains        | `langchain`, `langchain-community` | Retriever, prompt templates              |
| Vector store        | `chromadb`                         | In-memory, no persistence                |
| Embeddings          | `sentence-transformers`            | all-MiniLM-L6-v2, local CPU              |
| Web scraping        | `requests`, `beautifulsoup4`       | docs + forum                             |
| Scheduler           | `apscheduler`                      | hourly re-ingestion                      |
| Web server          | `fastapi`, `uvicorn[standard]`     | SSE streaming                            |
| File watching       | standard `pathlib`, `os`           | no extra libs                            |
| Frontend            | Vanilla HTML/CSS/JS                | single index.html, no npm, no build step |

Do NOT add: Redis, PostgreSQL, SQLite, Celery, Docker, React, Vue, webpack,
or any other database or frontend framework. The entire thing must work with
`pip install -e .` on a plain Python 3.10+ environment.

---

## 3. LLM configuration

**Provider**: OpenRouter — `https://openrouter.ai/api/v1`
**Default model**: `mistralai/mistral-7b-instruct:free`
**Auth**: `OPENROUTER_API_KEY` environment variable (required)

LangChain setup:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model=os.environ.get("PRECICE_AI_MODEL", "mistralai/mistral-7b-instruct:free"),
    openai_api_key=os.environ["OPENROUTER_API_KEY"],
    openai_api_base="https://openrouter.ai/api/v1",
    streaming=True,
    temperature=0.1,
    max_tokens=1024,
    default_headers={
        "HTTP-Referer": "https://precice.org",
        "X-Title": "preCICE AI",
    },
)
```

The model must be bound with tools before passing to the graph:

```python
llm_with_tools = llm.bind_tools(tools)
```

Free models that support tool calling on OpenRouter (test in this order):

- `mistralai/mistral-7b-instruct:free`
- `meta-llama/llama-3.1-8b-instruct:free`
- `google/gemma-3-12b-it:free`

---

## 4. Project file structure — create exactly this

```
precice_ai/                     ← Python package root
  __init__.py
  config.py                     ← all settings, read from env
  ingest.py                     ← scraping + chunking + embedding
  vectorstore.py                ← ChromaDB singleton
  tools.py                      ← all @tool functions
  graph.py                      ← LangGraph agent definition
  conversation.py               ← in-memory session management
  server.py                     ← FastAPI app, routes, SSE
  cli.py                        ← `precice-ai` entry point

static/
  index.html                    ← entire frontend (one file)

pyproject.toml
.env.example
README.md
CLAUDE.md                       ← this file
```

---

## 5. config.py — settings module

```python
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
MAX_SESSION_MESSAGES: int = 20   # trim older messages beyond this
```

---

## 6. ingest.py — data ingestion pipeline

Scrape these sources and embed into ChromaDB:

### 6a. Documentation pages to scrape (fetch all of these)

```python
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
```

### 6b. Forum ingestion

- Hit `https://precice.discourse.group/latest.json` — fetch top 50 topics
- For each topic fetch `https://precice.discourse.group/t/{id}.json` — take first 5 posts
- Strip HTML with BeautifulSoup, concatenate post text
- Add `time.sleep(0.3)` between topic requests to be polite
- Store metadata: `{"source": url, "title": title, "type": "forum"}`

### 6c. Chunking and embedding

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=120,
    separators=["\n\n", "\n", ". ", " ", ""],
)
```

Use `hashlib.md5(f"{source}::{chunk_index}".encode()).hexdigest()` as the Chroma
document ID so re-ingestion is idempotent (upsert, not duplicate).

### 6d. Ingestion status dict

Maintain a module-level dict:

```python
status = {
    "state": "starting",   # "starting" | "ingesting" | "ready" | "error"
    "message": "",
    "chunk_count": 0,
    "source_count": 0,
    "last_updated": None,  # ISO string
}
```

Expose this dict to `server.py` for the `/api/status` endpoint.

---

## 7. vectorstore.py — ChromaDB singleton

```python
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

_vectorstore = None

def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
        _vectorstore = Chroma(
            collection_name="precice_docs",
            embedding_function=embeddings,
        )
    return _vectorstore
```

---

## 8. tools.py — the agent's tools

All tools are decorated with `@tool` from `langchain_core.tools`.
Every tool docstring is the description the LLM sees — write them clearly.

### Tool 1: search_precice_docs

```python
@tool
def search_precice_docs(query: str) -> str:
    """
    Search the preCICE documentation, FAQ, and forum for information.
    Use this for any question about preCICE concepts, configuration, API,
    installation, tutorials, or troubleshooting.
    Returns relevant text chunks with source URLs.
    """
```

Implementation: MMR retrieval from ChromaDB with k=6, fetch_k=20, lambda_mult=0.6.
Return formatted string: each chunk as `[Source: {url}]\n{content}\n---`.

### Tool 2: list_project_files

```python
@tool
def list_project_files(working_dir: str, extension_filter: str = "") -> str:
    """
    List all files in the user's simulation project directory.
    working_dir: absolute path to the project folder.
    extension_filter: optional file extension to filter by, e.g. ".xml" or ".py".
    Returns a tree-style listing of files with sizes.
    """
```

Implementation: Walk the directory with `os.walk`. Skip hidden dirs (`.git`, `__pycache__`).
SECURITY: Resolve paths and verify they are inside `working_dir` before returning.
Return formatted file tree as string.

### Tool 3: read_project_file

```python
@tool
def read_project_file(file_path: str, working_dir: str) -> str:
    """
    Read the contents of a file in the user's project directory.
    Use this to read precice-config.xml, solver scripts, log files,
    CMakeLists.txt, or any other project file the user asks about.
    file_path: path relative to working_dir, or absolute path inside working_dir.
    working_dir: absolute path to the project folder (provided by session context).
    """
```

Implementation:

- Resolve to absolute path
- SECURITY CHECK: `assert resolved.is_relative_to(Path(working_dir))`
- Read and return file content as string
- If file > 50KB, return first 50KB with a note that it was truncated
- Handle binary files gracefully (return error message, don't crash)

### Tool 4: write_project_file

```python
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
```

Implementation:

- SECURITY CHECK: resolved path must be inside `working_dir`
- Create parent directories if needed (`parents=True, exist_ok=True`)
- Write content as UTF-8
- Return `f"Written {len(content)} chars to {abs_path}"`

### Tool 5: validate_precice_config

```python
@tool
def validate_precice_config(config_path: str, working_dir: str) -> str:
    """
    Validate a preCICE XML configuration file using the preCICE config checker.
    Runs `precice-tools check <config_path>` and returns any errors or warnings.
    Use this when the user has a config file and wants to check it for errors.
    config_path: path to the precice-config.xml file, relative to working_dir.
    working_dir: absolute path to the project folder.
    """
```

Implementation:

- SECURITY CHECK: resolved path must be inside `working_dir`
- Run: `subprocess.run(["precice-tools", "check", str(abs_path)], capture_output=True, text=True, timeout=30)`
- If `precice-tools` not found: return helpful message explaining it needs to be installed
- Return stdout + stderr, or "Config is valid." if both are empty and returncode==0

### Tool 6: search_forum_live

```python
@tool
def search_forum_live(query: str) -> str:
    """
    Search the preCICE Discourse forum in real-time for recent discussions.
    Use this for very recent questions, bugs, or topics not yet in the local index.
    query: search terms to look for on the forum.
    Returns up to 5 relevant topic titles, summaries, and URLs.
    """
```

Implementation: `GET https://precice.discourse.group/search.json?q={urllib.parse.quote(query)}`
Parse response JSON: `data["topics"][:5]` — return title + excerpt + URL for each.

### Tool 7: read_attached_file (for user file attachments)

```python
@tool
def read_attached_file(filename: str, session_id: str) -> str:
    """
    Read the content of a file attached by the user in the chat interface.
    Use this when the user says they have attached a file or asks about an uploaded file.
    filename: the name of the file as uploaded.
    session_id: the current session ID.
    Returns the file content as text.
    """
```

Implementation: Read from `ATTACHMENT_STORE[session_id][filename]` (an in-memory dict
populated by the `/api/upload` endpoint). Return content or error if not found.

### Tool registry — pass to graph

```python
ALL_TOOLS = [
    search_precice_docs,
    list_project_files,
    read_project_file,
    write_project_file,
    validate_precice_config,
    search_forum_live,
    read_attached_file,
]
```

---

## 9. graph.py — LangGraph ReAct agent

### 9a. State definition

```python
from typing import TypedDict, Annotated, List
import operator
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    working_dir: str       # absolute path set per-session
    session_id: str        # for attachment lookup
```

### 9b. Node: call_llm

```python
def call_llm(state: AgentState) -> dict:
    system = SystemMessage(content=f"""You are preCICE AI, an expert assistant for the
preCICE multiphysics coupling library. You help with configuration, coupling schemes,
installation, adapters, tutorials, and debugging.

The user's simulation project is at: {state['working_dir'] or 'no folder selected'}
Session ID: {state['session_id']}

Use your tools to look up documentation, read project files, or validate configs.
Always cite which source (doc page or forum post) your answer comes from.
When writing or modifying files, confirm with the user what you are about to do first.
""")
    # Prepend system message, keeping last MAX_SESSION_MESSAGES messages
    messages = [system] + state["messages"][-20:]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}
```

### 9c. Node: call_tools

Use `ToolNode` from LangGraph:

```python
from langgraph.prebuilt import ToolNode
tool_node = ToolNode(ALL_TOOLS)
```

### 9d. Conditional edge

```python
from langgraph.prebuilt import tools_condition

# tools_condition returns "tools" if last message has tool_calls, else END
```

### 9e. Graph assembly

```python
from langgraph.graph import StateGraph, END

def build_graph() -> CompiledGraph:
    graph = StateGraph(AgentState)
    graph.add_node("agent", call_llm)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")   # loop back after tool execution
    return graph.compile()
```

### 9f. Streaming helper

```python
async def stream_agent(
    app: CompiledGraph,
    messages: list,
    working_dir: str,
    session_id: str,
) -> AsyncIterator[dict]:
    """
    Yields dicts:
      {"type": "token",      "content": str}
      {"type": "tool_start", "tool": str, "input": dict}
      {"type": "tool_end",   "tool": str, "output": str}
      {"type": "sources",    "content": list[dict]}
      {"type": "done"}
    """
    state = AgentState(
        messages=messages,
        working_dir=working_dir,
        session_id=session_id,
    )
    sources_seen = []

    async for event in app.astream_events(state, version="v2"):
        kind = event["event"]

        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if chunk.content:
                yield {"type": "token", "content": chunk.content}

        elif kind == "on_tool_start":
            yield {
                "type": "tool_start",
                "tool": event["name"],
                "input": event["data"].get("input", {}),
            }

        elif kind == "on_tool_end":
            output = event["data"].get("output", "")
            yield {
                "type": "tool_end",
                "tool": event["name"],
                "output": str(output)[:500],  # truncate for UI
            }
            # Extract sources from search_precice_docs results
            if event["name"] == "search_precice_docs":
                # parse [Source: url] markers from output
                import re
                for match in re.finditer(r'\[Source: (https?://[^\]]+)\]', str(output)):
                    url = match.group(1)
                    if url not in [s["url"] for s in sources_seen]:
                        sources_seen.append({"url": url, "type": "docs"})

    if sources_seen:
        yield {"type": "sources", "content": sources_seen}
    yield {"type": "done"}
```

---

## 10. conversation.py — session management

```python
import uuid
from typing import Dict, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# In-memory session store — cleared on restart
SESSIONS: Dict[str, Dict] = {}
# {session_id: {"messages": [...], "working_dir": str}}

# In-memory attachment store — cleared on restart
ATTACHMENT_STORE: Dict[str, Dict[str, str]] = {}
# {session_id: {filename: content_string}}

def create_session() -> str:
    sid = str(uuid.uuid4())
    SESSIONS[sid] = {"messages": [], "working_dir": ""}
    ATTACHMENT_STORE[sid] = {}
    return sid

def get_session(sid: str) -> dict | None:
    return SESSIONS.get(sid)

def append_human(sid: str, text: str):
    SESSIONS[sid]["messages"].append(HumanMessage(content=text))
    # trim to last MAX_SESSION_MESSAGES
    if len(SESSIONS[sid]["messages"]) > 20:
        SESSIONS[sid]["messages"] = SESSIONS[sid]["messages"][-20:]

def append_ai(sid: str, text: str):
    SESSIONS[sid]["messages"].append(AIMessage(content=text))

def set_working_dir(sid: str, path: str):
    SESSIONS[sid]["working_dir"] = path

def store_attachment(sid: str, filename: str, content: str):
    ATTACHMENT_STORE[sid][filename] = content

def delete_session(sid: str):
    SESSIONS.pop(sid, None)
    ATTACHMENT_STORE.pop(sid, None)
```

---

## 11. server.py — FastAPI application

### 11a. Lifespan

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Load vectorstore (singleton)
    vs = get_vectorstore()

    # 2. Build LangGraph app
    global rag_app
    rag_app = build_graph()

    # 3. Run initial ingestion in background thread
    threading.Thread(target=run_ingestion, daemon=True).start()

    # 4. Schedule hourly re-ingestion
    scheduler = BackgroundScheduler()
    scheduler.add_job(run_ingestion, "interval", hours=1)
    scheduler.start()

    yield
    scheduler.shutdown()
```

### 11b. Endpoints to implement

| Method   | Path                         | Description                                      |
| -------- | ---------------------------- | ------------------------------------------------ |
| `GET`    | `/`                          | Serve `static/index.html`                        |
| `GET`    | `/api/status`                | Return ingestion status dict                     |
| `POST`   | `/api/session`               | Create session → `{"session_id": "..."}`         |
| `DELETE` | `/api/session/{sid}`         | Clear session                                    |
| `POST`   | `/api/session/{sid}/workdir` | Set working directory, body: `{"path": "..."}`   |
| `POST`   | `/api/upload/{sid}`          | Accept multipart file, store in ATTACHMENT_STORE |
| `POST`   | `/api/chat`                  | SSE stream chat response                         |
| `POST`   | `/api/reingest`              | Trigger manual re-ingestion                      |

### 11c. /api/chat endpoint details

```python
class ChatRequest(BaseModel):
    message: str
    session_id: str

@app.post("/api/chat")
async def chat(req: ChatRequest):
    session = get_session(req.session_id)
    if not session:
        return JSONResponse({"error": "session not found"}, 404)

    append_human(req.session_id, req.message)

    async def event_stream():
        full_answer = ""
        async for event in stream_agent(
            rag_app,
            session["messages"],
            session["working_dir"],
            req.session_id,
        ):
            yield f"data: {json.dumps(event)}\n\n"
            if event["type"] == "token":
                full_answer += event["content"]

        if full_answer:
            append_ai(req.session_id, full_answer)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

### 11d. /api/upload endpoint details

```python
@app.post("/api/upload/{session_id}")
async def upload_file(session_id: str, file: UploadFile = File(...)):
    content = await file.read()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return JSONResponse({"error": "Binary files not supported"}, 400)

    store_attachment(session_id, file.filename, text)
    return {"ok": True, "filename": file.filename, "size": len(text)}
```

---

## 12. static/index.html — full frontend

Build a single HTML file with embedded CSS and JS. Dark theme matching the preCICE
aesthetic. No external JS dependencies except Google Fonts.

### 12a. Layout

Three-panel layout:

```
┌─────────────────────────────────────────────────────┐
│  Header: logo | status pill | model badge           │
├──────────────┬──────────────────────────┬───────────┤
│   Sidebar    │      Chat messages       │  (hidden  │
│  (220px)     │      (flex-grow)         │   on      │
│              │                          │   mobile) │
│  • Working   │                          │           │
│    directory │                          │           │
│    picker    │                          │           │
│              │                          │           │
│  • Attached  │                          │           │
│    files     │                          │           │
│              │                          │           │
│  • KB stats  │                          │           │
│              │                          │           │
│  • Links     │                          │           │
├──────────────┴──────────────────────────┴───────────┤
│  Input bar: textarea | attach button | send button  │
└─────────────────────────────────────────────────────┘
```

### 12b. Working directory picker

- Text input where user pastes an absolute path
- "Set folder" button calls `POST /api/session/{sid}/workdir`
- Show current path in green when set, greyed out when empty
- On session create/clear, reset to empty

### 12c. File attachment

- Paperclip button next to textarea opens `<input type="file">` (accept="\*")
- Upload via `POST /api/upload/{sid}` with FormData
- Show attached files as pills above the textarea with × to remove
- When sending message, list attached filenames in the message so the LLM knows

### 12d. Tool call display (IMPORTANT)

When `type == "tool_start"` event arrives:

- Render a collapsible card ABOVE the streaming text:
  ```
  ┌─────────────────────────────────────────┐
  │ ⚙ calling read_project_file        ▼   │
  │ { "file_path": "precice-config.xml" }  │ ← collapsed by default
  └─────────────────────────────────────────┘
  ```
  When `type == "tool_end"` arrives:
- Update the card to show result (first 200 chars, expandable)
- Mark as complete with a ✓ icon

Tool display must appear inline in the message, between any preceding text
and the text that follows the tool result.

### 12e. Source citations

When `type == "sources"` arrives:

- Render pills at the bottom of the AI message:
  ```
  Sources:  📄 configuration-mapping  💬 forum: mapping-rbf-question
  ```
- Each pill is a clickable link

### 12f. SSE event handling skeleton

```javascript
async function sendMessage() {
  const resp = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, session_id: sessionId }),
  });

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let currentAiMsgEl = createAiMessageElement();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const event = JSON.parse(line.slice(6));

      switch (event.type) {
        case "token":
          appendToken(currentAiMsgEl, event.content);
          break;
        case "tool_start":
          appendToolCard(currentAiMsgEl, event.tool, event.input);
          break;
        case "tool_end":
          updateToolCard(currentAiMsgEl, event.tool, event.output);
          break;
        case "sources":
          appendSources(currentAiMsgEl, event.content);
          break;
        case "done":
          finalize(currentAiMsgEl);
          break;
      }
    }
  }
}
```

---

## 13. pyproject.toml

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "precice-ai"
version = "0.1.0"
description = "Agentic local assistant for preCICE multiphysics simulations"
requires-python = ">=3.10"

dependencies = [
    "langchain>=0.3.0",
    "langchain-core>=0.3.0",
    "langchain-community>=0.3.0",
    "langchain-openai>=0.2.0",
    "langgraph>=0.2.0",
    "chromadb>=0.5.0",
    "sentence-transformers>=3.0.0",
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "apscheduler>=3.10.0",
    "requests>=2.32.0",
    "beautifulsoup4>=4.12.0",
    "python-multipart>=0.0.9",
    "pydantic>=2.0.0",
]

[project.scripts]
precice-ai = "precice_ai.cli:main"

[tool.setuptools.packages.find]
where = ["."]
include = ["precice_ai*"]

[tool.setuptools.package-data]
precice_ai = ["../static/*"]
```

---

## 14. cli.py — entry point

```python
def main():
    import argparse, os, sys, time, threading, webbrowser
    import uvicorn

    parser = argparse.ArgumentParser(description="preCICE AI local assistant")
    parser.add_argument("--host", default=os.environ.get("PRECICE_AI_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PRECICE_AI_PORT", "7860")))
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY not set.")
        print("Get a free key at https://openrouter.ai/ then:")
        print("  export OPENROUTER_API_KEY=sk-or-...")
        sys.exit(1)

    if args.model:
        os.environ["PRECICE_AI_MODEL"] = args.model

    if not args.no_browser:
        def _open():
            time.sleep(2.5)
            webbrowser.open(f"http://{args.host}:{args.port}")
        threading.Thread(target=_open, daemon=True).start()

    print(f"preCICE AI starting at http://{args.host}:{args.port}")
    uvicorn.run("precice_ai.server:app", host=args.host, port=args.port, reload=False)
```

---

## 15. Build order — follow this sequence exactly

Build and verify each step before moving to the next. Do not skip ahead.

1. **`pyproject.toml` + `config.py`** — get `pip install -e .` working
2. **`vectorstore.py`** — verify embeddings load on CPU
3. **`ingest.py`** — run ingestion standalone, verify chunks appear in ChromaDB
4. **`tools.py`** — test each tool function individually in a Python REPL
5. **`graph.py`** — test the graph with a simple text question (no file tools)
6. **`conversation.py`** — verify session create/append/trim works
7. **`server.py`** — start the server, test `/api/status` and `/api/session`
8. **`static/index.html`** — build the UI, test end-to-end chat
9. **`cli.py`** — wrap everything, test `precice-ai` command

At each step, run a quick smoke test before proceeding.

---

## 16. Security rules — NEVER violate these

1. All file read/write operations MUST verify the resolved path is inside `working_dir`
2. Use `Path(path).resolve().is_relative_to(Path(working_dir).resolve())` for the check
3. Never allow `..` path traversal
4. Never execute arbitrary shell commands — only `precice-tools check <config_path>`
5. `working_dir` can only be set by the user explicitly via `/api/session/{sid}/workdir`
6. Attachments are stored in memory only — never written to disk by the server

---

## 17. Error handling rules

- Every tool must return a string — never raise exceptions to the agent
- Wrap tool bodies in try/except and return `f"Error: {e}"` on failure
- Ingest failures log a warning but do not crash the server
- If `OPENROUTER_API_KEY` is missing, `cli.py` exits with a clear message before starting
- If Chroma is empty when a query arrives, return a message explaining ingestion is still running

---

## 18. What NOT to do

- Do not add a database (no SQLite, PostgreSQL, Redis)
- Do not add authentication or user accounts
- Do not store any chat history to disk
- Do not use React, Vue, or any JS framework — vanilla JS only
- Do not use Docker — pure `pip install` setup
- Do not hardcode the API key anywhere — always read from environment
- Do not write files outside the user's selected working directory
- Do not allow the agent to run arbitrary shell commands

---

## 19. .env.example

```
# Copy to .env and fill in your key
# Get a free API key at https://openrouter.ai/

OPENROUTER_API_KEY=sk-or-your-key-here

# Optional: change the model (must support tool calling)
# PRECICE_AI_MODEL=mistralai/mistral-7b-instruct:free
# PRECICE_AI_MODEL=meta-llama/llama-3.1-8b-instruct:free
# PRECICE_AI_MODEL=google/gemma-3-12b-it:free

# Optional: change host/port
# PRECICE_AI_HOST=127.0.0.1
# PRECICE_AI_PORT=7860
```

---

## 20. Definition of done

The project is complete when:

- [ ] `pip install -e .` runs without errors on Python 3.10+
- [ ] `precice-ai` starts the server and opens the browser
- [ ] On startup, ingestion runs and `/api/status` shows `"state": "ready"`
- [ ] User can type a preCICE question and get a streamed answer with source citations
- [ ] User can select a folder and ask "what files are in this project?"
- [ ] User can ask "read my precice-config.xml and explain the coupling scheme"
- [ ] User can ask "write a minimal precice-config.xml for a two-participant simulation"
- [ ] User can attach a log file and ask "what errors are in this log?"
- [ ] Tool call cards appear inline in the chat during agent execution
- [ ] Clearing the session wipes message history (verified by fresh question having no prior context)
- [ ] Server runs for 2+ hours without crashing (APScheduler re-ingests cleanly)
