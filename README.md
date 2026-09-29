# preCICE AI

A local, agentic assistant for the [preCICE](https://precice.org) multiphysics coupling
library, built with **LangGraph** and backed by the **`precice-ai` MCP server**. It runs
entirely on your machine, opens a browser chat UI, and lets an LLM agent:

- answer preCICE questions from the MCP server's **local knowledge-base cache** (docs,
  tutorials, forum, GitHub issues/PRs — embedded per category),
- read, write, and validate files inside a project folder you pick per chat session,
- inspect preCICE projects, configs, and logs, and wrap `precice-cli` through the MCP
  server's tools,
- search the preCICE Discourse forum live.

The intended user flow:

1. clone this repo **and** the sibling [`precice-ai`](https://github.com/vaibhavd2103/precice-ai) MCP server,
2. install both,
3. configure the chat LLM through `.env` or CLI flags,
4. start the local server with `precice-ai`,
5. choose a working directory for each chat session in the UI,
6. chat — preCICE knowledge comes from the MCP KB, project file access stays inside the
   chosen directory.

---

## Contents

- [Quick Start — install and run](#quick-start--install-and-run)
- [1. Architecture](#1-architecture)
- [2. The preCICE MCP server dependency](#2-the-precice-mcp-server-dependency)
- [3. The knowledge base (local cache)](#3-the-knowledge-base-local-cache)
- [4. The ReAct loop (`graph.py`)](#4-the-react-loop-precice_aigraphpy)
- [5. Tools](#5-tools)
- [6. Request lifecycle](#6-request-lifecycle)
- [7. HTTP API](#7-http-api)
- [8. Logging](#8-activity-logging)
- [9. File map](#9-file-map)
- [10. Extending](#10-extending)
- [Known limitations](#known-limitations)

---

## Quick Start — install and run

### Overview

This project is made of **two repos that are checked out side by side**:

| Repo | Role |
|---|---|
| [`precice-ai-lang`](https://github.com/vaibhavd2103/precice-ai-lang) (this repo) | Browser chat UI, FastAPI server, LangGraph agent, local sandboxed file tools |
| [`precice-ai`](https://github.com/vaibhavd2103/precice-ai) | The preCICE **MCP server** — knowledge base + KB cache, project/config/log inspection, `precice-cli` wrappers |

At startup `precice-ai-lang` launches the sibling server as a subprocess
(`python -m precice_ai.server`, MCP over stdio) and binds all of its tools to the agent.
By default it expects the sibling at `../precice-ai`:

```text
some-parent-folder/
  precice-ai-lang/   ← this repo
  precice-ai/        ← the MCP server, cloned next to it
```

If the sibling lives elsewhere, set `PRECICE_AI_MCP_SERVER` to its `server.py` path.

> **Both repos use the same distribution name (`precice-ai`), Python package
> (`precice_ai`), and console script (`precice-ai`).** Install each into **its own
> virtualenv**. If you install both into one venv, whichever was installed last owns the
> `precice-ai` command. The MCP subprocess still resolves the sibling's package correctly
> because it is started with `cwd` set to the sibling repo.

You need an API key for an OpenAI-compatible, tool-calling chat provider. **OpenRouter**
(`https://openrouter.ai`) is the default. The MCP server also needs an **embedding** key
(OpenRouter or Blablador) to embed your questions for KB search.

### Prerequisites (all platforms)

- Python 3.10+ and `git`
- Optional: `tkinter` for the **native folder-picker button** (`GET /api/pick-directory`).
  You can always paste an absolute path into the text field instead. `tkinter` is a
  system package, not a PyPI one — don't add it to `pyproject.toml`:
  - **Debian/Ubuntu**: `sudo apt install python3-tk`
  - **Fedora/RHEL**: `sudo dnf install python3-tkinter`
  - **Arch**: `sudo pacman -S tk`
  - **macOS (Homebrew Python)**: `brew install python-tk`; the python.org installer
    already bundles Tcl/Tk
  - **Windows**: bundled with the python.org installer ("tcl/tk and IDLE"); the Microsoft
    Store build omits it
  - **WSL**: not needed — the picker opens a Windows folder dialog through
    `powershell.exe` and converts the result with `wslpath`.
  - The picker imports `tkinter` lazily on each click, so installing it takes effect
    without a restart.
- Optional: preCICE itself (provides `precice-tools` / `precice-cli`) for config
  validation, `precice_init`, and profiling tools. Without it those tools return an
  install hint instead of failing.

### 1. Clone both repos next to each other

**Linux / macOS / WSL**

```bash
mkdir -p ~/precice && cd ~/precice
git clone https://github.com/vaibhavd2103/precice-ai-lang.git
git clone https://github.com/vaibhavd2103/precice-ai.git
```

**Windows (PowerShell)**

```powershell
mkdir $HOME\precice; cd $HOME\precice
git clone https://github.com/vaibhavd2103/precice-ai-lang.git
git clone https://github.com/vaibhavd2103/precice-ai.git
```

### 2. Set up the MCP server (`precice-ai`)

```bash
cd precice-ai
python3 -m venv .venv          # Windows: python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install -e .
cp .env.example .env           # Windows: copy .env.example .env
```

Add your **embedding** key to `precice-ai/.env`:

```text
OPENROUTER_API_KEY=sk-or-...
```

Warm the local KB cache (optional — the MCP server also syncs it on every start, but
doing it now avoids a delay on your first question):

```bash
precice-ai kb ingest     # downloads the category-wise KB assets into ~/.precice-ai/kb_store
precice-ai kb status
deactivate
```

See that repo's `README.md` for Blablador embeddings, other MCP clients, and the full tool
reference.

### 3. Set up the chat UI (`precice-ai-lang`, this repo)

```bash
cd ../precice-ai-lang
python3 -m venv .venv          # Windows: python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install -e .
cp .env.example .env           # Windows: copy .env.example .env
```

Set the **chat** model in `.env`:

```text
PRECICE_AI_PROVIDER=openrouter
PRECICE_AI_API_KEY=sk-or-...
PRECICE_AI_MODEL=openai/gpt-4o-mini
PRECICE_AI_BASE_URL=https://openrouter.ai/api/v1
```

> `pip install -e .` here also installs the MCP server's runtime dependencies (`mcp`,
> `fastmcp`, `httpx`, `lxml`, `numpy`, `openai`), so this venv's Python can launch the
> sibling server directly. If you'd rather use the sibling's own venv, set
> `PRECICE_AI_MCP_PYTHON=../precice-ai/.venv/bin/python`.
>
> It also pulls in `chromadb` and `sentence-transformers` (and PyTorch — several GB).
> These are left over from the earlier local-RAG design (`vectorstore.py`) and no code
> path imports them today.

### 4. Run it

```bash
precice-ai            # from inside precice-ai-lang, with its venv active
```

This opens `http://127.0.0.1:7860`. On startup you should see something like:

```text
preCICE AI starting at http://127.0.0.1:7860
[preCICE AI] {"event": "mcp_startup", "connected": true, "tools": [...27 tool names...], ...}
Using sibling preCICE MCP server for preCICE knowledge questions.
INFO:     Application startup complete.
```

`"connected": true` means the MCP handoff worked. If you see `"connected": false` with an
`error`, see [Troubleshooting](#troubleshooting).

In the browser:

1. A session is created automatically (**New chat** starts a fresh one).
2. Pick a **working directory** — the absolute path to your preCICE project. The server
   rejects chat turns until one is set, and every local file tool is sandboxed to it.
3. Optionally attach text files (logs, configs) with the paperclip.
4. Chat — e.g. *"explain the coupling scheme in my precice-config.xml"* or *"what does
   `<acceleration:IQN-ILS>` do?"*.

CLI flags override `.env`:

```bash
precice-ai --provider openrouter --api-key sk-or-... --model openai/gpt-4o-mini \
           --base-url https://openrouter.ai/api/v1 --port 7861 --no-browser --env-file .env
```

### Environment variables

**Read by this app (`precice-ai-lang`)**

| Variable | Purpose | Default |
|---|---|---|
| `PRECICE_AI_PROVIDER` / `LLM_PROVIDER` | Chat LLM provider name (`openrouter` adds OpenRouter headers and default base URL) | `openrouter` |
| `PRECICE_AI_API_KEY` / `LLM_API_KEY` / `OPENROUTER_API_KEY` | Chat model API key | *(required)* |
| `PRECICE_AI_MODEL` / `LLM_MODEL` | Chat model id (must support tool calling) | `openai/gpt-4o-mini` |
| `PRECICE_AI_BASE_URL` / `LLM_BASE_URL` | OpenAI-compatible base URL | `https://openrouter.ai/api/v1` when provider is `openrouter` |
| `PRECICE_AI_MAX_TOKENS` | Max tokens per assistant reply | `1024` |
| `PRECICE_AI_HOST` / `PRECICE_AI_PORT` | Bind address | `127.0.0.1` / `7860` |
| `PRECICE_AI_MCP_SERVER` | Path to the sibling MCP server's `server.py` (its parent dir becomes the subprocess `cwd`) | `../precice-ai/server.py` |
| `PRECICE_AI_MCP_PYTHON` | Python interpreter used to launch the MCP subprocess | this app's interpreter |
| `PRECICE_KB_STORE_DIR` | KB cache directory (read here for `/api/status`, and inherited by the MCP server) | `~/.precice-ai/kb_store` |
| `PRECICE_AI_ALLOW_LOCAL_SEARCH_FALLBACK` | Expose `search_precice_docs` (lexical search over the cached KB) when MCP is **not** connected | `false` |
| `PRECICE_AI_LOG_FILE` | JSONL activity log path | `./logs/agent.jsonl` |

**Passed through to the MCP server subprocess** (it inherits this app's environment)

| Variable | Purpose | Default |
|---|---|---|
| `OPENROUTER_API_KEY` / `BLABLADOR_API_KEY` | Embedding key for KB queries | falls back to `PRECICE_AI_API_KEY` if unset |
| `EMBEDDING_BASE_URL` / `EMBEDDING_MODEL` | Embedding endpoint/model (must match what the KB was built with) | `https://openrouter.ai/api/v1` / `openai/text-embedding-3-small` |
| `PRECICE_PROJECTS_DIR` | Root folder for the MCP **project/config/log** tools (`project_name` is resolved under it) | `<sibling repo>/test-projects` |
| `PRECICE_AI_SKIP_KB_BOOTSTRAP` | Skip the KB sync the MCP server does on start | unset |

The sibling server also loads its own `.env` (`precice-ai/.env`) without overriding
values that are already set.

### Troubleshooting

- **`ERROR: no LLM API key found.`** — no `PRECICE_AI_API_KEY` / `LLM_API_KEY` /
  `OPENROUTER_API_KEY`. The `.env.example` placeholder counts as set, so with it the
  server starts but the first chat fails with a provider auth error.
- **`mcp_startup` with `"connected": false`, or `tools.mcp_connected: false` in
  `GET /api/status`** — check `tools.mcp_error`. Typical causes:
  - `MCP server not found at …` — the sibling isn't at `../precice-ai`; set
    `PRECICE_AI_MCP_SERVER`.
  - `missing Python packages …` / `MCP server failed to start …` — the interpreter can't
    import the sibling's dependencies; re-run `pip install -e .` here, or set
    `PRECICE_AI_MCP_PYTHON` to the sibling's `.venv/bin/python`
    (`.venv\Scripts\python.exe` on Windows).
  - `Timed out after 20 seconds while loading MCP tools` — usually a slow first-run KB
    sync; run `precice-ai kb ingest` in the sibling once, then restart.
- **Agent says preCICE knowledge is unavailable / answers have no KB context** — confirm
  `mcp_connected: true`, then check `GET /api/status` → `state` (should be `ready`) and
  `precice-ai kb status` in the sibling repo.
- **KB search errors about embeddings / 401** — the MCP server has no valid embedding key.
  If your chat provider is **not** OpenRouter, its key is forwarded as `OPENROUTER_API_KEY`
  unless you set one explicitly — set `OPENROUTER_API_KEY` (or `BLABLADOR_API_KEY`) in
  `precice-ai/.env`.
- **MCP project/log tools say the project doesn't exist** — those tools resolve
  `project_name` under `PRECICE_PROJECTS_DIR`, not the session working directory. Set
  `PRECICE_PROJECTS_DIR` to the folder that contains your projects (see
  [Known limitations](#known-limitations)).
- **Folder picker alert `Directory picker unavailable: No module named 'tkinter'`** —
  install the system Tcl/Tk package (see Prerequisites) or paste the path manually.
- **`402` mentioning `max_tokens` / "can only afford N"** — lower `PRECICE_AI_MAX_TOKENS`
  (e.g. `512`) or add credits.
- **`402` `Prompt tokens limit exceeded: X > Y` even for "hi"** — every LLM call binds
  ~30 tool schemas (6 local + 27 MCP) plus the system prompt, which is several thousand
  input tokens. Add credits or switch to a $0 model such as
  `PRECICE_AI_MODEL=openrouter/free` (free models are weaker and rate-limited).
- **Port already in use** — `precice-ai --port 7861`.
- **`validate_precice_config` / `precice_config_check` return an install hint** —
  `precice-tools` / `precice-cli` isn't on `PATH`; install preCICE.

---

## 1. Architecture

```
┌──────────────────────── Browser (static/index.html) ─────────────────────────┐
│  session · working-dir picker · attachments · chat · tool cards · KB status  │
└───────────────┬──────────────────────────────────────────────▲───────────────┘
                │ REST (session/workdir/upload)  POST /api/chat│ SSE events
                ▼                                              │
┌─────────────────────────── precice-ai-lang (this repo) ──────┴───────────────┐
│ server.py  FastAPI · lifespan: initialize_tools() → run_ingestion() →        │
│            build_graph() → APScheduler (hourly KB status refresh)            │
│ conversation.py  in-memory SESSIONS + ATTACHMENT_STORE (no disk, no DB)      │
│                                                                              │
│ graph.py  LangGraph StateGraph                                               │
│   ┌──────────────────────────────┐  tool_calls  ┌──────────────────────┐     │
│   │ agent (call_llm)             │─────────────▶│ tools (ToolNode)     │     │
│   │  1. preCICE question?  ──┐   │◀─────────────│  local + MCP tools   │     │
│   │  2. MCP prefetch  ◀──────┘   │              └──────────┬───────────┘     │
│   │  3. system prompt + history  │── no calls ──▶ END      │                 │
│   │  4. llm.bind_tools().invoke  │                         │                 │
│   └──────────────┬───────────────┘                         │                 │
│                  │ llm.py → ChatOpenAI (OpenRouter / any OpenAI-compatible)  │
│                                                            │                 │
│ tools.py ── local tools (sandboxed to session working_dir) ┤                 │
│          └─ MultiServerMCPClient ─────── stdio ────────────┼──────┐          │
│ ingest.py / mcp_kb.py ── read KB cache status (no scraping)│      │          │
└────────────────────────────────────────────────────────────┼──────┼──────────┘
                                                             │      ▼
┌──────────────────── precice-ai MCP server (sibling repo, subprocess) ────────┐
│ kb_* tools · project/config/log tools · precice-cli wrappers                 │
│ on start: sync KB from GitHub Release "kb-latest" (≤ every 96 h)             │
└───────────────┬────────────────────────────────────────┬─────────────────────┘
                ▼                                        ▼
   ~/.precice-ai/kb_store/                     embedding API (OpenRouter /
     kb-embeddings-<category>.npz               Blablador) for query vectors
     knowledge_base.json (lexical)
```

Key design points:

- **The MCP server is the source of truth for preCICE knowledge.** This app no longer
  scrapes or embeds anything. It launches the MCP server, binds its tools, and reads the
  KB cache on disk only to report status.
- **Two sandboxes.** Local file tools are confined to the session's `working_dir`. MCP
  project tools are confined to `PRECICE_PROJECTS_DIR`. These are different roots — see
  [Known limitations](#known-limitations).
- **Graceful degradation.** If the MCP server can't start, the app still runs with its
  local tools. The system prompt tells the model MCP is down so it doesn't claim to have
  searched the KB. An opt-in lexical fallback (`search_precice_docs`) can be enabled.
- **Local-first and stateless.** Sessions and attachments live in memory and are lost on
  restart. The only on-disk state is the MCP KB cache and the JSONL activity log.

## 2. The preCICE MCP server dependency

`precice_ai/tools.py::initialize_tools()` runs once in the FastAPI lifespan:

1. Resolves the server path (`PRECICE_AI_MCP_SERVER` or `../precice-ai/server.py`) and the
   interpreter (`PRECICE_AI_MCP_PYTHON` or `sys.executable`).
2. **Preflight.** When using this app's own interpreter, checks that `dotenv`, `mcp`,
   `fastmcp`, `httpx`, `lxml`, `numpy`, and `openai` are importable. It then runs
   `server.py` for up to 5 s: a timeout means the server started and is waiting on stdio
   (good), and a non-zero exit is reported as the startup error.
3. Builds the child environment from this process's env. If `OPENROUTER_API_KEY` isn't
   set, it is filled from the chat API key. `EMBEDDING_BASE_URL` / `EMBEDDING_MODEL` get
   OpenRouter defaults.
4. Connects with `langchain_mcp_adapters.MultiServerMCPClient` over **stdio**
   (`python -m precice_ai.server`, `cwd` = sibling repo) and loads the tools with a 20 s
   timeout.
5. Records the result in a status dict (exposed at `/api/tools` and `/api/status` →
   `tools`) and logs an `mcp_startup` event.

The MCP tools are ordinary LangChain tools after loading, so they're bound to the model
and executed by the same `ToolNode` as the local tools.

MCP tools exposed by the sibling server (27):

| Group | Tools |
|---|---|
| Knowledge base | `kb_precice_status`, `kb_ingest_precice_data`, `kb_query_precice`, `kb_query_precice_live`, `kb_query_precice_lexical` |
| Projects | `list_precice_projects`, `inspect_project_structure`, `find_precice_config`, `run_command_in_project` (allow-listed commands only) |
| Config | `inspect_precice_config`, `summarize_precice_config`, `backup_precice_config` |
| Logs | `list_project_logs`, `read_project_logs`, `read_latest_log`, `analyze_precice_logs` |
| `precice-cli` | `precice_version`, `precice_config_check`, `precice_config_visualize`, `precice_config_format`, `precice_config_doc`, `precice_init`, `precice_profiling_analyze` / `_trace` / `_export` / `_histogram` / `_merge` |

See the `precice-ai` repo for each tool's parameters.

## 3. The knowledge base (local cache)

The KB is built **outside** both repos by a scheduled GitHub Action in `precice-ai`. It
runs every 4 days and publishes category-wise assets to the `kb-latest` release. The MCP
server mirrors those assets into a local cache:

```
~/.precice-ai/kb_store/            (override with PRECICE_KB_STORE_DIR)
  kb-embeddings-about.npz
  kb-embeddings-community.npz
  kb-embeddings-documentation.npz
  kb-embeddings-tutorials.npz
  kb-embeddings-forum.npz
  kb-embeddings-issues.npz
  kb-embeddings-pulls.npz
  knowledge_base.json              lexical (BM25-style) index over the same chunks
```

- **Sync.** On every MCP server start (unless `PRECICE_AI_SKIP_KB_BOOTSTRAP` is set), and
  whenever `kb_ingest_precice_data` / `kb_query_precice_live` runs. A cached asset is
  trusted for **96 h** without any network call, then re-downloaded.
- **Query.** `kb_query_precice` embeds the question through the embedding API and runs
  vector search over the cached `.npz` files, optionally restricted to one `category`.
  `kb_query_precice_live` refreshes the category first. `kb_query_precice_lexical` runs
  keyword search over `knowledge_base.json`.
- **Status in this app.** `precice_ai/mcp_kb.py::local_kb_status()` reads the same
  directory directly (file presence, chunk counts from the `.npz` files, document count
  from the JSON, and modification times). `ingest.py::run_ingestion()` copies that into
  the `status` dict served at `GET /api/status`. It runs at import, at startup, hourly via
  APScheduler, and on `POST /api/reingest`. **It never downloads or rebuilds the KB** —
  that is the MCP server's job.
- **Opt-in fallback.** `search_precice_docs` (local tool) runs a BM25-like search over
  `knowledge_base.json` directly. It's only bound to the model when MCP is **not**
  connected **and** `PRECICE_AI_ALLOW_LOCAL_SEARCH_FALLBACK=true`. Its `[Source: url]`
  markers become the source pills in the UI.

## 4. The ReAct loop (`precice_ai/graph.py`)

### State

```python
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]  # append-only transcript
    working_dir: str   # session sandbox root
    session_id: str    # for attachment lookups
```

### Graph

```python
graph.add_node("agent", call_llm)
graph.add_node("tools", ToolNode(get_all_tools()))   # local + MCP tools
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", tools_condition) # "tools" or END
graph.add_edge("tools", "agent")
```

### The `agent` node (`call_llm`)

1. **Deterministic MCP prefetch.** If the latest user message looks like a preCICE
   question (keyword match on `precice`, `coupling`, `participant`, `mapping`, `adapter`,
   …) and MCP is connected, the node runs these steps itself before the LLM sees the
   message:
   - calls `kb_precice_status`;
   - calls `kb_ingest_precice_data` if no vector category is `ok` or any is stale;
   - calls `kb_query_precice` (or `kb_query_precice_live` as a fallback) with
     `top_k=5`.

   The result is appended to the system prompt as *"Deterministic MCP context"*. This
   gives the model grounded KB context even if it would otherwise skip the tools.
2. **System prompt.** Covers the assistant's role, the working directory, the session id,
   and the MCP connection status and error. It also sets the KB protocol (status → ingest
   if stale → query; live query only when needed), requires sources to be cited, forbids
   answering preCICE questions from memory while MCP is available, and requires
   confirmation before writing files.
3. **LLM call.** The system prompt plus the last `MAX_SESSION_MESSAGES` (20) messages go
   to `build_chat_model().bind_tools(get_all_tools())`. The bound model is cached and
   rebuilt if the provider, model, or base URL changes.

### Streaming (`stream_agent`)

`app.astream_events(state, version="v2")` is translated into SSE events:

| LangGraph event | SSE payload |
|---|---|
| `on_chat_model_stream` | `{"type": "token", "content"}` |
| `on_tool_start` | `{"type": "tool_start", "tool", "call_id", "input"}` |
| `on_tool_end` | `{"type": "tool_end", "tool", "call_id", "output"}` |
| *(derived)* | `{"type": "sources", "content": [{"url", "type"}]}` — from `search_precice_docs` output |
| *(server error)* | `{"type": "error", "content"}` |
| *(end)* | `{"type": "done"}` |

`call_id` is the LangGraph run id. The UI uses it to match each `tool_end` to its card, so
parallel calls to the same tool render correctly. Each tool event is also logged with
`source: "mcp"` or `"local"`.

## 5. Tools

`get_all_tools()` returns the bound set: **local tools** (minus `search_precice_docs`
unless the fallback applies) **plus every MCP tool**.

### Local tools (`precice_ai/tools.py`)

| Tool | Purpose | Notes |
|---|---|---|
| `list_project_files(working_dir, extension_filter="")` | Tree listing of the project | skips dot-dirs, `__pycache__`, `node_modules` |
| `read_project_file(file_path, working_dir)` | Read a project file | 50 KB cap, rejects binary |
| `write_project_file(file_path, content, working_dir)` | Create/overwrite a file | creates parent dirs; prompt requires user confirmation |
| `validate_precice_config(config_path, working_dir)` | `precice-tools check <file>` | fixed argv, 30 s timeout |
| `search_forum_live(query)` | Live Discourse search | top 5 topics |
| `read_attached_file(filename, session_id)` | Read a chat attachment | in-memory only |
| `search_precice_docs(query)` | Lexical search over the cached KB | **fallback only** (see §3) |

**Sandbox contract** (every local file tool):

```python
base = Path(working_dir).resolve()
resolved = (base / file_path).resolve()          # or Path(file_path).resolve() if absolute
if not resolved.is_relative_to(base):
    return "Access denied: ... is outside the project directory."
```

`working_dir` is set only through `POST /api/session/{sid}/workdir`, and `/api/chat`
refuses to run until it is set. Every tool returns a string and never raises: exceptions
come back as `"Error ...: {e}"` so a failing tool can't kill the SSE stream.

## 6. Request lifecycle

1. `precice-ai` (`cli.py`) loads `.env`, applies CLI overrides, exits if no API key is
   set, opens the browser, and starts uvicorn.
2. **Lifespan.** Loads MCP tools, runs `run_ingestion()` (KB status), builds the graph,
   and schedules an hourly status refresh.
3. The browser calls `POST /api/session`, `POST /api/session/{sid}/workdir`, and
   optionally `POST /api/upload/{sid}`.
4. `POST /api/chat {message, session_id}` appends the `HumanMessage` (history is capped at
   20 messages) and streams `stream_agent(...)`.
5. The graph runs `agent → [tools → agent]* → END`. The agent node does the MCP prefetch
   for preCICE questions.
6. The concatenated tokens are stored as an `AIMessage`. Tool messages are not kept in
   session history.

## 7. HTTP API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Serve `static/index.html` |
| `GET` | `/api/config` | Public runtime config (provider, model, base URL, `.env` path, log path, host/port) |
| `GET` | `/api/status` | KB cache status (`state`, `chunk_count`, `source_count`, `category_count`, `kb_dir`, `last_updated`, …) + `tools` + `log_file` |
| `GET` | `/api/tools` | MCP connection status, server path/interpreter, local and MCP tool names |
| `POST` | `/api/reingest` | Re-read the KB cache status (the UI's **Re-index** button). Does **not** download anything — use `kb_ingest_precice_data` or `precice-ai kb ingest` for that |
| `POST` | `/api/session` | Create a session → `{"session_id"}` |
| `DELETE` | `/api/session/{sid}` | Delete the session and its attachments |
| `POST` | `/api/session/{sid}/workdir` | Set the working directory `{"path": "..."}` |
| `GET` | `/api/pick-directory` | Open a native folder dialog on the server machine (tkinter, or PowerShell under WSL) |
| `POST` | `/api/upload/{sid}` | Attach a UTF-8 text file (multipart) |
| `POST` | `/api/chat` | SSE-streamed chat turn |

## 8. Activity logging

`precice_ai/logger.py` writes JSONL to `logs/agent.jsonl` (override with
`PRECICE_AI_LOG_FILE`) and echoes each event to the terminal. Event types: `mcp_startup`,
`chat_request`, `agent_input`, `llm_call`, `agent_output`, `tool_call`, `tool_response`,
`chat_response`. Tool events carry `source: "mcp" | "local"`, and prefetch calls are
logged as MCP tool calls. Logging errors are swallowed so they never break a chat stream.

## 9. File map

```
precice_ai/
  cli.py           `precice-ai` entry point: .env + CLI overrides, key check, browser, uvicorn
  config.py        env-driven settings (loads repo .env), public_runtime_config()
  llm.py           provider-aware ChatOpenAI factory
  tools.py         local @tool functions, MCP client startup/preflight, get_all_tools()
  graph.py         AgentState, MCP prefetch, call_llm, StateGraph, stream_agent (SSE events)
  mcp_kb.py        reads the MCP KB cache on disk: status + lexical BM25 fallback search
  ingest.py        KB status dict for /api/status (no scraping; refreshed hourly)
  conversation.py  in-memory sessions, attachments, per-session working_dir
  logger.py        JSONL activity logging
  server.py        FastAPI app, lifespan, REST + SSE endpoints, native folder picker
  vectorstore.py   legacy ChromaDB singleton — not imported anywhere (vestigial)
static/
  index.html       single-file vanilla JS UI: SSE consumer, tool cards, source pills
CLAUDE.md                     original build brief (see its "Current architecture" note)
CODEX_AGENT_REQUIREMENTS.md   product requirements / direction
```

## 10. Extending

**A new local tool** (needs the session sandbox or in-memory session state):

1. Add an `@tool` function to `precice_ai/tools.py`. The docstring is all the LLM sees, so
   say *when* to use the tool.
2. Wrap the body in `try/except` and return a string on every path.
3. For filesystem access, apply the `is_relative_to(working_dir)` check.
4. Add it to `LOCAL_TOOLS`.

**A new preCICE knowledge or `precice-cli` capability**: add it to the `precice-ai` MCP
server instead. It is picked up automatically on the next start of this app, and other
MCP clients (Claude Code, Claude Desktop, …) get it too.

## Known limitations

- **Two different project roots.** Local file tools use the session working directory.
  The MCP project/config/log tools resolve `project_name` under `PRECICE_PROJECTS_DIR`,
  which defaults to the sibling repo's `test-projects/`. To point both at your work, set
  `PRECICE_PROJECTS_DIR` to the **parent** of your project folder and refer to projects
  by folder name. The MCP tools are not bound by the session sandbox — their boundary is
  `PRECICE_PROJECTS_DIR` plus the sibling's command allow-list.
- **Source pills** only come from `search_precice_docs`, so MCP KB answers show citations
  inline in the text (the prompt requires it) rather than as pills.
- **Keyword-based prefetch.** Questions that don't mention a preCICE term skip the
  prefetch, and the model decides for itself whether to call the KB tools.
- **Heavy install.** `chromadb` / `sentence-transformers` are still in `pyproject.toml`
  but unused.
