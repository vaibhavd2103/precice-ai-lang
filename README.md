# preCICE AI

A local, agentic assistant for the [preCICE](https://precice.org) multiphysics coupling
library, built with **LangGraph**. It runs entirely on your machine, opens a browser chat
UI, and lets an LLM agent read/write files in your simulation project, search preCICE
docs + forum, and validate `precice-config.xml`.

The intended user flow is:

1. clone this repo,
2. install the Python dependencies,
3. configure the LLM through `.env` or CLI flags,
4. start the local server,
5. choose a working directory for each chat session in the UI,
6. let the agent read, write, and validate files only inside that selected directory.

This document explains **how the agent is built**, **how each tool works**, and how the
tool set mirrors what you'd expose from an MCP server — so you can use this repo as a
template for wiring the same tools into either a LangGraph agent or an MCP server.

---

## 1. Why LangGraph (vs. a plain LLM loop)

An MCP server exposes tools over a protocol so *any* MCP-compatible client (Claude
Desktop, Claude Code, etc.) can call them. LangGraph solves a different problem: it's the
**agent runtime** — the loop that decides when to call a tool, executes it, feeds the
result back to the LLM, and repeats until the LLM has a final answer.

You can think of it as:

```
MCP server        = the tools, exposed over a protocol, for an external host to drive
LangGraph agent   = the tools + an LLM + a loop, all running inside your own process
```

Same tool *implementations*, different exposure. This project defines the tools once as
plain Python functions (`precice_ai/tools.py`) decorated with LangChain's `@tool`, and
LangGraph drives the loop. If you also want them served over MCP, wrap the same
functions with `mcp.server.fastmcp.FastMCP` — the function bodies don't change (see
§7).

---

## 2. Architecture at a glance

```
                        ┌─────────────────────────┐
 browser  ── SSE ──▶    │        server.py         │  FastAPI, /api/chat streams
                        └────────────┬─────────────┘
                                     │
                       startup config + session state
                startup LLM config from env/CLI, per-session workdir
                                     │
                                     ▼
                        ┌─────────────────────────┐
                        │        graph.py          │  LangGraph StateGraph
                        │                           │
                        │   ┌───────┐   tool_calls? ┌───────┐
                        │   │ agent │──────yes─────▶│ tools │
                        │   │ node  │◀───────────────┘ node │
                        │   └───┬───┘        no       └───────┘
                        │       │ END                    │
                        └───────┼────────────────────────┼───┘
                                │                         │
                                ▼                         ▼
                        streamed AIMessage        precice_ai/tools.py
                                                   (ALL_TOOLS list)
                                                         │
                             ┌───────────────┬───────────┼───────────────┬──────────────┐
                             ▼               ▼           ▼               ▼              ▼
                     search_precice_docs  list_project  read_project  write_project  validate_
                     (ChromaDB RAG)       _files        _file         _file          precice_config
                                                                                          │
                                                                                          ▼
                                                                                  search_forum_live
                                                                                  read_attached_file
```

Every request re-enters the graph at `agent`; the graph loops `agent → tools → agent`
until the LLM stops requesting tool calls, then `tools_condition` routes to `END`.

---

## 3. The ReAct loop (`precice_ai/graph.py`)

### 3a. State

```python
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]  # append-only message log
    working_dir: str   # absolute path to the user's project, set per-session
    session_id: str    # for attachment lookups inside tools
```

`operator.add` on `messages` means every node's return value is **appended** to the
list, not replacing it — this is what lets the `tools` node's output flow back into the
same running transcript the `agent` node sees next.

### 3b. Nodes

- **`agent`** (`call_llm`) — builds a `SystemMessage` describing the assistant's role,
  the current `working_dir`, and the local sandbox rules, prepends it to the last N
  messages, and calls `llm_with_tools.invoke(messages)`. The LLM either returns plain
  text (done) or an `AIMessage` with `tool_calls` attached.
- **`tools`** — a prebuilt LangGraph `ToolNode(ALL_TOOLS)`. It reads `tool_calls` off the
  last message, matches them by name against `ALL_TOOLS`, executes each, and returns
  `ToolMessage`s.

### 3c. Edges

```python
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", tools_condition)  # "tools" or END
graph.add_edge("tools", "agent")                       # always loop back
```

`tools_condition` is a LangGraph helper: if the last `AIMessage` has `tool_calls`, route
to `"tools"`; otherwise route to `END`. This one line *is* the entire ReAct decision —
no custom routing logic needed.

### 3d. Streaming (`stream_agent`)

The server needs token-by-token text **and** visibility into tool calls as they happen
(for the UI's tool-call cards). `graph.astream_events(state, version="v2")` gives both
in one event stream:

| LangGraph event        | Emitted as SSE                          |
| ----------------------- | ---------------------------------------- |
| `on_chat_model_stream`  | `{"type": "token", "content": ...}`      |
| `on_tool_start`         | `{"type": "tool_start", "tool", "input"}`|
| `on_tool_end`           | `{"type": "tool_end", "tool", "output"}` |
| *(derived)*             | `{"type": "sources", "content": [...]}` — parsed from `search_precice_docs` output |
| *(end of stream)*       | `{"type": "done"}`                       |

---

## 4. Tool set — the shared contract

Every tool is a plain Python function decorated with `langchain_core.tools.tool`. The
**docstring is the tool description the LLM sees** — write it like you would an MCP tool
schema description, because functionally that's what it is: the LLM has no other
information about what the tool does or when to use it.

All tools live in `precice_ai/tools.py` and are collected into `ALL_TOOLS`, which is the
single list bound to the model (`llm.bind_tools(ALL_TOOLS)`) and passed to `ToolNode`.

| Tool | Purpose | Notes |
|---|---|---|
| `search_precice_docs(query)` | RAG lookup over scraped docs + forum in ChromaDB | MMR retrieval, `k=6, fetch_k=20, lambda_mult=0.6` |
| `list_project_files(working_dir, extension_filter="")` | Tree listing of the user's project folder | `os.walk`, skips `.git`/`__pycache__` |
| `read_project_file(file_path, working_dir)` | Read a file from the project | truncates at 50 KB, rejects binary |
| `write_project_file(file_path, content, working_dir)` | Create/overwrite a file | creates parent dirs |
| `validate_precice_config(config_path, working_dir)` | Runs `precice-tools check` | only shell command ever executed, fixed argv |
| `search_forum_live(query)` | Live Discourse search (bypasses the index) | `GET .../search.json?q=...` |
| `read_attached_file(filename, session_id)` | Reads a chat-uploaded attachment | in-memory only, keyed by session |

### Security contract (applies to every file tool)

```python
base = Path(working_dir).resolve()
resolved = (base / file_path).resolve() if not Path(file_path).is_absolute() else Path(file_path).resolve()
if not resolved.is_relative_to(base):
    return "Access denied: ... is outside the project directory."
```

This is checked in `list_project_files`, `read_project_file`, `write_project_file`, and
`validate_precice_config`. `working_dir` itself can only be set through the explicit
`POST /api/session/{sid}/workdir` endpoint — the LLM cannot change it mid-conversation,
and the server rejects chat requests until a working directory is chosen for that session.

**Every tool returns a string, never raises.** Bodies are wrapped in `try/except` and
return `f"Error: {e}"` on failure — the agent loop has no exception-handling path, so a
raised exception would kill the whole `/api/chat` stream.

---

## 5. Request lifecycle

1. User clones the repo, installs dependencies, and starts the server with `.env` values
   or CLI flags such as `--provider`, `--api-key`, `--model`, and `--base-url`.
2. Browser calls `POST /api/session` → gets a `session_id`; `SESSIONS[sid]` created
   in-memory (`conversation.py`), no disk, no DB.
3. User sets a working directory for that chat session → `POST /api/session/{sid}/workdir`.
4. User sends a message → `POST /api/chat {message, session_id}`.
5. `server.py` appends the `HumanMessage`, calls
   `stream_agent(rag_app, session["messages"], session["working_dir"], session_id)`.
6. LangGraph runs `agent → [tools → agent]* → END`, streaming SSE events the whole time.
7. Final assistant text is appended back into `SESSIONS[sid]["messages"]`, capped to
   `MAX_SESSION_MESSAGES` (20) so context doesn't grow unbounded.

Nothing is ever written to a database. Restarting the process wipes all sessions and
attachments — this is intentional (see `CLAUDE.md` §16–18: no auth, no persistence, no
extra infra beyond ChromaDB in memory).

---

## 6. Ingestion pipeline (`precice_ai/ingest.py`)

Runs once at startup (background thread) and every hour after (`APScheduler`):

1. Scrapes ~26 fixed preCICE doc pages + top 50 Discourse topics (5 posts each,
   `time.sleep(0.3)` between requests to be polite).
2. Strips HTML with BeautifulSoup, splits with
   `RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)`.
3. Upserts into ChromaDB using `md5(f"{source}::{chunk_index}")` as the doc ID — same
   source re-ingested later overwrites the same IDs instead of duplicating.
4. Publishes progress via the module-level `status` dict, read by `GET /api/status`.

`search_precice_docs` and `search_forum_live` are two different tools for a reason:
the first is the **cheap, pre-indexed** path; the second is a **live, uncached** fallback
for things that happened in the last hour and haven't been re-ingested yet.

---

## 7. Reusing these tools from an MCP server

Since the tool *bodies* have no LangChain-specific logic (they're plain functions that
happen to be decorated with `@tool`), exposing the same functionality over MCP is a thin
wrapper, not a rewrite. Using the official `mcp` Python SDK:

```python
from mcp.server.fastmcp import FastMCP
from precice_ai.tools import (
    search_precice_docs, list_project_files, read_project_file,
    write_project_file, validate_precice_config, search_forum_live,
)

mcp = FastMCP("precice-ai")

# LangChain @tool-wrapped functions expose the underlying callable via .func
mcp.tool()(search_precice_docs.func)
mcp.tool()(list_project_files.func)
mcp.tool()(read_project_file.func)
mcp.tool()(write_project_file.func)
mcp.tool()(validate_precice_config.func)
mcp.tool()(search_forum_live.func)

if __name__ == "__main__":
    mcp.run()
```

`read_attached_file` is chat-session-specific (backed by `ATTACHMENT_STORE`) and doesn't
make sense over MCP, since MCP clients don't share this app's in-memory session model —
drop it or replace `session_id` with a client-supplied file path.

The docstrings you already wrote for the LangGraph tools are exactly the descriptions
MCP clients will show the LLM on the other end — no need to write them twice.

---

## 8. Running it

```bash
git clone <your-fork-or-copy>
cd precice-ai
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env
precice-ai
```

Or start it without editing `.env`:

```bash
precice-ai \
  --provider openrouter \
  --api-key sk-or-... \
  --model openai/gpt-4o-mini \
  --base-url https://openrouter.ai/api/v1
```

After startup, open `http://127.0.0.1:7860`, create a chat session, choose the working
directory for that chat, and then interact with the agent. Useful endpoints while developing:

| Endpoint | Purpose |
|---|---|
| `GET /api/config` | current startup LLM/runtime config exposed to the UI |
| `GET /api/status` | ingestion state: `starting` → `ingesting` → `ready`/`error` |
| `POST /api/reingest` | force a re-ingestion without waiting an hour |
| `POST /api/session` | new session |
| `POST /api/session/{sid}/workdir` | point that chat session at a project folder |
| `POST /api/upload/{sid}` | attach a file (text only) to the session |
| `POST /api/chat` | SSE-streamed chat turn |

### Activity logging

Agent and tool activity is written as JSONL to `logs/agent.jsonl` by default. Each chat
input/output and each tool call/response is recorded; MCP events include
`"source": "mcp"` and the MCP tool name. Set `PRECICE_AI_LOG_FILE` to choose another
path. The same events are printed to the server terminal in real time. Logging failures
never interrupt a chat stream.

The preCICE MCP server is loaded from the sibling `precice-ai` checkout. Its `fastmcp`
dependency is included in this project; after installing dependencies, restart the app
and check `GET /api/status` for `tools.mcp_connected` and the MCP tool names.

## 9. File map

```
precice_ai/
  config.py        env-driven startup settings and public runtime config
  llm.py           provider-aware ChatOpenAI factory
  vectorstore.py   ChromaDB singleton + HuggingFace embeddings
  ingest.py        scrape → chunk → upsert, hourly via APScheduler
  tools.py         the 7 @tool functions + ALL_TOOLS
  graph.py         AgentState, call_llm node, StateGraph wiring, stream_agent
  conversation.py  in-memory sessions, attachments, per-session working_dir
  server.py        FastAPI app, SSE /api/chat, lifespan (ingestion + scheduler)
  cli.py           `precice-ai` entry point, loads `.env`, opens browser
static/
  index.html       chat UI: SSE consumer, tool-call cards, source pills
```

## 10. Extending with a new tool

1. Write the function in `precice_ai/tools.py`, decorate with `@tool`, write a docstring
   that tells the LLM *when* to use it (this is the only thing the LLM sees).
2. Wrap the body in `try/except`, return a string in every branch, never raise.
3. If it touches the filesystem, apply the same `is_relative_to(working_dir)` check as
   the existing file tools.
4. Add it to `ALL_TOOLS` — that's the only place `graph.py` and `ToolNode` read from, so
   nothing else needs to change.
