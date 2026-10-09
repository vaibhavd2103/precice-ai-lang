import json
import re
from typing import Annotated, AsyncIterator, List
import operator

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from precice_ai import config as settings
from precice_ai.llm import build_chat_model
from precice_ai.logger import log_event
from precice_ai.tools import get_all_tools, get_mcp_tool, get_tool_status


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    working_dir: str
    session_id: str


_llm = None
_llm_with_tools = None
_llm_signature = None


def _content_to_text(content: object) -> str:
    """Normalize LangChain string or structured content blocks to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts)
    return str(content)


def _is_precice_question(text: object) -> bool:
    lowered = _content_to_text(text).lower()
    terms = (
        "precice", "pre-cice", "coupling", "participant", "mapping", "adapter",
        "precice-config", "serial-implicit", "fluid-structure", "mesh exchange",
    )
    return any(term in lowered for term in terms)


async def _mcp_call(tool: object, payload: dict, config: RunnableConfig | None) -> str:
    """Invoke an async MCP tool for deterministic prefetching.

    MCP tools are async-only, so they must be awaited via ``ainvoke``. Passing the
    run config keeps the call visible as a tool card in the event stream.
    """
    try:
        result = await tool.ainvoke(payload, config)
        return _content_to_text(getattr(result, "content", result))
    except Exception as exc:
        return f"MCP tool error: {exc}"


def _parse_json(text: str) -> dict:
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


_kb_ingest_attempted = False


async def _prefetch_precice_context(question: str, config: RunnableConfig | None) -> str | None:
    """Fetch MCP KB context once per user turn, before the LLM answers a preCICE question."""
    global _kb_ingest_attempted
    if not _is_precice_question(question) or not get_tool_status()["mcp_connected"]:
        return None

    status_tool = get_mcp_tool("kb_precice_status")
    if status_tool is not None and not _kb_ingest_attempted:
        status = _parse_json(await _mcp_call(status_tool, {}, config))
        categories = (status.get("vector") or {}).get("categories")
        # Only ingest when the KB is genuinely empty; ingest is slow, so try once per process.
        if isinstance(categories, dict) and not categories:
            ingest_tool = get_mcp_tool("kb_ingest_precice_data")
            if ingest_tool is not None:
                _kb_ingest_attempted = True
                await _mcp_call(ingest_tool, {}, config)

    query_tool = get_mcp_tool("kb_query_precice") or get_mcp_tool("kb_query_precice_live")
    if query_tool is None:
        return "The preCICE MCP server is connected, but no query tool is available."

    result = await _mcp_call(query_tool, {"question": question, "top_k": 5}, config)
    parsed = _parse_json(result)
    if result.startswith("MCP tool error") or parsed.get("status") in {"error", "empty"}:
        return (
            f"The preCICE MCP knowledge base query ({query_tool.name}) returned no usable "
            f"results: {result[:600]}\nTell the user the knowledge base is unavailable or "
            "needs rebuilding, and do not present memory-based answers as KB-sourced."
        )
    return f"PreCICE MCP knowledge retrieved with {query_tool.name}:\n{result}"


def _get_llm_with_tools():
    global _llm, _llm_with_tools, _llm_signature
    signature = (settings.LLM_PROVIDER, settings.MODEL, settings.LLM_BASE_URL, bool(settings.LLM_API_KEY))
    if _llm_with_tools is None or _llm_signature != signature:
        _llm = build_chat_model()
        _llm_with_tools = _llm.bind_tools(get_all_tools())
        _llm_signature = signature
    return _llm_with_tools


async def call_llm(state: AgentState, config: RunnableConfig) -> dict:
    tool_status = get_tool_status()
    log_event(
        "agent_input",
        session_id=state["session_id"],
        working_dir=state["working_dir"],
        message_count=len(state["messages"]),
        input=state["messages"][-1].content if state["messages"] else "",
    )
    # Prefetch only on the first loop iteration of a turn (last message is the user's);
    # later iterations follow tool results and must not re-query with tool output.
    last = state["messages"][-1] if state["messages"] else None
    prefetched_context = (
        await _prefetch_precice_context(_content_to_text(last.content), config)
        if isinstance(last, HumanMessage) else None
    )
    context_message = (
        f"\n\nDeterministic MCP context for this question:\n{prefetched_context}"
        if prefetched_context else ""
    )
    system = SystemMessage(content=f"""You are preCICE AI, an expert assistant for the \
preCICE multiphysics coupling library. You help with configuration, coupling schemes, \
installation, adapters, tutorials, and debugging.

The user's simulation project is at: {state['working_dir'] or 'no folder selected'}
Session ID: {state['session_id']}

Use your tools to look up documentation, read project files, or validate configs.
Always cite which source (doc page or forum post) your answer comes from.
For every question involving preCICE, use the connected preCICE MCP knowledge tools.
Call kb_precice_status first. If the vector categories are empty or stale, call
kb_ingest_precice_data to build/refresh the category-wise embedding files, then call
kb_precice_status again and query with kb_query_precice. Use kb_query_precice_live only
when ingestion cannot provide the required category. Do not answer a preCICE knowledge
question from memory when an MCP knowledge tool is available. If MCP is not connected,
explain that the MCP server is unavailable instead of presenting a fallback result as
current MCP knowledge.
The server is configured at startup using environment variables or CLI flags.
The user chooses a working directory per chat session, and all file tool calls must stay inside it.
If no working directory is set and the task needs local files, ask the user to set one first.
When writing or modifying files, confirm with the user what you are about to do first.
MCP connection at startup: {tool_status['mcp_connected']}. MCP startup error: {tool_status['mcp_error'] or 'none'}.
If MCP is unavailable, say so clearly and do not claim that you searched the preCICE KB.
""" + context_message)
    messages = [system] + state["messages"][-settings.MAX_SESSION_MESSAGES:]
    llm = _get_llm_with_tools()
    log_event(
        "llm_call",
        session_id=state["session_id"],
        model=settings.MODEL,
        message_count=len(messages),
    )
    response = await llm.ainvoke(messages, config)
    log_event(
        "agent_output",
        session_id=state["session_id"],
        output=response.content,
        tool_calls=[call.get("name") for call in getattr(response, "tool_calls", [])],
    )
    return {"messages": [response]}


def _tool_arg_names(tool: object) -> set[str]:
    schema = getattr(tool, "args_schema", None)
    if isinstance(schema, dict):
        return set(schema.get("properties", {}))
    if schema is not None and hasattr(schema, "model_json_schema"):
        return set(schema.model_json_schema().get("properties", {}))
    return set()


def make_tool_runner(tools: list):
    """Run tool calls with session-scoped arguments forced by the server.

    The model fills tool arguments itself, so ``working_dir``/``session_id``/``cwd``
    are overwritten with the session's values; the model cannot widen file access.
    """
    tool_node = ToolNode(tools)
    arg_names = {t.name: _tool_arg_names(t) for t in tools}

    async def run_tools(state: AgentState, config: RunnableConfig) -> dict:
        last = state["messages"][-1]
        pinned = {
            "working_dir": state["working_dir"],
            "cwd": state["working_dir"],
            "session_id": state["session_id"],
        }
        calls = []
        for call in last.tool_calls:
            args = dict(call["args"])
            for key, value in pinned.items():
                if key in arg_names.get(call["name"], ()):
                    args[key] = value
            calls.append({**call, "args": args})
        patched = last.model_copy(update={"tool_calls": calls})
        return await tool_node.ainvoke({**state, "messages": [patched]}, config)

    return run_tools


def build_graph():
    tools = get_all_tools()
    tool_status = get_tool_status()
    if tool_status["mcp_connected"]:
        print("Using sibling preCICE MCP server for preCICE knowledge questions.")
    else:
        print("preCICE MCP unavailable; using local documentation search fallback.")
    print(f"Graph tools: {[t.name for t in tools]}")

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_llm)
    graph.add_node("tools", make_tool_runner(tools))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile()


async def stream_agent(
    app,
    messages: list,
    working_dir: str,
    session_id: str,
) -> AsyncIterator[dict]:
    """
    Yields dicts:
      {"type": "token",      "content": str}
      {"type": "tool_start", "tool": str, "call_id": str, "input": dict}
      {"type": "tool_end",   "tool": str, "call_id": str, "output": str}
      {"type": "sources",    "content": list[dict]}
      {"type": "done"}
    """
    state = {
        "messages": messages,
        "working_dir": working_dir,
        "session_id": session_id,
    }
    sources_seen: list[dict] = []

    async for event in app.astream_events(state, version="v2", config={"recursion_limit": 40}):
        kind = event["event"]

        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if chunk.content:
                yield {"type": "token", "content": chunk.content}

        elif kind == "on_tool_start":
            tool_name = event["name"]
            call_id = str(event.get("run_id") or event.get("data", {}).get("tool_call_id") or "")
            mcp_tool_names = set(get_tool_status()["mcp_tools"])
            log_event(
                "tool_call",
                session_id=session_id,
                source="mcp" if tool_name in mcp_tool_names else "local",
                tool=tool_name,
                input=event["data"].get("input", {}),
            )
            yield {
                "type": "tool_start",
                "tool": tool_name,
                "call_id": call_id,
                "input": event["data"].get("input", {}),
            }

        elif kind in ("on_tool_end", "on_tool_error"):
            output = event["data"].get("output", event["data"].get("error", ""))
            output = getattr(output, "content", output)
            if kind == "on_tool_error":
                output = f"Error: {output}"
            tool_name = event["name"]
            call_id = str(event.get("run_id") or event.get("data", {}).get("tool_call_id") or "")
            mcp_tool_names = set(get_tool_status()["mcp_tools"])
            log_event(
                "tool_response",
                session_id=session_id,
                source="mcp" if tool_name in mcp_tool_names else "local",
                tool=tool_name,
                output=_content_to_text(output),
            )
            yield {
                "type": "tool_end",
                "tool": tool_name,
                "call_id": call_id,
                "output": _content_to_text(output),
            }
            if tool_name == "search_precice_docs":
                for match in re.finditer(r'\[Source: (https?://[^\]]+)\]', _content_to_text(output)):
                    url = match.group(1)
                    if url not in [s["url"] for s in sources_seen]:
                        sources_seen.append({"url": url, "type": "docs"})

    if sources_seen:
        yield {"type": "sources", "content": sources_seen}
    yield {"type": "done"}
