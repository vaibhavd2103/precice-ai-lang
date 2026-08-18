import re
from typing import Annotated, AsyncIterator, List
import operator

from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from precice_ai import config
from precice_ai.llm import build_chat_model
from precice_ai.logger import log_event
from precice_ai.tools import get_all_tools, get_tool_status


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    working_dir: str
    session_id: str


_llm = None
_llm_with_tools = None
_llm_signature = None


def _get_llm_with_tools():
    global _llm, _llm_with_tools, _llm_signature
    signature = (config.LLM_PROVIDER, config.MODEL, config.LLM_BASE_URL, bool(config.LLM_API_KEY))
    if _llm_with_tools is None or _llm_signature != signature:
        _llm = build_chat_model()
        _llm_with_tools = _llm.bind_tools(get_all_tools())
        _llm_signature = signature
    return _llm_with_tools


def call_llm(state: AgentState) -> dict:
    tool_status = get_tool_status()
    log_event(
        "agent_input",
        session_id=state["session_id"],
        working_dir=state["working_dir"],
        message_count=len(state["messages"]),
        input=state["messages"][-1].content if state["messages"] else "",
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
""")
    messages = [system] + state["messages"][-config.MAX_SESSION_MESSAGES:]
    llm = _get_llm_with_tools()
    log_event(
        "llm_call",
        session_id=state["session_id"],
        model=config.MODEL,
        message_count=len(messages),
    )
    response = llm.invoke(messages)
    log_event(
        "agent_output",
        session_id=state["session_id"],
        output=response.content,
        tool_calls=[call.get("name") for call in getattr(response, "tool_calls", [])],
    )
    return {"messages": [response]}


def build_graph():
    tools = get_all_tools()
    tool_status = get_tool_status()
    if tool_status["mcp_connected"]:
        print("Using sibling preCICE MCP server for preCICE knowledge questions.")
    else:
        print("preCICE MCP unavailable; using local documentation search fallback.")
    tool_node = ToolNode(tools)
    print(f"Graph tools: {[t.name for t in tools]}")

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_llm)
    graph.add_node("tools", tool_node)
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
      {"type": "tool_start", "tool": str, "input": dict}
      {"type": "tool_end",   "tool": str, "output": str}
      {"type": "sources",    "content": list[dict]}
      {"type": "done"}
    """
    state = {
        "messages": messages,
        "working_dir": working_dir,
        "session_id": session_id,
    }
    sources_seen: list[dict] = []

    async for event in app.astream_events(state, version="v2"):
        kind = event["event"]

        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if chunk.content:
                yield {"type": "token", "content": chunk.content}

        elif kind == "on_tool_start":
            tool_name = event["name"]
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
                "input": event["data"].get("input", {}),
            }

        elif kind == "on_tool_end":
            output = event["data"].get("output", "")
            tool_name = event["name"]
            mcp_tool_names = set(get_tool_status()["mcp_tools"])
            log_event(
                "tool_response",
                session_id=session_id,
                source="mcp" if tool_name in mcp_tool_names else "local",
                tool=tool_name,
                output=str(output),
            )
            yield {
                "type": "tool_end",
                "tool": tool_name,
                "output": str(output)[:500],
            }
            if tool_name == "search_precice_docs":
                for match in re.finditer(r'\[Source: (https?://[^\]]+)\]', str(output)):
                    url = match.group(1)
                    if url not in [s["url"] for s in sources_seen]:
                        sources_seen.append({"url": url, "type": "docs"})

    if sources_seen:
        yield {"type": "sources", "content": sources_seen}
    yield {"type": "done"}
