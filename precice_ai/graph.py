import re
from typing import Annotated, AsyncIterator, List
import operator

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from precice_ai import config
from precice_ai.tools import ALL_TOOLS


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    working_dir: str
    session_id: str


def _build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=config.MODEL,
        openai_api_key=config.OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        streaming=True,
        temperature=0.1,
        max_tokens=1024,
        default_headers={
            "HTTP-Referer": "https://precice.org",
            "X-Title": "preCICE AI",
        },
    )


_llm = None
_llm_with_tools = None


def _get_llm_with_tools():
    global _llm, _llm_with_tools
    if _llm_with_tools is None:
        _llm = _build_llm()
        _llm_with_tools = _llm.bind_tools(ALL_TOOLS)
    return _llm_with_tools


def call_llm(state: AgentState) -> dict:
    system = SystemMessage(content=f"""You are preCICE AI, an expert assistant for the \
preCICE multiphysics coupling library. You help with configuration, coupling schemes, \
installation, adapters, tutorials, and debugging.

The user's simulation project is at: {state['working_dir'] or 'no folder selected'}
Session ID: {state['session_id']}

Use your tools to look up documentation, read project files, or validate configs.
Always cite which source (doc page or forum post) your answer comes from.
When writing or modifying files, confirm with the user what you are about to do first.
""")
    messages = [system] + state["messages"][-config.MAX_SESSION_MESSAGES:]
    llm = _get_llm_with_tools()
    response = llm.invoke(messages)
    return {"messages": [response]}


def build_graph():
    tool_node = ToolNode(ALL_TOOLS)

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
                "output": str(output)[:500],
            }
            if event["name"] == "search_precice_docs":
                for match in re.finditer(r'\[Source: (https?://[^\]]+)\]', str(output)):
                    url = match.group(1)
                    if url not in [s["url"] for s in sources_seen]:
                        sources_seen.append({"url": url, "type": "docs"})

    if sources_seen:
        yield {"type": "sources", "content": sources_seen}
    yield {"type": "done"}
