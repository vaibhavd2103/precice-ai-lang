import uuid
from typing import Dict, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from precice_ai import config

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


def append_human(sid: str, text: str) -> None:
    SESSIONS[sid]["messages"].append(HumanMessage(content=text))
    if len(SESSIONS[sid]["messages"]) > config.MAX_SESSION_MESSAGES:
        SESSIONS[sid]["messages"] = SESSIONS[sid]["messages"][-config.MAX_SESSION_MESSAGES:]


def append_ai(sid: str, text: str) -> None:
    SESSIONS[sid]["messages"].append(AIMessage(content=text))
    if len(SESSIONS[sid]["messages"]) > config.MAX_SESSION_MESSAGES:
        SESSIONS[sid]["messages"] = SESSIONS[sid]["messages"][-config.MAX_SESSION_MESSAGES:]


def set_working_dir(sid: str, path: str) -> None:
    SESSIONS[sid]["working_dir"] = path


def has_working_dir(sid: str) -> bool:
    return bool(SESSIONS.get(sid, {}).get("working_dir", "").strip())


def store_attachment(sid: str, filename: str, content: str) -> None:
    ATTACHMENT_STORE[sid][filename] = content


def delete_session(sid: str) -> None:
    SESSIONS.pop(sid, None)
    ATTACHMENT_STORE.pop(sid, None)
