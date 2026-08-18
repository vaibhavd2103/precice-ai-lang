import json
import os
import subprocess
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from precice_ai.conversation import (
    append_ai,
    append_human,
    create_session,
    delete_session,
    get_session,
    has_working_dir,
    set_working_dir,
    store_attachment,
)
from precice_ai.graph import build_graph, stream_agent
from precice_ai.ingest import run_ingestion, status as ingest_status
from precice_ai.logger import log_event, log_file_path
from precice_ai.tools import get_tool_status, initialize_tools
from precice_ai import config

STATIC_DIR = Path(__file__).parent.parent / "static"

rag_app = None


def _pick_directory_path() -> str:
    if os.environ.get("WSL_DISTRO_NAME"):
        script = (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$dialog = New-Object System.Windows.Forms.FolderBrowserDialog; "
            "$dialog.Description = 'Select preCICE project directory'; "
            "if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { "
            "Write-Output $dialog.SelectedPath }"
        )
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-STA", "-Command", script],
            capture_output=True,
            text=True,
            timeout=120,
        )
        selected = result.stdout.strip()
        if not selected:
            return ""
        converted = subprocess.run(
            ["wslpath", "-a", selected],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return converted.stdout.strip() or selected

    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(title="Select preCICE project directory")
        root.destroy()
        return selected.strip()
    except Exception as exc:
        raise RuntimeError(f"Directory picker unavailable: {exc}") from exc


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_app

    tool_status = await initialize_tools()
    print(f"Available tools at startup: {tool_status['startup_tools']}")
    if tool_status["mcp_error"]:
        print(f"MCP tool loading warning: {tool_status['mcp_error']}")

    run_ingestion()
    rag_app = build_graph()

    scheduler = BackgroundScheduler()
    scheduler.add_job(run_ingestion, "interval", hours=1)
    scheduler.start()

    yield
    scheduler.shutdown()


app = FastAPI(title="preCICE AI", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Static files ---

@app.get("/")
async def serve_index():
    return FileResponse(STATIC_DIR / "index.html")


# --- Status ---

@app.get("/api/status")
async def get_status():
    return {
        **ingest_status,
        "tools": get_tool_status(),
        "log_file": log_file_path(),
    }


@app.get("/api/tools")
async def get_tools():
    return get_tool_status()


@app.get("/api/config")
async def get_runtime_config():
    return config.public_runtime_config()


# --- Session management ---

@app.post("/api/session")
async def new_session():
    sid = create_session()
    return {"session_id": sid}


@app.delete("/api/session/{sid}")
async def remove_session(sid: str):
    delete_session(sid)
    return {"ok": True}


class WorkDirRequest(BaseModel):
    path: str


@app.post("/api/session/{sid}/workdir")
async def set_workdir(sid: str, body: WorkDirRequest):
    session = get_session(sid)
    if not session:
        return JSONResponse({"error": "session not found"}, status_code=404)
    if not body.path.strip():
        return JSONResponse({"error": "path is required"}, status_code=400)
    set_working_dir(sid, body.path)
    return {"ok": True, "path": body.path}


@app.get("/api/pick-directory")
async def pick_directory():
    try:
        selected = _pick_directory_path()
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
    if not selected:
        return JSONResponse({"error": "No directory selected"}, status_code=400)
    return {"ok": True, "path": selected}


# --- File upload ---

@app.post("/api/upload/{session_id}")
async def upload_file(session_id: str, file: UploadFile = File(...)):
    session = get_session(session_id)
    if not session:
        return JSONResponse({"error": "session not found"}, status_code=404)
    content = await file.read()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return JSONResponse({"error": "Binary files are not supported"}, status_code=400)
    store_attachment(session_id, file.filename, text)
    return {"ok": True, "filename": file.filename, "size": len(text)}


# --- Chat (SSE streaming) ---

class ChatRequest(BaseModel):
    message: str
    session_id: str


@app.post("/api/chat")
async def chat(req: ChatRequest):
    session = get_session(req.session_id)
    if not session:
        return JSONResponse({"error": "session not found"}, status_code=404)
    if not config.llm_is_configured():
        return JSONResponse(
            {"error": "LLM is not configured. Set the API key and model via .env or CLI flags before starting the server."},
            status_code=400,
        )
    if not has_working_dir(req.session_id):
        return JSONResponse(
            {"error": "No working directory selected for this session. Set a folder before chatting."},
            status_code=400,
        )

    append_human(req.session_id, req.message)
    log_event(
        "chat_request",
        session_id=req.session_id,
        working_dir=session["working_dir"],
        input=req.message,
    )

    async def event_stream():
        full_answer = ""
        try:
            async for event in stream_agent(
                rag_app,
                session["messages"],
                session["working_dir"],
                req.session_id,
            ):
                yield f"data: {json.dumps(event)}\n\n"
                if event["type"] == "token":
                    full_answer += event["content"]
        except Exception as e:
            err = {"type": "error", "content": str(e)}
            yield f"data: {json.dumps(err)}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        if full_answer:
            append_ai(req.session_id, full_answer)
            log_event(
                "chat_response",
                session_id=req.session_id,
                output=full_answer,
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --- Manual re-ingestion trigger ---

@app.post("/api/reingest")
async def reingest():
    threading.Thread(target=run_ingestion, daemon=True).start()
    return {"ok": True, "message": "Re-ingestion started"}
