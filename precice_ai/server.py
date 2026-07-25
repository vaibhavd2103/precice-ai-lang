import json
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from precice_ai.conversation import (
    append_ai,
    append_human,
    create_session,
    delete_session,
    get_session,
    set_working_dir,
    store_attachment,
)
from precice_ai.graph import build_graph, stream_agent
from precice_ai.ingest import run_ingestion, status as ingest_status
from precice_ai.vectorstore import get_vectorstore

STATIC_DIR = Path(__file__).parent.parent / "static"

rag_app = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_app

    get_vectorstore()
    rag_app = build_graph()

    threading.Thread(target=run_ingestion, daemon=True).start()

    scheduler = BackgroundScheduler()
    scheduler.add_job(run_ingestion, "interval", hours=1)
    scheduler.start()

    yield
    scheduler.shutdown()


app = FastAPI(title="preCICE AI", lifespan=lifespan)


# --- Static files ---

@app.get("/")
async def serve_index():
    return FileResponse(STATIC_DIR / "index.html")


# --- Status ---

@app.get("/api/status")
async def get_status():
    return ingest_status


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
    set_working_dir(sid, body.path)
    return {"ok": True, "path": body.path}


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

    append_human(req.session_id, req.message)

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
