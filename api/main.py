"""FastAPI backend.

Two kinds of endpoint:

  READ (everything precomputed) - the dashboard only ever reads. Nothing is
  transcribed at request time, which is an explicit requirement of the brief.

  POST /api/process - the live demo path. Judges hand over an mp3 + metadata
  json as multipart form data; it runs the same pipeline, stores the result and
  returns the full CallRecord. Same models as the bulk run, so a call processed
  live is indistinguishable from one processed in advance.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from callradar import aggregates, config, db
from callradar.pipeline import process_call, store

app = FastAPI(
    title="Call-Centre Radar API",
    version="1.0.0",
    description="Transcripts and evidence-cited analysis over recorded support calls.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_conn = None


def get_conn():
    global _conn
    if _conn is None:
        _conn = db.connect()
    return _conn


@app.get("/api/health")
def health(conn=Depends(get_conn)):
    n = conn.execute("SELECT COUNT(*) c FROM calls").fetchone()["c"]
    return {"status": "ok", "calls": n, "asr": config.ASR_BACKEND, "llm": config.LLM_BACKEND}


# ------------------------------------------------------------------ reads
@app.get("/api/calls")
def list_calls(
    conn=Depends(get_conn),
    limit: int = Query(50, le=500),
    offset: int = 0,
    customerId: str | None = None,
    agentId: str | None = None,
):
    sql = "SELECT record FROM calls WHERE 1=1"
    args: list = []
    if customerId:
        sql += " AND customer_id = ?"
        args.append(customerId)
    if agentId:
        sql += " AND agent_id = ?"
        args.append(agentId)
    sql += " ORDER BY started_ms DESC LIMIT ? OFFSET ?"
    args += [limit, offset]
    return [json.loads(r["record"]) for r in conn.execute(sql, args)]


@app.get("/api/calls/{call_id}")
def get_call(call_id: str, conn=Depends(get_conn)):
    rec = db.get_call(conn, call_id)
    if not rec:
        raise HTTPException(404, f"Call {call_id} not found")
    return rec


@app.get("/api/calls/{call_id}/audio")
def get_audio(call_id: str):
    """Serves the recording. FileResponse handles HTTP Range, which is what
    makes seek-to-timestamp work in the browser - without it, clicking a
    citation silently fails to jump."""
    for ext in (".mp3", ".wav", ".m4a"):
        p = config.AUDIO_DIR / f"{call_id}{ext}"
        if p.exists():
            return FileResponse(p, media_type="audio/mpeg", filename=p.name)
    raise HTTPException(404, f"No audio file for {call_id}")


@app.get("/api/customers")
def list_customers(conn=Depends(get_conn)):
    return aggregates.customers(conn)


@app.get("/api/customers/{customer_id}/calls")
def customer_calls(customer_id: str, conn=Depends(get_conn)):
    return [
        json.loads(r["record"])
        for r in conn.execute(
            "SELECT record FROM calls WHERE customer_id = ? ORDER BY started_ms DESC",
            (customer_id,),
        )
    ]


@app.get("/api/agents")
def list_agents(conn=Depends(get_conn)):
    return aggregates.agents(conn)


@app.get("/api/agents/metrics")
def agent_metrics(conn=Depends(get_conn)):
    return aggregates.agent_metrics(conn)


@app.get("/api/attention")
def attention_queue(conn=Depends(get_conn), limit: int = Query(50, le=500)):
    """The ranked 'needs a manager today' view - the landing page."""
    return [
        json.loads(r["record"])
        for r in conn.execute(
            "SELECT record FROM calls ORDER BY needs_attention DESC, started_ms DESC LIMIT ?",
            (limit,),
        )
    ]


@app.get("/api/trends")
def trends(conn=Depends(get_conn), windowDays: int = 7):
    return aggregates.trending_issues(conn, windowDays)


@app.get("/api/search")
def search(q: str, conn=Depends(get_conn), limit: int = Query(50, le=200)):
    """Full-text search across every transcript, via SQLite FTS5."""
    return db.search_transcripts(conn, q, limit)


# ------------------------------------------------------------- live process
@app.post("/api/process")
async def process(
    audio: UploadFile = File(..., description="Call recording (stereo mp3)"),
    metadata: UploadFile | None = File(None, description="Call metadata JSON"),
    metadataJson: str | None = Form(None, description="Metadata JSON as a string"),
    conn=Depends(get_conn),
):
    """Transcribe + analyse one call and return the full CallRecord.

    Accepts the metadata either as a second file or as a JSON string field, so
    it works from curl, Postman and a browser form without fuss.
    """
    if metadata is not None:
        raw = json.loads((await metadata.read()).decode("utf-8"))
    elif metadataJson:
        raw = json.loads(metadataJson)
    else:
        raise HTTPException(400, "Provide metadata as a file or metadataJson field")

    call_id = str(raw.get("sid") or Path(audio.filename).stem)
    config.AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    dest = config.AUDIO_DIR / f"{call_id}.mp3"
    dest.write_bytes(await audio.read())

    try:
        result = process_call(dest, raw, conn=conn)
    except Exception as exc:  # surface the real cause, don't swallow it
        raise HTTPException(500, f"Processing failed: {exc}") from exc

    store(conn, result, raw)
    return JSONResponse(result.record)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
