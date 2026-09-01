"""FastAPI backend.

Two kinds of endpoint:

  READ (everything precomputed) - the dashboard only ever reads. Nothing is
  transcribed at request time, which is an explicit requirement of the brief.

  GET /api/bundle - agents + customers + calls in one round trip, so the
  dashboard's first screen is one request rather than three.

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
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
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
def _query_calls(
    conn,
    *,
    limit: int | None = None,
    offset: int = 0,
    customerId: str | None = None,
    agentId: str | None = None,
) -> list[dict]:
    """The one place calls are listed. `/api/calls` and `/api/bundle` both go
    through here so their ordering and filtering can never drift apart.

    `limit=None` means no limit - SQLite reads a negative LIMIT as unbounded,
    which keeps OFFSET usable without a second query shape."""
    sql = "SELECT record FROM calls WHERE 1=1"
    args: list = []
    if customerId:
        sql += " AND customer_id = ?"
        args.append(customerId)
    if agentId:
        sql += " AND agent_id = ?"
        args.append(agentId)
    sql += " ORDER BY started_ms DESC LIMIT ? OFFSET ?"
    args += [-1 if limit is None else limit, offset]
    return [json.loads(r["record"]) for r in conn.execute(sql, args)]


@app.get("/api/calls")
def list_calls(
    conn=Depends(get_conn),
    limit: int = Query(50, le=500),
    offset: int = 0,
    customerId: str | None = None,
    agentId: str | None = None,
):
    return _query_calls(
        conn, limit=limit, offset=offset, customerId=customerId, agentId=agentId
    )


@app.get("/api/bundle")
def bundle(
    conn=Depends(get_conn),
    limit: int | None = Query(None, ge=1, description="applies to `calls` only"),
    offset: int = Query(0, ge=0, description="applies to `calls` only"),
):
    """`BundleResponse`: agents + customers + calls in one round trip.

    The dashboard needs all three to render its first screen; three requests
    to fetch them is three chances to render half a page. `calls` is unlimited
    by default - the corpus is ~1,400 rows, which is smaller than the audio for
    a single call.
    """
    return {
        "agents": aggregates.agents(conn),
        "customers": aggregates.customers(conn),
        "calls": _query_calls(conn, limit=limit, offset=offset),
    }


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
# The metadata field is read straight off the raw multipart body rather than
# declared as a parameter, so it is documented by hand.
_PROCESS_FORM = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": ["audio"],
                    "properties": {
                        "audio": {
                            "type": "string",
                            "format": "binary",
                            "description": "Call recording (stereo mp3)",
                        },
                        "metadata": {
                            "type": "string",
                            "description": "Call metadata JSON - either an "
                            "uploaded .json file or the JSON text itself",
                        },
                        "metadataJson": {
                            "type": "string",
                            "description": "Legacy alias for metadata as a string",
                        },
                    },
                }
            }
        },
    }
}


async def _form_text(value: Any) -> str | None:
    """Collapse one multipart value to text, whether it arrived as a file part
    or a plain field.

    A caller should not have to use a different field name depending on how
    they packed the same JSON: the browser sends `metadata` as a string, curl's
    `-F metadata=@file.json` sends it as a file, and both are legitimate. A
    file part has `.read()`; a plain field is already a string.
    """
    if value is None:
        return None
    if hasattr(value, "read"):
        data = await value.read()
        if isinstance(data, (bytes, bytearray)):
            try:
                value = data.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise HTTPException(400, f"metadata is not UTF-8 text: {exc}") from exc
        else:
            value = data
    return str(value).strip() or None


@app.post("/api/process", openapi_extra=_PROCESS_FORM)
async def process(
    request: Request,
    audio: UploadFile = File(..., description="Call recording (stereo mp3)"),
    conn=Depends(get_conn),
):
    """Transcribe + analyse one call and return the full CallRecord.

    Metadata is accepted, in priority order, as: a `metadata` file part, a
    `metadata` string field, or a `metadataJson` string field (kept for
    backwards compatibility). FastAPI cannot overload one parameter name
    across File and Form, so the form is inspected directly instead.
    """
    form = await request.form()

    text = source = None
    for field in ("metadata", "metadataJson"):
        text = await _form_text(form.get(field))
        if text:
            source = field
            break
    if not text:
        raise HTTPException(
            400,
            "No call metadata. Send it as 'metadata' - either a .json file part "
            "or the JSON text as a form field - or as 'metadataJson'.",
        )

    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise HTTPException(400, f"{source} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise HTTPException(
            400, f"{source} must be a JSON object, got {type(raw).__name__}"
        )

    call_id = str(raw.get("sid") or Path(audio.filename or "upload").stem)
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
