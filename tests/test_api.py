"""API surface tests.

Covers the two things a caller can get wrong without the server telling them
why: how the metadata is packed into POST /api/process, and whether the
one-request GET /api/bundle really returns the same data as the three
endpoints it replaces.

Everything runs against a throwaway copy of the database in a temp directory,
so the tests can process calls without touching data/callradar.db or
data/audio/.

    ASR_BACKEND=mock LLM_BACKEND=mock python3 tests/test_api.py
"""
from __future__ import annotations

import atexit
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ["ASR_BACKEND"] = "mock"
os.environ["LLM_BACKEND"] = "mock"

from tests.test_contract import Skipped, assert_contract  # noqa: E402

# --------------------------------------------------------------- sandboxing
# Set on the config module rather than only through the environment: config is
# read once at import, and a test run under pytest may have imported it already
# via another test module.
from callradar import config  # noqa: E402

_TMP = Path(tempfile.mkdtemp(prefix="callradar_api_test_"))
atexit.register(shutil.rmtree, _TMP, True)

config.ASR_BACKEND = "mock"
config.LLM_BACKEND = "mock"
config.DB_PATH = _TMP / "callradar.db"
config.AUDIO_DIR = _TMP / "audio"
config.WORK_DIR = _TMP / "work"


def _seed_db() -> None:
    """Copy the shipped database so the read endpoints have real rows.

    Uses the backup API rather than copying the file: the live database runs in
    WAL mode, so the .db file on its own can be missing the most recent writes.
    """
    src = ROOT / "data" / "callradar.db"
    if not src.exists():
        return
    try:
        source = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
        dest = sqlite3.connect(str(config.DB_PATH))
        try:
            source.backup(dest)
        finally:
            source.close()
            dest.close()
    except sqlite3.Error:
        # A read-only open of a WAL database needs a writable -shm file, which
        # is not guaranteed. Copying the whole set works too: SQLite replays
        # the log on the copy the first time it is opened for writing.
        for suffix in ("", "-wal", "-shm"):
            f = src.with_name(src.name + suffix)
            if f.exists():
                shutil.copy2(f, config.DB_PATH.with_name(config.DB_PATH.name + suffix))


_seed_db()

# fastapi is optional for the pipeline itself, so a run under an interpreter
# that lacks it must report "not verified" rather than an import crash.
try:
    from fastapi.testclient import TestClient  # noqa: E402

    from api.main import app  # noqa: E402

    client = TestClient(app)
    _UNAVAILABLE = ""
except ImportError as exc:  # pragma: no cover - depends on the interpreter
    client = None
    _UNAVAILABLE = f"fastapi/httpx not installed for {sys.executable}: {exc}"


def _client():
    if _UNAVAILABLE:
        raise Skipped(_UNAVAILABLE)
    return client


def _sample() -> tuple[str, bytes, dict]:
    """(call_id, mp3 bytes, metadata dict) for the first staged sample pair."""
    for mp3 in sorted((ROOT / "data" / "audio").glob("*.mp3")):
        meta = ROOT / "data" / "metadata" / f"{mp3.stem}.json"
        if meta.exists():
            return mp3.stem, mp3.read_bytes(), json.loads(meta.read_text())
    raise Skipped("no audio/metadata pair staged in data/")


# ------------------------------------------------------- POST /api/process
def test_process_accepts_metadata_as_a_file():
    call_id, blob, meta = _sample()
    r = _client().post(
        "/api/process",
        files={
            "audio": ("sample.mp3", blob, "audio/mpeg"),
            "metadata": ("meta.json", json.dumps(meta).encode(), "application/json"),
        },
    )
    assert r.status_code == 200, f"file metadata rejected: {r.status_code} {r.text}"
    rec = r.json()
    assert rec["id"] == call_id, f"wrong call id: {rec['id']}"
    assert_contract(rec)


def test_process_accepts_metadata_as_a_string():
    """What the frontend actually sends: name="metadata", value = JSON text.

    This used to fail with "Expected UploadFile, received: <class 'str'>"
    because the parameter was declared as a file.
    """
    call_id, blob, meta = _sample()
    r = _client().post(
        "/api/process",
        files={"audio": ("sample.mp3", blob, "audio/mpeg")},
        data={"metadata": json.dumps(meta)},
    )
    assert r.status_code == 200, f"string metadata rejected: {r.status_code} {r.text}"
    rec = r.json()
    assert rec["id"] == call_id, f"wrong call id: {rec['id']}"
    assert_contract(rec)


def test_process_still_accepts_metadataJson():
    """The old field name has to keep working - something out there uses it."""
    _, blob, meta = _sample()
    r = _client().post(
        "/api/process",
        files={"audio": ("sample.mp3", blob, "audio/mpeg")},
        data={"metadataJson": json.dumps(meta)},
    )
    assert r.status_code == 200, f"metadataJson rejected: {r.status_code} {r.text}"
    assert_contract(r.json())


def test_process_without_metadata_is_400():
    _, blob, _ = _sample()
    r = _client().post("/api/process", files={"audio": ("sample.mp3", blob, "audio/mpeg")})
    assert r.status_code == 400, f"expected 400, got {r.status_code}: {r.text}"
    assert "metadata" in r.json()["detail"].lower()


def test_process_with_malformed_metadata_is_400_not_500():
    """A bad body is the caller's mistake; it must not read as a server fault.

    The message has to carry the parse error too - "invalid JSON" alone leaves
    the caller hunting for the character that broke it.
    """
    _, blob, _ = _sample()
    for label, kwargs in (
        ("string field", {"data": {"metadata": '{"sid": "x", oops}'}}),
        ("legacy field", {"data": {"metadataJson": "not json at all"}}),
        (
            "file part",
            {
                "files": {
                    "audio": ("sample.mp3", blob, "audio/mpeg"),
                    "metadata": ("meta.json", b'{"sid": ', "application/json"),
                }
            },
        ),
    ):
        files = kwargs.pop("files", {"audio": ("sample.mp3", blob, "audio/mpeg")})
        r = _client().post("/api/process", files=files, **kwargs)
        assert r.status_code == 400, (
            f"malformed metadata as a {label} returned {r.status_code}, "
            f"expected 400: {r.text}"
        )
        detail = r.json()["detail"]
        assert "JSON" in detail, f"{label}: no parse error in the message: {detail}"


def test_string_metadata_case_would_catch_a_regression():
    """Mutation check on test_process_accepts_metadata_as_a_string.

    Reinstate the old file-only behaviour and the request must stop returning
    200. Without this, a test that passes for the wrong reason - say, because
    the endpoint accepted a request it should have rejected - looks identical
    to one that verified the fix.
    """
    _client()  # skip cleanly before importing the app module
    import api.main as main

    _, blob, meta = _sample()
    payload = {
        "files": {"audio": ("sample.mp3", blob, "audio/mpeg")},
        "data": {"metadata": json.dumps(meta)},
    }

    real = main._form_text

    async def file_parts_only(value):
        return await real(value) if hasattr(value, "read") else None

    main._form_text = file_parts_only
    try:
        r = _client().post("/api/process", **payload)
        assert r.status_code == 400, (
            "the string-metadata path was mutated out and the request still "
            f"returned {r.status_code} - the test is not exercising it"
        )
    finally:
        main._form_text = real

    assert _client().post("/api/process", **payload).status_code == 200, (
        "the real implementation stopped working after the mutation was undone"
    )


# --------------------------------------------------------- GET /api/bundle
def test_bundle_matches_the_endpoints_it_replaces():
    r = _client().get("/api/bundle")
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"agents", "customers", "calls"}, f"bad keys: {set(body)}"

    if not body["calls"]:
        raise Skipped("database copy has no calls to compare")

    assert body["agents"] == _client().get("/api/agents").json(), "agents differ"
    assert body["customers"] == _client().get("/api/customers").json(), "customers differ"
    # limit=500 is the cap on /api/calls; /api/bundle is unlimited by default,
    # so compare against the largest page the other endpoint can return.
    assert body["calls"] == _client().get("/api/calls?limit=500").json(), "calls differ"

    for rec in body["calls"]:
        assert_contract(rec)


def test_bundle_limit_and_offset_apply_to_calls_only():
    full = _client().get("/api/bundle").json()
    if len(full["calls"]) < 2:
        raise Skipped("need at least 2 calls to test paging")

    page = _client().get("/api/bundle?limit=1&offset=1").json()
    assert len(page["calls"]) == 1, f"limit ignored: {len(page['calls'])} calls"
    assert page["calls"][0] == full["calls"][1], "offset ignored"
    assert page["agents"] == full["agents"], "limit must not touch agents"
    assert page["customers"] == full["customers"], "limit must not touch customers"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    skipped: list[str] = []
    for t in tests:
        try:
            t()
            passed += 1
            print(f"  PASS  {t.__name__}")
        except Skipped as e:
            skipped.append(t.__name__)
            print(f"  SKIP  {t.__name__}: {e}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")

    summary = f"\n{passed} passed"
    if skipped:
        summary += f", {len(skipped)} SKIPPED (not verified)"
    if failed:
        summary += f", {failed} FAILED"
    print(summary)
    if skipped:
        print(f"  not verified: {', '.join(skipped)}")
    raise SystemExit(1 if failed else 0)
