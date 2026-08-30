#!/usr/bin/env python3
"""Bulk-process the whole dataset locally, then commit the resulting DB.

Resumability is not optional here: a 1,441-file run WILL be interrupted.
Every call's status is checkpointed in SQLite, so re-running skips whatever is
already done.

    python scripts/run_batch.py --limit 5          # smoke test
    python scripts/run_batch.py                    # the real thing
    python scripts/run_batch.py --retry-failed     # only the ones that broke
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from callradar import config, db
from callradar.metadata import load_metadata
from callradar.pipeline import process_call, store

_db_lock = threading.Lock()


def discover(audio_dir: Path, meta_dir: Path) -> list[tuple[str, Path, Path]]:
    """Pair audio with metadata by call id (the filename stem)."""
    pairs = []
    for audio_path in sorted(audio_dir.glob("*.mp3")):
        call_id = audio_path.stem
        meta_path = meta_dir / f"{call_id}.json"
        if meta_path.exists():
            pairs.append((call_id, audio_path, meta_path))
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", type=Path, default=config.AUDIO_DIR)
    ap.add_argument("--metadata-dir", type=Path, default=config.METADATA_DIR)
    ap.add_argument("--limit", type=int, default=None, help="process only the first N")
    ap.add_argument("--workers", type=int, default=config.BATCH_WORKERS)
    ap.add_argument("--asr", default=None, help="override ASR_BACKEND")
    ap.add_argument("--llm", default=None, help="override LLM_BACKEND")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--force", action="store_true", help="reprocess even if done")
    args = ap.parse_args()

    conn = db.connect()
    pairs = discover(args.audio_dir, args.metadata_dir)
    if not pairs:
        print(f"No matched audio/metadata pairs in {args.audio_dir} + {args.metadata_dir}")
        print("Expect audio/<id>.mp3 alongside metadata/<id>.json")
        return 1

    already = set() if args.force else db.done_ids(conn)
    if args.retry_failed:
        failed = {
            r["call_id"]
            for r in conn.execute("SELECT call_id FROM jobs WHERE status='failed'")
        }
        pairs = [p for p in pairs if p[0] in failed]
    else:
        pairs = [p for p in pairs if p[0] not in already]

    if args.limit:
        pairs = pairs[: args.limit]

    total = len(pairs)
    print(f"{total} calls to process  (workers={args.workers}, "
          f"asr={args.asr or config.ASR_BACKEND}, llm={args.llm or config.LLM_BACKEND})")
    if total == 0:
        print("Nothing to do. Use --force to reprocess.")
        return 0

    done = failed_n = review_n = 0
    started = time.time()

    def work(item):
        call_id, audio_path, meta_path = item
        raw = load_metadata(meta_path)
        result = process_call(
            audio_path, raw, conn=conn, asr_backend=args.asr, llm_backend=args.llm
        )
        with _db_lock:
            store(conn, result, raw)
            db.set_job(conn, call_id, "done")
        return call_id, result

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, item): item for item in pairs}
        for fut in as_completed(futures):
            call_id = futures[fut][0]
            try:
                _, result = fut.result()
                done += 1
                if result.needs_review:
                    review_n += 1
            except Exception as exc:
                failed_n += 1
                with _db_lock:
                    db.set_job(conn, call_id, "failed", f"{exc}\n{traceback.format_exc()}")
                print(f"  FAILED {call_id}: {exc}")
            n = done + failed_n
            if n % 10 == 0 or n == total:
                rate = n / max(time.time() - started, 1e-6)
                eta = (total - n) / rate if rate else 0
                print(f"  {n}/{total}  ok={done} failed={failed_n} "
                      f"review={review_n}  {rate:.1f}/s  eta={eta/60:.1f}m")

    print(f"\nFinished in {(time.time()-started)/60:.1f} min")
    print(f"  processed     : {done}")
    print(f"  failed        : {failed_n}")
    print(f"  needs review  : {review_n}")
    print(f"  database      : {config.DB_PATH}")
    if failed_n:
        print("\nRe-run failures with:  python scripts/run_batch.py --retry-failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
