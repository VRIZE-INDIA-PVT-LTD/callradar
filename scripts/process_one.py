#!/usr/bin/env python3
"""Process one call and print the CallRecord. Useful for eyeballing output.

    python scripts/process_one.py data/audio/abc.mp3 data/metadata/abc.json
    python scripts/process_one.py a.mp3 a.json --asr mock --llm mock   # no keys
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from callradar import db
from callradar.metadata import load_metadata
from callradar.pipeline import process_call, store


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    ap.add_argument("metadata", type=Path)
    ap.add_argument("--asr", default=None)
    ap.add_argument("--llm", default=None)
    ap.add_argument("--save", action="store_true", help="also write to the database")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    raw = load_metadata(args.metadata)
    conn = db.connect() if args.save else None
    result = process_call(
        args.audio, raw, conn=conn, asr_backend=args.asr, llm_backend=args.llm
    )

    print(json.dumps(result.record, indent=2, ensure_ascii=False))

    if args.debug:
        print("\n--- debug ---", file=sys.stderr)
        print(json.dumps(result.debug, indent=2), file=sys.stderr)
    if result.warnings:
        print("\n--- validation warnings ---", file=sys.stderr)
        for w in result.warnings:
            print(f"  - {w}", file=sys.stderr)

    if conn is not None:
        store(conn, result, raw)
        print(f"\nsaved to {conn}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
