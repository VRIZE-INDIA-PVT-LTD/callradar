#!/usr/bin/env python3
"""Run this FIRST, before the bulk job.

Checks the three assumptions the whole architecture rests on:
  1. files really are stereo
  2. total audio duration (drives cost and runtime estimates)
  3. per-channel energy (a hint only - see below)

This script does NOT decide the channel assignment for you. Confirm it from the
transcript instead: `make sample && python scripts/show_transcript.py`.

    python scripts/preflight.py --sample 20
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from callradar import audio, config
from callradar.audio import _read_wav_mono, _speech_mask
from callradar.metadata import load_metadata


def _first_energy_frame(wav: Path) -> int:
    """Index of the first frame above the energy floor, or a large sentinel.

    A hint only. It cannot tell you who the agent is - a channel can carry
    tone or noise long before anybody speaks on it.
    """
    mask = _speech_mask(_read_wav_mono(wav))
    return int(np.argmax(mask)) if mask.any() else 10**9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", type=Path, default=config.AUDIO_DIR)
    ap.add_argument("--metadata-dir", type=Path, default=config.METADATA_DIR)
    ap.add_argument("--sample", type=int, default=20)
    args = ap.parse_args()

    files = sorted(args.audio_dir.glob("*.mp3"))
    if not files:
        print(f"No mp3 files in {args.audio_dir}")
        return 1

    print(f"Found {len(files)} audio files\n")

    # ---- 1. duration + channel layout across the whole set ----------------
    total = 0.0
    channels = Counter()
    rates = Counter()
    for p in files:
        try:
            info = audio.probe(p)
        except Exception as exc:
            print(f"  probe failed {p.name}: {exc}")
            continue
        total += info["duration_sec"]
        channels[info["channels"]] += 1
        rates[info["sample_rate"]] += 1

    n = max(sum(channels.values()), 1)
    print(f"Total audio      : {total/3600:.2f} hours ({total/60:.0f} min)")
    print(f"Average call     : {total/n:.1f} s")
    print(f"Channel layouts  : {dict(channels)}")
    print(f"Sample rates     : {dict(rates)}")
    mono = sum(v for k, v in channels.items() if k < 2)
    if mono:
        print(f"  WARNING: {mono} mono file(s) - speaker split will fall back for these")

    # ---- 2. cost estimate --------------------------------------------------
    hours = total / 3600
    print(f"\nEstimated Groq Whisper cost (both channels): "
          f"${hours * 2 * 0.04:.2f} at $0.04/audio-hour")

    # ---- 3. per-channel energy (hint only) ---------------------------------
    print(f"\nMeasuring per-channel first-speech energy on {args.sample} files "
          f"(current setting: left={config.LEFT_CHANNEL_SPEAKER})")
    agree = disagree = 0
    with tempfile.TemporaryDirectory() as work:
        for p in files[: args.sample]:
            try:
                prep = audio.prepare(p, work, p.stem)
                first_a = _first_energy_frame(prep.agent_wav)
                first_c = _first_energy_frame(prep.customer_wav)
                if first_a < first_c:
                    agree += 1
                else:
                    disagree += 1
            except Exception as exc:
                print(f"  {p.name}: {exc}")

    print(f"  channel labelled 'agent' had energy first    : {agree}")
    print(f"  channel labelled 'customer' had energy first : {disagree}")
    print("\n  NOTE: this is only a weak hint, NOT a verdict. Both sides very often")
    print("  start at the same instant on these recordings, so 'who speaks first'")
    print("  cannot decide the channel assignment on its own.")
    print("\n  Confirm it from the words instead:")
    print("      make sample && python scripts/show_transcript.py")
    print("  Whoever gives the greeting ('thank you for calling ...', 'my name is ...')")
    print("  is the agent. Only flip LEFT_CHANNEL_SPEAKER in .env if that greeting")
    print("  shows up on the channel labelled 'customer'.")

    # ---- 4. metadata sanity ------------------------------------------------
    metas = sorted(args.metadata_dir.glob("*.json"))
    print(f"\nMetadata files   : {len(metas)}")
    unmatched = [p.stem for p in files if not (args.metadata_dir / f"{p.stem}.json").exists()]
    if unmatched:
        print(f"  WARNING: {len(unmatched)} audio files have no metadata "
              f"(e.g. {unmatched[:3]})")

    sessions, rated = Counter(), 0
    for m in metas[:200]:
        try:
            raw = load_metadata(m)
        except Exception:
            continue
        sessions[raw.get("session")] += 1
        if ((raw.get("caller") or {}).get("survey_response") or {}).get("data", {}).get(
            "partner_rating"
        ):
            rated += 1
    print(f"  sessions (sample): {dict(sessions.most_common(5))}")
    print(f"  with partner_rating: {rated}/{min(len(metas),200)} "
          f"(used to validate the attention score)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
