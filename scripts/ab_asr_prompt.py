#!/usr/bin/env python3
"""A/B test: does ASR_PROMPT change the digits Whisper transcribes?

Throwaway experiment, not a pipeline feature. Delete it once the question is
settled.

Whisper's `prompt` biases decoding - it primes vocabulary and style. That is
what makes it fix "Papa Valley" -> "Harper Valley", and it is also what makes
it dangerous: a prompt containing "Times like 2:30 PM" can push an ambiguous
utterance toward writing 2:30, producing a confident transcript of something
the model did not actually hear.

This does NOT judge what was said - nobody here can listen. It tests one
falsifiable question: holding the audio fixed, do the transcribed digits change
when the prompt changes?

Read the output like this:
  * D_current says 2:30, A/B/C say 3:30   -> the prompt injected the digit.
  * every variant says the same thing     -> the prompt is not the cause.
  * values differ BETWEEN REPEATS of one
    variant                               -> sampling noise, not the prompt.

    python scripts/ab_asr_prompt.py
    python scripts/ab_asr_prompt.py --call-id 0a37e848a9c04e0c --repeats 5
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from callradar import audio, config
from callradar.transcribe import merge_turns, transcribe_channel

# 2:30, 2.30, 2:30 PM, 2.30 p.m.
TIME_RE = re.compile(r"\d{1,2}[:.]\d{2}(?:\s*[AaPp]\.?[Mm]\.?)?")

_TIME_HEAD = re.compile(r"\s*(\d{1,2})[:.](\d{2})")


def normalise_time(token: str) -> str | None:
    """Canonical "H:MM AM/PM" so formatting differences stop reading as changes.

    Whisper writes the same clock value several ways depending on the prompt -
    "2.30 p.m.", "2:30 PM", "2.30 PM." - and comparing raw strings reports a
    prompt effect where there is only punctuation. Only the hour, the minute
    and the meridiem carry meaning; everything else is noise.
    """
    m = _TIME_HEAD.match(token)
    if not m:
        return None
    hour, minute = int(m.group(1)), int(m.group(2))
    tail = re.sub(r"[^a-z]", "", token[m.end():].lower())
    meridiem = "PM" if tail.startswith("p") else "AM" if tail.startswith("a") else ""
    return f"{hour}:{minute:02d}" + (f" {meridiem}" if meridiem else "")


def normalise_all(tokens) -> tuple[str, ...]:
    return tuple(sorted({n for n in (normalise_time(t) for t in tokens) if n}))


VARIANTS: dict[str, str] = {
    "A_none": "",
    "B_nounonly": "Harper Valley National Bank.",
    "C_days": (
        "Harper Valley National Bank. Monday, Tuesday, Wednesday, Thursday, "
        "Friday, Saturday, Sunday."
    ),
    "D_current": config.ASR_PROMPT,
}


def _env_file_prompt() -> str | None:
    """ASR_PROMPT as literally written in .env, if present.

    config.ASR_PROMPT only reflects .env when python-dotenv is installed. If it
    is not, D_current silently becomes the code default and the experiment
    tests the wrong string - so check and say so rather than reporting a clean
    result from a broken setup.
    """
    env = Path(__file__).resolve().parent.parent / ".env"
    if not env.exists():
        return None
    for line in env.read_text().splitlines():
        line = line.strip()
        if line.startswith("ASR_PROMPT="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def times_in(turns, speaker: str, near: float, window: float) -> list[str]:
    """Time-like tokens on `speaker` turns starting within `window` of `near`."""
    found: list[str] = []
    for t in turns:
        if t.speaker != speaker or abs(t.startSec - near) > window:
            continue
        found.extend(m.group(0).strip() for m in TIME_RE.finditer(t.text))
    return found


def anchor_text(turns, speaker: str, near: float, window: float) -> list[str]:
    """Raw text of the anchor turns, matched or not.

    The regex only catches well-formed times, and the failure mode under
    investigation is a DROPPED digit ("Tuesday. 30 p.m."). Keeping the raw text
    means a run that matches nothing is still readable evidence.
    """
    return [
        t.text
        for t in turns
        if t.speaker == speaker and abs(t.startSec - near) <= window
    ]


def all_times(turns) -> list[dict]:
    out = []
    for t in turns:
        for m in TIME_RE.finditer(t.text):
            out.append(
                {
                    "speaker": t.speaker,
                    "startSec": t.startSec,
                    "match": m.group(0).strip(),
                    "text": t.text,
                }
            )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--call-id", default="0a9e9e9be6634e38")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--backend", default=None, help="override ASR backend (e.g. mock)")
    ap.add_argument("--agent-at", type=float, default=34.0,
                    help="approx start of the agent confirmation turn")
    ap.add_argument("--customer-at", type=float, default=21.0,
                    help="approx start of the mumbled customer turn")
    ap.add_argument("--window", type=float, default=10.0,
                    help="+/- seconds around those anchors")
    ap.add_argument("--out", type=Path, help="also write raw results as JSON")
    ap.add_argument("--from-json", type=Path,
                    help="re-run the summary over a previous --out file; no Groq calls")
    args = ap.parse_args()

    if args.from_json:
        saved = json.loads(args.from_json.read_text())
        print(f"re-summarising {args.from_json} (no Groq calls)\n")
        return summarise(saved, args)

    backend = args.backend or config.ASR_BACKEND
    src = config.AUDIO_DIR / f"{args.call_id}.mp3"
    if not src.exists():
        print(f"No audio at {src}")
        return 1

    print(f"call        : {args.call_id}")
    print(f"backend     : {backend}")
    print(f"repeats     : {args.repeats}   (4 variants x {args.repeats} x 2 channels "
          f"= {4 * args.repeats * 2} Groq calls)")
    print("\nprompt variants under test:")
    for name, val in VARIANTS.items():
        print(f"  {name:<11} {val!r}")

    on_disk = _env_file_prompt()
    if on_disk is not None and on_disk != config.ASR_PROMPT:
        print("\n  !! WARNING: .env sets ASR_PROMPT to a DIFFERENT string than the one")
        print("     resolved into config.ASR_PROMPT (python-dotenv may not be installed).")
        print(f"     .env  : {on_disk!r}")
        print(f"     config: {config.ASR_PROMPT!r}")
        print("     D_current is therefore NOT what the pipeline actually used.")
        print("     Install python-dotenv (pip install python-dotenv) and re-run,")
        print("     or this experiment tests the wrong prompt.")

    results: dict[str, list[dict]] = defaultdict(list)
    original_prompt = config.ASR_PROMPT

    with tempfile.TemporaryDirectory() as work:
        # Prepared once: identical audio for every variant, so the prompt is the
        # only thing that changes between runs.
        prep = audio.prepare(src, work, args.call_id)
        try:
            for name, prompt in VARIANTS.items():
                for rep in range(args.repeats):
                    config.ASR_PROMPT = prompt
                    segs = transcribe_channel(prep.agent_wav, "agent", backend)
                    segs += transcribe_channel(prep.customer_wav, "customer", backend)
                    turns = merge_turns(segs)
                    results[name].append(
                        {
                            "repeat": rep,
                            "agent": times_in(turns, "agent", args.agent_at, args.window),
                            "customer": times_in(
                                turns, "customer", args.customer_at, args.window
                            ),
                            "agent_text": anchor_text(
                                turns, "agent", args.agent_at, args.window
                            ),
                            "customer_text": anchor_text(
                                turns, "customer", args.customer_at, args.window
                            ),
                            "all_times": all_times(turns),
                        }
                    )
                    print(f"  ran {name} repeat {rep + 1}/{args.repeats}", flush=True)
        finally:
            config.ASR_PROMPT = original_prompt

    # ------------------------------------------------------------------ table
    print(f"\n{'variant':<12} {'rep':<4} {'agent @~' + str(args.agent_at) + 's':<26} "
          f"customer @~{args.customer_at}s")
    print("-" * 78)
    for name in VARIANTS:
        for r in results[name]:
            a = ", ".join(r["agent"]) or "-"
            c = ", ".join(r["customer"]) or "-"
            print(f"{name:<12} {r['repeat'] + 1:<4} {a:<26} {c}")

    summarise(results, args)

    if args.out:
        args.out.write_text(json.dumps(dict(results), indent=2))
        print(f"\nraw results -> {args.out}")
    return 0


def summarise(results, args) -> int:
    """Compare NORMALISED times across variants. No Groq calls."""
    print("\nsummary - distinct agent-side time values per variant (normalised)")
    agent_sets: dict[str, set[str]] = {}
    customer_sets: dict[str, set[str]] = {}
    noisy: list[str] = []
    for name in VARIANTS:
        per_repeat = [normalise_all(r["agent"]) for r in results[name]]
        vals = {v for rep in per_repeat for v in rep}
        agent_sets[name] = vals
        customer_sets[name] = {
            v for r in results[name] for v in normalise_all(r["customer"])
        }
        unstable = len(set(per_repeat)) > 1
        if unstable:
            noisy.append(name)
        raw = sorted({t for r in results[name] for t in r["agent"]})
        print(f"  {name:<12} {sorted(vals) if vals else '(none found)'}"
              f"{'   <-- VARIES BETWEEN REPEATS' if unstable else ''}")
        if len(raw) > 1:
            print(f"               (raw spellings, same value: {raw})")

    print("\nsummary - distinct customer-side time values per variant (normalised)")
    for name in VARIANTS:
        vals = customer_sets[name]
        print(f"  {name:<12} {sorted(vals) if vals else '(none found - no time transcribed)'}")
    recovered = [n for n, v in customer_sets.items() if v]
    dropped = [n for n, v in customer_sets.items() if not v]
    if recovered and dropped:
        print(f"\n  NOTE: the customer's time was transcribed under {', '.join(recovered)}")
        print(f"        but DROPPED ENTIRELY under {', '.join(dropped)}.")
        print("        The prompt is not changing this digit, it is deciding whether")
        print("        the utterance gets transcribed at all. That is a recall effect,")
        print("        and it is the opposite of the contamination we were testing for.")

    distinct = {frozenset(v) for v in agent_sets.values()}
    print()
    if not any(agent_sets.values()):
        # Never let "found nothing" masquerade as "found no difference".
        print("VERDICT: INCONCLUSIVE - no time-like token was found on any agent turn")
        print(f"  within +/-{args.window}s of {args.agent_at}s, under any variant.")
        print("  Nothing was measured, so nothing is ruled in or out. Check that:")
        print("    - the backend really is an ASR backend (mock has no times in it)")
        print("    - --agent-at matches where the confirmation turn actually starts")
        print("      (widen with --window, or read the raw text via --out)")
        sample = results[next(iter(VARIANTS))][0]["agent_text"]
        if sample:
            print("  agent turns seen in that window (first run):")
            for txt in sample[:3]:
                print(f"    {txt[:88]}")
        return 2
    if noisy:
        print("VERDICT: values differ between repeats of the same variant "
              f"({', '.join(noisy)}).")
        print("  That is sampling noise, not a prompt effect. Pin the ASR call to")
        print("  temperature 0 and re-run before drawing any conclusion about the prompt.")
    elif len(distinct) > 1:
        print("VERDICT: the transcribed digits DEPEND ON THE PROMPT.")
        for name, vals in agent_sets.items():
            print(f"    {name:<12} -> {sorted(vals) if vals else '(none)'}")
        print("  The prompt is writing the digit rather than the model hearing it.")
        print("  Recommend removing example times from ASR_PROMPT.")
    else:
        print("VERDICT: every variant produced the same agent-side time value.")
        print("  ASR_PROMPT is NOT the cause. Whisper genuinely transcribes this")
        print("  utterance the same way with no prompt at all. The transcript may")
        print("  still be wrong, but only re-listening to the audio settles that.")

    if args.out:
        args.out.write_text(json.dumps(dict(results), indent=2))
        print(f"\nraw results -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
