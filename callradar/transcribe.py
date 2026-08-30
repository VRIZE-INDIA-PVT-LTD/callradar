"""Speech to text, per channel, then merged into numbered turns.

Each channel is transcribed independently. Because a channel is silent while
the other person talks, voice-activity detection skips most of it - so
transcribing both channels costs roughly the same as transcribing the mono mix
once (63% of the sample file was silence).

Backends:
  groq            - hosted Whisper large-v3-turbo. ~$0.04/audio-hour, ~200x
                    real-time. Default, and the same backend used by the live
                    API so local results and demo results are identical.
  faster_whisper  - runs the open-weights model on your own machine. Free, no
                    network, but needs a GPU to be quick.
  mock            - deterministic fake transcript, no key needed. Lets you
                    exercise the whole pipeline end to end offline.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

from . import config


@dataclass
class Segment:
    speaker: str
    start: float
    end: float
    text: str
    # True when these times came from word-level alignment, which is accurate
    # enough that waveform snapping is unnecessary (and could only do harm).
    from_words: bool = False


@dataclass
class Turn:
    """A numbered turn. The id is what the LLM is allowed to cite."""

    id: int
    speaker: str
    startSec: float
    endSec: float
    text: str

    def as_public(self) -> dict:
        # the frontend's TranscriptTurn has no id field
        return {
            "speaker": self.speaker,
            "startSec": self.startSec,
            "endSec": self.endSec,
            "text": self.text,
        }


# --------------------------------------------------------------------- Groq
_groq_client = None


def _groq():
    global _groq_client
    if _groq_client is None:
        from groq import Groq

        if not config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set (see .env.example)")
        _groq_client = Groq(api_key=config.GROQ_API_KEY)
    return _groq_client


def _asr_groq(wav: Path, speaker: str) -> list[Segment]:
    kwargs: dict[str, Any] = {}
    if config.ASR_PROMPT:
        kwargs["prompt"] = config.ASR_PROMPT
    with open(wav, "rb") as fh:
        resp = _groq().audio.transcriptions.create(
            file=(wav.name, fh.read()),
            model=config.ASR_MODEL_GROQ,
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
            language=config.ASR_LANGUAGE,
            temperature=0.0,
            **kwargs,
        )
    data = resp if isinstance(resp, dict) else resp.model_dump()
    words = data.get("words") or []
    out: list[Segment] = []
    for seg in data.get("segments") or []:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        # Whisper's classic failure on silence is looping a phrase. Drop
        # segments the model itself is unsure contain speech.
        if float(seg.get("no_speech_prob", 0.0)) > 0.6:
            continue
        start, end = float(seg["start"]), float(seg["end"])
        # Tighten boundaries using the words that fall inside them.
        inner = [w for w in words if start - 0.05 <= float(w.get("start", -1)) <= end + 0.05]
        if inner:
            start = min(float(w["start"]) for w in inner)
            end = max(float(w["end"]) for w in inner)
        out.append(Segment(speaker, start, end, text, from_words=bool(inner)))
    return out


# ----------------------------------------------------------- faster-whisper
_local_model = None


def _asr_faster_whisper(wav: Path, speaker: str) -> list[Segment]:
    global _local_model
    from faster_whisper import WhisperModel

    if _local_model is None:
        try:
            _local_model = WhisperModel(
                config.ASR_MODEL_LOCAL, device="cuda", compute_type="int8_float16"
            )
        except Exception:
            _local_model = WhisperModel(
                config.ASR_MODEL_LOCAL, device="cpu", compute_type="int8"
            )

    segments, _ = _local_model.transcribe(
        str(wav),
        language=config.ASR_LANGUAGE,
        vad_filter=True,                 # skip silence - this is the speed win
        condition_on_previous_text=False,  # stops repetition loops on silence
        word_timestamps=True,              # segment bounds alone are too coarse
        initial_prompt=config.ASR_PROMPT or None,
    )
    out: list[Segment] = []
    for s in segments:
        text = (s.text or "").strip()
        if not text or getattr(s, "no_speech_prob", 0.0) > 0.6:
            continue
        words = list(getattr(s, "words", None) or [])
        if words:
            start = min(float(w.start) for w in words)
            end = max(float(w.end) for w in words)
            out.append(Segment(speaker, start, end, text, from_words=True))
        else:
            out.append(Segment(speaker, float(s.start), float(s.end), text))
    return out


# --------------------------------------------------------------------- mock
_MOCK_AGENT = [
    "Thank you for calling Northbank, my name is Robert, how can I help you today?",
    "I'm sorry to hear that. Can I take your account number please?",
    "I can see the transaction here. Let me raise a dispute for you.",
    "I have submitted the case and you should hear from back office in two business days.",
]
_MOCK_CUSTOMER = [
    "Hi, yes, I've been charged twice for the same transaction.",
    "This is the third time I've called about this and nobody has fixed it.",
    "Two business days? That is exactly why I am losing trust in your bank.",
]


def _asr_mock(wav: Path, speaker: str) -> list[Segment]:
    """Deterministic fake transcript so the pipeline is testable with no keys."""
    lines = _MOCK_AGENT if speaker == "agent" else _MOCK_CUSTOMER
    seed = int(hashlib.md5(wav.name.encode()).hexdigest()[:8], 16)
    offset = 0.5 if speaker == "agent" else 5.0
    step = 13.0
    return [
        Segment(speaker, offset + i * step, offset + i * step + 4.5, line)
        for i, line in enumerate(lines)
        if (seed + i) >= 0
    ]


_BACKENDS = {
    "groq": _asr_groq,
    "faster_whisper": _asr_faster_whisper,
    "mock": _asr_mock,
}


def transcribe_channel(wav: Path, speaker: str, backend: str | None = None) -> list[Segment]:
    backend = backend or config.ASR_BACKEND
    if backend not in _BACKENDS:
        raise ValueError(f"Unknown ASR_BACKEND {backend!r}; pick one of {list(_BACKENDS)}")
    return _BACKENDS[backend](wav, speaker)


# -------------------------------------------------------------------- merge
def merge_turns(segments: Iterable[Segment]) -> list[Turn]:
    """Sort both channels by time and group consecutive same-speaker segments.

    Turn ids are assigned here and are the ONLY thing the LLM may cite as
    evidence. Timestamps are resolved from these ids afterwards, in code, which
    is what makes fabricated timestamps structurally impossible.
    """
    segs = sorted(segments, key=lambda s: (s.start, s.end, s.speaker))
    turns: list[Turn] = []
    for s in segs:
        if (
            turns
            and turns[-1].speaker == s.speaker
            and s.start - turns[-1].endSec <= config.TURN_MERGE_GAP_SEC
        ):
            turns[-1].endSec = round(s.end, 2)
            turns[-1].text = f"{turns[-1].text} {s.text}".strip()
        else:
            turns.append(
                Turn(
                    id=len(turns) + 1,
                    speaker=s.speaker,
                    startSec=round(s.start, 2),
                    endSec=round(s.end, 2),
                    text=s.text,
                )
            )
    for i, t in enumerate(turns, start=1):
        t.id = i
    return turns


def transcribe_call(
    agent_wav: Path, customer_wav: Path, backend: str | None = None
) -> list[Turn]:
    backend = backend or config.ASR_BACKEND
    agent_segs = transcribe_channel(agent_wav, "agent", backend)
    cust_segs = transcribe_channel(customer_wav, "customer", backend)
    # The mock backend fabricates its timestamps by design, so it would trip
    # this on every keyless run - and a warning that always fires is a warning
    # nobody reads.
    channels = () if backend == "mock" else (
        ("agent", agent_segs), ("customer", cust_segs)
    )
    for name, segs in channels:
        if segs and not any(sg.from_words for sg in segs):
            import warnings
            warnings.warn(
                f"{name} channel has no word-level timestamps; turn ordering may be "
                f"wrong when both channels report the same start time. "
                f"Check ASR backend configuration.",
                RuntimeWarning,
            )
    return merge_turns(agent_segs + cust_segs)


def render_for_llm(turns: list[Turn]) -> str:
    """Numbered transcript, deliberately WITHOUT timestamps.

    The model never sees a clock value, so it cannot invent one. It cites turn
    ids; we resolve those to real timestamps ourselves.
    """
    return "\n".join(f"[{t.id}] {t.speaker.upper()}: {t.text}" for t in turns)


def turns_to_public(turns: list[Turn]) -> list[dict]:
    return [t.as_public() for t in turns]


def turns_to_dicts(turns: list[Turn]) -> list[dict]:
    return [asdict(t) for t in turns]
