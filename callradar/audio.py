"""Audio preparation.

The single most important fact about this dataset: the two stereo channels are
recorded per speaker and are perfectly isolated (measured correlation ~0.0000 on
the sample, with both speakers active simultaneously only 0.9% of the time).
That means speaker attribution is free and 100% accurate - no diarisation model,
no pyannote, no clustering.

Second important fact: the channels are NOT balanced. On the sample the right
channel was ~17 dB quieter than the left, and the left clipped slightly. So each
channel is loudness-normalised independently before it reaches the ASR model.
Skipping this silently loses customer speech to voice-activity detection.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import wave
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from . import config


class AudioError(RuntimeError):
    pass


def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise AudioError(
            "ffmpeg/ffprobe not found on PATH. Install with:\n"
            "  macOS:  brew install ffmpeg\n"
            "  Ubuntu: sudo apt-get install ffmpeg"
        )


def probe(path: str | Path) -> dict:
    _require_ffmpeg()
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    info = json.loads(out.stdout)
    stream = next((s for s in info["streams"] if s.get("codec_type") == "audio"), None)
    if stream is None:
        raise AudioError(f"No audio stream in {path}")
    return {
        "channels": int(stream.get("channels", 0)),
        "sample_rate": int(stream.get("sample_rate", 0)),
        "duration_sec": float(info.get("format", {}).get("duration") or 0.0),
        "codec": stream.get("codec_name"),
    }


@dataclass
class ChannelStats:
    """Cheap per-channel metrics straight from the waveform - no AI involved.

    Because each speaker is on their own channel these are trustworthy, and
    they feed the needs-attention score for free.
    """

    agent_speech_frac: float
    customer_speech_frac: float
    dead_air_frac: float
    overlap_frac: float
    talk_ratio_agent: float
    agent_rms: float
    customer_rms: float


def _read_wav_mono(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def _speech_mask(sig: np.ndarray, frame: int = 1600, thresh: float = 0.01) -> np.ndarray:
    if sig.size < frame:
        return np.zeros(0, dtype=bool)
    frames = sig[: sig.size // frame * frame].reshape(-1, frame)
    return np.sqrt((frames ** 2).mean(axis=1)) > thresh


def channel_stats(agent_wav: Path, customer_wav: Path) -> ChannelStats:
    a, c = _read_wav_mono(agent_wav), _read_wav_mono(customer_wav)
    ma, mc = _speech_mask(a), _speech_mask(c)
    n = min(ma.size, mc.size)
    ma, mc = ma[:n], mc[:n]
    if n == 0:
        return ChannelStats(0, 0, 1, 0, 0, 0, 0)

    either = float((ma | mc).mean())
    total_talk = float(ma.mean() + mc.mean()) or 1.0
    return ChannelStats(
        agent_speech_frac=round(float(ma.mean()), 4),
        customer_speech_frac=round(float(mc.mean()), 4),
        dead_air_frac=round(1.0 - either, 4),
        overlap_frac=round(float((ma & mc).mean()), 4),
        talk_ratio_agent=round(float(ma.mean()) / total_talk, 4),
        agent_rms=round(float(np.sqrt((a ** 2).mean())), 5),
        customer_rms=round(float(np.sqrt((c ** 2).mean())), 5),
    )


@dataclass
class PreparedAudio:
    agent_wav: Path
    customer_wav: Path
    duration_sec: float
    was_mono: bool
    stats: ChannelStats

    def as_dict(self) -> dict:
        return {
            "duration_sec": self.duration_sec,
            "was_mono": self.was_mono,
            **asdict(self.stats),
        }


def prepare(src: str | Path, work_dir: str | Path, call_id: str) -> PreparedAudio:
    """Split stereo into two loudness-normalised 16 kHz mono WAVs.

    Falls back gracefully if a file turns out to be mono: both "channels" then
    point at the same audio and speaker labels come from the ASR merge order
    only (flagged via `was_mono` so you can exclude those calls if needed).
    """
    _require_ffmpeg()
    src = Path(src)
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    info = probe(src)
    agent_wav = work / f"{call_id}_agent.wav"
    customer_wav = work / f"{call_id}_customer.wav"

    norm = f"{config.NORMALISE_FILTER},aresample={config.ASR_SAMPLE_RATE}"

    if info["channels"] >= 2:
        left_out, right_out = (
            (agent_wav, customer_wav)
            if config.LEFT_CHANNEL_SPEAKER == "agent"
            else (customer_wav, agent_wav)
        )
        cmd = [
            "ffmpeg", "-v", "error", "-y", "-i", str(src),
            "-filter_complex",
            f"[0:a]channelsplit=channel_layout=stereo[l][r];"
            f"[l]{norm}[la];[r]{norm}[ra]",
            "-map", "[la]", "-ac", "1", str(left_out),
            "-map", "[ra]", "-ac", "1", str(right_out),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        was_mono = False
    else:
        cmd = [
            "ffmpeg", "-v", "error", "-y", "-i", str(src),
            "-af", norm, "-ac", "1", str(agent_wav),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        shutil.copy(agent_wav, customer_wav)
        was_mono = True

    return PreparedAudio(
        agent_wav=agent_wav,
        customer_wav=customer_wav,
        duration_sec=info["duration_sec"],
        was_mono=was_mono,
        stats=channel_stats(agent_wav, customer_wav),
    )
