"""Central configuration. Everything tunable lives here or in .env."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv is optional
    pass

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- providers
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ASR backend: "groq" (hosted, cheap, fast) | "faster_whisper" (local, free) | "mock"
ASR_BACKEND = os.getenv("ASR_BACKEND", "groq")
ASR_MODEL_GROQ = os.getenv("ASR_MODEL_GROQ", "whisper-large-v3-turbo")
ASR_MODEL_LOCAL = os.getenv("ASR_MODEL_LOCAL", "large-v3-turbo")
ASR_LANGUAGE = os.getenv("ASR_LANGUAGE", "en")

# Domain vocabulary passed to Whisper as a decoding hint. Cheap and effective on
# 8 kHz telephone audio: without it the bank name came back as "Papa Valley".
ASR_PROMPT = os.getenv(
    "ASR_PROMPT",
    "Harper Valley National Bank. Customer service call about appointments, "
    "transfers, balances, and card services.",
)

# LLM backend: "groq" | "mock"
LLM_BACKEND = os.getenv("LLM_BACKEND", "groq")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))

# ---------------------------------------------------------------- audio
# Channel mapping. The brief says left=agent, right=customer.
# Flip here if a sanity check on real files proves otherwise.
LEFT_CHANNEL_SPEAKER = os.getenv("LEFT_CHANNEL_SPEAKER", "agent")
RIGHT_CHANNEL_SPEAKER = os.getenv("RIGHT_CHANNEL_SPEAKER", "customer")

# Per-channel normalisation.
#
# Measured on the sample file: the customer channel was 17.3 dB quieter than
# the agent channel. Filters compared on that file (speech-only RMS):
#
#   loudnorm only            -> 11.1 dB gap remaining   (not good enough)
#   dynaudnorm                ->  0.7 dB gap, SNR 40 dB
#   speechnorm + loudnorm     ->  0.3 dB gap, SNR 46 dB  <- chosen
#
# loudnorm alone under-corrects because it measures integrated loudness across
# the WHOLE file, and the quiet channel is ~89% silence, which drags its
# measurement down. speechnorm targets speech segments directly, then loudnorm
# sets a consistent output level.
#
# The noise floor matters as much as the gap: over-amplified silence is what
# makes Whisper hallucinate loops. This chain keeps SNR above 46 dB.
NORMALISE_FILTER = os.getenv(
    "NORMALISE_FILTER",
    "speechnorm=e=12.5:r=0.0001:l=1,loudnorm=I=-16:TP=-1.5",
)
ASR_SAMPLE_RATE = 16000

# Gap (seconds) below which two consecutive segments from the same speaker
# get merged into a single turn.
TURN_MERGE_GAP_SEC = float(os.getenv("TURN_MERGE_GAP_SEC", "0.8"))

# ---------------------------------------------------------------- mood
# "turn"   -> one mood point per customer turn (minute = startSec/60, 2dp).
#             Better for the short calls in this dataset (~45-60s).
# "minute" -> integer minute buckets, averaged (matches the original mock).
MOOD_TIMELINE_MODE = os.getenv("MOOD_TIMELINE_MODE", "turn")

# Emitted when the model reports no mood shift. Frontend should hide the
# marker when it sees this value.
NO_MOOD_SHIFT_SEC = -1

# ---------------------------------------------------------------- storage
DB_PATH = Path(os.getenv("DB_PATH", ROOT / "data" / "callradar.db"))
AUDIO_DIR = Path(os.getenv("AUDIO_DIR", ROOT / "data" / "audio"))
METADATA_DIR = Path(os.getenv("METADATA_DIR", ROOT / "data" / "metadata"))
WORK_DIR = Path(os.getenv("WORK_DIR", ROOT / "data" / "work"))

# ---------------------------------------------------------------- scoring
REPEAT_CONTACT_WINDOW_DAYS = int(os.getenv("REPEAT_CONTACT_WINDOW_DAYS", "7"))
LONG_HANDLE_TIME_SEC = float(os.getenv("LONG_HANDLE_TIME_SEC", "90"))
# Turn-taking with natural gaps puts these calls around 40-45% silence, so the
# old 0.25 default fired on literally every call and contributed nothing but a
# constant offset. Set this above your corpus norm - check the distribution
# after a bulk run with scripts/validate_scores.py.
DEAD_AIR_THRESHOLD = float(os.getenv("DEAD_AIR_THRESHOLD", "0.55"))
LOW_MOOD_THRESHOLD = int(os.getenv("LOW_MOOD_THRESHOLD", "40"))

BATCH_WORKERS = int(os.getenv("BATCH_WORKERS", "4"))
