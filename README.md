# Call-Centre Radar

Turns raw stereo call recordings into transcripts and evidence-cited analysis:
intent, mood timeline with the moment it shifted, resolution status, a ≤40-word
summary, and a 0–100 needs-attention score — every judgement citing the exact
second and the words spoken there.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # add your GROQ_API_KEY
```

You also need **ffmpeg** on your PATH:

```bash
brew install ffmpeg            # macOS
sudo apt-get install ffmpeg    # Ubuntu/Debian
```

### Three ways to run

```bash
make test        # 1. contract + anti-hallucination tests. No API key needed.
make demo        # 2. serve the prebuilt database. 60 seconds, no processing.
make sample      # 3. process 10 real calls end to end. ~2 minutes.
make pipeline    #    the full dataset. ~1 hour.
```

Then open **http://localhost:8000/docs** for the live API browser.

---

## Put the data in place

```
data/
  audio/<call_id>.mp3        # stereo, 8 kHz, left = agent, right = customer
  metadata/<call_id>.json    # matched by filename
```

Unzip `callradar-data.zip` so `audio/` and `metadata/` land under `data/`.

---

## Run the pipeline from scratch

**Step 0 — always do this first.** It checks the three assumptions the whole
design rests on and prints your real cost estimate:

```bash
make preflight
```

It verifies files are genuinely stereo, that the left channel really is the
agent, and totals the audio duration. If it reports the customer speaks first
on most files, set `LEFT_CHANNEL_SPEAKER=customer` in `.env` and re-run.

**Step 1 — process everything.**

```bash
make sample     # try 10 first
make pipeline   # then the rest
```

The batch job is **resumable**. Every call is checkpointed in SQLite, so if it
crashes (it will, at some point) just run it again and it skips what's done.

```bash
python scripts/run_batch.py --retry-failed   # only the ones that broke
python scripts/run_batch.py --force          # reprocess everything
```

**Step 2 — sanity-check the scores.**

```bash
make validate
```

**Step 3 — serve it.**

```bash
make api
```

---

## How it works

```
mp3 ──► split channels + normalise ──► ASR per channel ──► numbered turns
                                                                │
                     SQLite ◄── attention score ◄── LLM analysis┘
                        │                          (cites turn IDs)
                        └──► FastAPI ──► React dashboard
```

### 1. No diarisation model. At all.

The two stereo channels are recorded per speaker and are perfectly isolated —
measured correlation between them is **0.0000** on the sample file, with both
speakers active simultaneously only **0.9%** of the time. So speaker
attribution is a `ffmpeg` channel split, not a model. 100% accurate, free,
instant. Most teams will lose half a day to `pyannote` for a worse result.

### 2. Per-channel normalisation is not optional

Measured on the sample: the customer channel was **17.3 dB quieter** than the
agent channel. Without correction, voice-activity detection silently discards
customer speech. Filters compared on real audio (speech-only RMS):

| Filter | Gap remaining | SNR |
|---|---|---|
| `loudnorm` alone | 11.1 dB | 52 dB |
| `dynaudnorm` | 0.7 dB | 40 dB |
| **`speechnorm` + `loudnorm`** | **0.3 dB** | **46 dB** |

`loudnorm` alone under-corrects because it measures loudness across the whole
file, and the quiet channel is ~89% silence, which drags the measurement down.
The noise floor matters too: over-amplified silence is exactly what makes
Whisper hallucinate loops.

### 3. The LLM never sees a timestamp

This is the core of the design, because the scoring rubric says a claim with no
evidence scores zero and *wrong* evidence scores **negative**.

The model receives numbered turns and nothing else:

```
[1] AGENT: Thank you for calling, how can I help?
[2] CUSTOMER: This is the third time I've called about this charge.
```

It may only cite turn IDs. Then, in Python:

- every cited ID is checked to exist
- a mood shift cited on an **agent** turn is rejected
- every quote is checked to be a **verbatim substring** of the turn it cites
- timestamps are resolved from the turn, interpolated by character position

A hallucinating model fails validation and gets one retry with the errors fed
back. A second failure marks the call `needsReview` instead of shipping a bad
citation. **Fabricating a timestamp is structurally impossible**, because the
model has never seen one.

### 4. The attention score is a formula, not an opinion

| Factor | Points |
|---|---|
| Issue unresolved | +30 |
| Customer ended the call unhappy | +20 |
| Escalation requested | +15 |
| Same customer called within 7 days | +15 |
| Churn risk signalled | +10 |
| High-severity issue (fraud, unauthorised charge…) | +10 |
| Long handle time / dead air / sharp mood drop / long wait | +5 each |

Capped at 100. Every call carries `needsAttentionFactors` so the UI can answer
*"why is this an 87?"* — which is the first thing a judge asks.

`partner_rating` from the customer's own survey is deliberately **excluded**
from the score, so it stays usable as independent ground truth. That's what
`make validate` checks.

### 5. Fixed issue taxonomy

Free-text intents give ~900 unique strings across 1,441 calls and nothing you
can count. `callradar/taxonomy.py` freezes ~20 tags, so "trending issues" is a
real chart with week-over-week deltas. Sample 100 calls, extend the list, then
freeze it before the bulk run.

---

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | status + call count |
| `GET /api/calls` | list, filter by `customerId` / `agentId` |
| `GET /api/calls/{id}` | full `CallRecord` |
| `GET /api/calls/{id}/audio` | the recording, **with HTTP Range support** |
| `GET /api/customers` | customer list + call counts |
| `GET /api/customers/{id}/calls` | one customer's history |
| `GET /api/agents` · `/api/agents/metrics` | volume, handle time, outcomes |
| `GET /api/attention?limit=50` | the ranked "needs a manager today" queue |
| `GET /api/trends?windowDays=7` | trending issues + deltas |
| `GET /api/search?q=refund` | full-text search across every transcript |
| `GET /api/bundle` | `agents` + `customers` + `calls` in one request; optional `limit`/`offset` apply to `calls` only, unlimited by default |
| `POST /api/process` | **live**: send an mp3 + metadata, get the analysis back |

`/api/calls/{id}/audio` returns `206 Partial Content` with `accept-ranges:
bytes`. Without that the browser cannot seek and click-a-timestamp silently
does nothing — test it early.

### Live processing (what judges will hit)

```bash
curl -X POST http://localhost:8000/api/process \
  -F "audio=@data/audio/abc123.mp3" \
  -F "metadata=@data/metadata/abc123.json"
```

Metadata may be a file **or** a JSON string, under either name — all four of
these work:

```bash
-F "metadata=@data/metadata/abc123.json"   # file
-F "metadata={...}"                        # string, same field name
-F "metadataJson={...}"                    # string, legacy field name
```

Missing or malformed metadata comes back as `400` with the parse error, never
a `500`.

Same models as the bulk run, so a call processed live on stage is
indistinguishable from one processed in advance.

---

## Response shape

Matches the frontend `CallRecord` type exactly. `audioUrl` is **removed** —
build it from the id: `/api/calls/{id}/audio`.

```jsonc
{
  "id": "fbf35114ccb44ace",
  "customerId": "17", "customerName": "James Johnson",
  "agentId": "53",   "agentName": "Robert",
  "startedAt": "2020-06-02T00:23:56Z",
  "durationSec": 52,
  "summary": "…≤ 40 words…",
  "intent": "Customer reports being charged twice…",
  "resolved": false,
  "needsAttention": 88,
  "moodShiftSec": 31.0,          // -1 means no shift detected — hide the marker
  "moodBefore": "frustrated", "moodAfter": "angry",
  "issueTag": "duplicate-charge-dispute",
  "transcript":   [{ "speaker": "customer", "startSec": 5.0, "endSec": 9.5, "text": "…" }],
  "moodTimeline": [{ "minute": 0.08, "mood": 44, "label": "frustrated" }],
  "evidence": {
    "intent":    { "timestampSec": 5.0,  "quote": "…", "rationale": "…" },
    "moodShift": { "timestampSec": 31.0, "quote": "…", "rationale": "…" },
    "outcome":   { "timestampSec": 39.5, "quote": "…", "rationale": "…" },
    "attention": { "timestampSec": 31.0, "quote": "…", "rationale": "…" }
  },
  "metadata": { /* the original JSON, camelCased */ },

  // additive — not in the original TS type, safe to ignore
  "needsAttentionFactors": [{ "key": "unresolved", "points": 30, "label": "…" }],
  "needsReview": false
}
```

**Three things to confirm with the frontend:**

1. `moodShiftSec: -1` means *no shift detected*. Hide the marker rather than
   drawing it at zero. Inventing a shift on a 45-second call is exactly what
   loses marks.
2. `moodTimeline.minute` is **fractional** by default (`0.08` = 4.8s). These
   calls average 45–60s, so integer-minute buckets collapse almost every call
   to a single point. Set `MOOD_TIMELINE_MODE=minute` to switch back.
3. `needsAttentionFactors` is extra. Use it for the score breakdown, or ignore
   it — extra keys don't break a structurally-typed consumer.

---

## Cost

Measured against this dataset (~1,441 calls, ~18 hours of audio):

| Stage | Choice | Cost |
|---|---|---|
| Transcription | Groq `whisper-large-v3-turbo` @ $0.04/audio-hr, both channels | **~$1.50** |
| Analysis | Groq `openai/gpt-oss-120b` @ $0.15/$0.60 per M tokens | **~$0.70** |
| Transcription (free alternative) | `faster-whisper` on a Kaggle T4 | **$0** |
| **Total** | | **under $3** |

`make preflight` prints the estimate for your actual data.

---

## Deploying to Azure App Service

Azure hosts the app; it does **no** AI compute. Transcription and analysis stay
on Groq, called from wherever the code runs — so local results and demo results
are byte-for-byte the same pipeline.

```bash
az webapp up --name callradar --runtime "PYTHON:3.11" --sku B1
az webapp config set --name callradar --resource-group <rg> \
  --startup-file "python -m uvicorn api.main:app --host 0.0.0.0 --port 8000"
az webapp config appsettings set --name callradar --resource-group <rg> \
  --settings GROQ_API_KEY=<key> ASR_BACKEND=groq LLM_BACKEND=groq
```

- Develop on **F1 (free)**. Switch to **B1 with Always On** shortly before the
  demo — F1 cold-starts take 30–60 seconds, which is exactly the wrong first
  impression for a judge. B1 is ~$0.018/hr, so the whole hackathon window costs
  a few dollars.
- **Commit `data/callradar.db`.** Never let a live demo depend on a long job or
  an API key working on the day.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ffmpeg not found` | install it; see Quick start |
| `GROQ_API_KEY is not set` | copy `.env.example` → `.env` |
| Everything scores 100 | you're in mock mode; unset `LLM_BACKEND=mock` |
| Transcript is empty | check `preflight` — file may be mono or silent |
| Speakers swapped | flip `LEFT_CHANNEL_SPEAKER` in `.env` |
| Many `needsReview` | model is fighting the schema; lower `LLM_TEMPERATURE` |
| Audio won't seek in browser | something is stripping Range headers on your proxy |

---

## Layout

```
callradar/
  audio.py        ffmpeg split, normalisation, free signal metrics
  transcribe.py   ASR backends (groq / faster_whisper / mock), turn merging
  analyze.py      LLM prompt, JSON validation, turn-ID → timestamp resolution
  scoring.py      the needs-attention formula
  metadata.py     awkward-JSON parsing, camelCase normalisation
  aggregates.py   trends, agent metrics, customer list
  db.py           SQLite + FTS5
  pipeline.py     one call, end to end
  taxonomy.py     the frozen issue tag list
api/main.py       FastAPI
scripts/          preflight, run_batch, process_one, validate_scores
tests/            contract + anti-hallucination tests
```
