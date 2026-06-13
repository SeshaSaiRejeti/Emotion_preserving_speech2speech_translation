"""
1.py — Phase 1: Whisper Transcription (German Source Audio)
============================================================

Transcribes German EmoDB audio using Whisper with explicit language="de"
to prevent misdetection on short speech segments.

Skip logic: if transcript.json already exists and is valid, this phase
            is skipped entirely without rerunning Whisper.

Usage:
  python 1.py <SHORT_ID>
"""

import json
import sys
from pathlib import Path

import whisper

# ============================================================
# INPUT CONTRACT
# ============================================================

if len(sys.argv) < 2:
    print("Usage: python 1.py <SHORT_ID>")
    sys.exit(1)

SHORT_ID    = sys.argv[1]
SOURCE_LANG = "de"   # EmoDB is German — explicit prevents Whisper misdetection

BASE_DIR    = Path("dataset/shorts") / SHORT_ID
AUDIO_PATH  = BASE_DIR / "vocals_raw.wav"
OUTPUT_JSON = BASE_DIR / "transcript.json"

if not AUDIO_PATH.exists():
    raise FileNotFoundError(f"[Phase 1] Audio not found: {AUDIO_PATH}")

# ============================================================
# SKIP CHECK
# ============================================================

def _is_valid_transcript(path: Path) -> bool:
    """Returns True if transcript.json exists, is valid JSON, and has segments."""
    if not path.exists() or path.stat().st_size < 10:
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        segs = data.get("segments", [])
        return isinstance(segs, list) and len(segs) > 0
    except Exception:
        return False


if _is_valid_transcript(OUTPUT_JSON):
    with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
        existing = json.load(f)
    n = len(existing.get("segments", []))
    print(f"[Phase 1] SKIP — transcript.json exists ({n} segments): {OUTPUT_JSON}")
    sys.exit(0)

# ============================================================
# LOAD MODEL
# ============================================================

print("[Phase 1] Loading Whisper model (small) ...")
model = whisper.load_model("small")

# ============================================================
# TRANSCRIBE — language forced to German
# ============================================================

print(f"[Phase 1] Transcribing {SHORT_ID} (lang={SOURCE_LANG}) ...")
result = model.transcribe(
    str(AUDIO_PATH),
    language=SOURCE_LANG,   # explicit — prevents auto-detect errors on short clips
    verbose=False,
)

segments = result["segments"]

# ============================================================
# PROCESS SEGMENTS
# ============================================================

processed = []
for idx, seg in enumerate(segments):
    processed.append({
        "id":    idx,
        "start": round(seg["start"], 2),
        "end":   round(seg["end"],   2),
        "text":  seg["text"].strip(),
    })

# ============================================================
# SAVE OUTPUT
# ============================================================

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(
        {"short_id": SHORT_ID, "segments": processed},
        f, indent=2, ensure_ascii=False,
    )

print(f"[OK] Transcript saved → {OUTPUT_JSON}  ({len(processed)} segments)")
print("[Phase 1 COMPLETE]")