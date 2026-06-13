"""
2.py — Phase 2: Speech Emotion Recognition
==========================================

Runs wav2vec2-SUPERB SER on each transcribed segment of German audio.
SUPERB outputs 4-class labels (ang/hap/neu/sad) which are normalized
to standard 7-class names before saving.

Skip logic: if emotion.json already exists and is valid, phase is skipped.

Note: SUPERB was trained on English (IEMOCAP). It physically processes
      acoustic features and will run on German audio, but accuracy on
      EmoDB emotion classes (esp. disgust, fear) may be limited.
      This output drives the LLM translation prompt. The source
      emotion DISTRIBUTION used for drift measurement is derived from
      gold labels (see 3_full.py), not from this model.

Usage:
  python 2.py <SHORT_ID>
"""

import json
import sys
from pathlib import Path

import torch
import torchaudio
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# ============================================================
# CONSTANTS
# ============================================================

MIN_DURATION  = 0.8   # seconds — skip very short chunks
MIN_TEXT_CHARS = 5

# SUPERB 4-class → standard 7-class name mapping
SUPERB_TO_STANDARD = {
    "ang": "anger",
    "hap": "joy",
    "neu": "neutral",
    "sad": "sadness",
}

# ============================================================
# INPUT CONTRACT
# ============================================================

if len(sys.argv) < 2:
    print("Usage: python 2.py <SHORT_ID>")
    sys.exit(1)

SHORT_ID        = sys.argv[1]
BASE_DIR        = Path("dataset/shorts") / SHORT_ID
AUDIO_PATH      = BASE_DIR / "vocals_raw.wav"
TRANSCRIPT_PATH = BASE_DIR / "transcript.json"
OUTPUT_PATH     = BASE_DIR / "emotion.json"

for p in [AUDIO_PATH, TRANSCRIPT_PATH]:
    if not p.exists():
        raise FileNotFoundError(f"[Phase 2] Missing: {p}")

# ============================================================
# SKIP CHECK
# ============================================================

def _is_valid_emotion(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 10:
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        segs = data.get("segments", [])
        return isinstance(segs, list) and len(segs) > 0
    except Exception:
        return False


if _is_valid_emotion(OUTPUT_PATH):
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        existing = json.load(f)
    n = len(existing.get("segments", []))
    print(f"[Phase 2] SKIP — emotion.json exists ({n} segments): {OUTPUT_PATH}")
    sys.exit(0)

# ============================================================
# LOAD TRANSCRIPT
# ============================================================

with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
    transcript = json.load(f)
segments = transcript["segments"]

# ============================================================
# LOAD SER MODEL
# ============================================================

print("[Phase 2] Loading SER model (superb/wav2vec2-base-superb-er) ...")
MODEL_NAME        = "superb/wav2vec2-base-superb-er"
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model             = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)
model.eval()
id2label          = model.config.id2label

# ============================================================
# LOAD AUDIO
# ============================================================

waveform, sample_rate = torchaudio.load(AUDIO_PATH)
if sample_rate != 16000:
    raise ValueError("[Phase 2] vocals_raw.wav must be 16 kHz")

# ============================================================
# INFERENCE
# ============================================================

print(f"[Phase 2] Running SER on {len(segments)} segments ...")
emotion_results = []

for seg in segments:
    seg_id    = seg["id"]
    text      = seg["text"].strip()
    duration  = seg["end"] - seg["start"]

    # --- Skip very short / empty segments ---
    if duration < MIN_DURATION or len(text) < MIN_TEXT_CHARS:
        emotion_results.append({"id": seg_id, "emotion": "neutral", "intensity": "low"})
        continue

    s     = int(seg["start"] * sample_rate)
    e     = int(seg["end"]   * sample_rate)
    chunk = waveform[:, s:e]

    if chunk.numel() == 0:
        emotion_results.append({"id": seg_id, "emotion": "neutral", "intensity": "low"})
        continue

    inputs = feature_extractor(
        chunk.squeeze().numpy(),
        sampling_rate=sample_rate,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = model(**inputs).logits
        probs  = torch.softmax(logits, dim=-1)

    confidence, pred_id = torch.max(probs, dim=-1)
    confidence  = confidence.item()
    raw_label   = id2label[pred_id.item()].lower()

    # Normalize SUPERB label to standard name
    std_label = SUPERB_TO_STANDARD.get(raw_label, raw_label)

    if confidence >= 0.85:
        intensity = "high"
    elif confidence >= 0.65:
        intensity = "medium"
    else:
        intensity = "low"

    emotion_results.append({
        "id":        seg_id,
        "emotion":   std_label,
        "intensity": intensity,
        "raw_label": raw_label,       # kept for debugging
        "confidence": round(confidence, 4),
    })

# ============================================================
# SAVE OUTPUT
# ============================================================

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(
        {"short_id": SHORT_ID, "segments": emotion_results},
        f, indent=2, ensure_ascii=False,
    )

print(f"[OK] Emotion detection complete → {OUTPUT_PATH}")
print("[Phase 2 COMPLETE]")