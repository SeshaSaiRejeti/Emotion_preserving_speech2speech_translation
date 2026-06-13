"""
5_multi.py — Phase 5: Acoustic Emotion Verification for All Strategies
======================================================================

Diagnostic phase. Runs SER on all three final audio outputs and
reports per-strategy acoustic emotion agreement with Phase 2 SER labels.

Note: This quantifies the acoustic emotion gap that text-level
optimization cannot close. Text-level drift (3_full.py) and acoustic
drift (this phase) serve different measurement purposes.

Skip logic: if acoustic_comparison.json already exists and is valid,
            phase is skipped. Per-strategy skip also applied.

Usage:
  python 5_multi.py <SHORT_ID>
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torchaudio
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

STRATEGIES      = ["baseline", "iterative", "candidate"]
SER_MODEL       = "superb/wav2vec2-base-superb-er"
MIN_JSON_BYTES  = 10

# SUPERB label → standard name
SUPERB_TO_STD = {
    "ang": "anger",
    "hap": "joy",
    "neu": "neutral",
    "sad": "sadness",
}

# ================================================================
# INPUT
# ================================================================

if len(sys.argv) < 2:
    print("Usage: python 5_multi.py <SHORT_ID>")
    sys.exit(1)

SHORT_ID         = sys.argv[1]
BASE_DIR         = Path("dataset/shorts") / SHORT_ID
TRANSCRIPT_PATH  = BASE_DIR / "transcript.json"
SRC_EMOTION_PATH = BASE_DIR / "emotion.json"
COMP_PATH        = BASE_DIR / "acoustic_comparison.json"

# ================================================================
# VALIDATION
# ================================================================

for p in [TRANSCRIPT_PATH, SRC_EMOTION_PATH]:
    if not p.exists():
        raise FileNotFoundError(f"[Phase 5] Missing: {p}")

for strategy in STRATEGIES:
    p = BASE_DIR / f"final_{strategy}.wav"
    if not p.exists():
        raise FileNotFoundError(f"[Phase 5] Missing: {p}. Run 4_4_multi.py first.")

# ================================================================
# GLOBAL SKIP CHECK
# ================================================================

def _comparison_valid() -> bool:
    if not COMP_PATH.exists() or COMP_PATH.stat().st_size < MIN_JSON_BYTES:
        return False
    try:
        with open(COMP_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return "strategies" in data and len(data["strategies"]) == len(STRATEGIES)
    except Exception:
        return False


if _comparison_valid():
    print(f"[Phase 5] SKIP — acoustic_comparison.json exists and is valid: {COMP_PATH}")
    sys.exit(0)

# ================================================================
# LOAD DATA
# ================================================================

with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
    transcript = json.load(f)

with open(SRC_EMOTION_PATH, "r", encoding="utf-8") as f:
    src_emotion = json.load(f)

segments     = transcript["segments"]
src_segments = src_emotion["segments"]

if len(segments) != len(src_segments):
    raise ValueError("[Phase 5] Segment count mismatch between transcript and emotion.json")

# ================================================================
# LOAD SER MODEL
# ================================================================

print("[Phase 5] Loading SER model ...")
feature_extractor = AutoFeatureExtractor.from_pretrained(SER_MODEL)
ser_model         = AutoModelForAudioClassification.from_pretrained(SER_MODEL)
ser_model.eval()
id2label          = ser_model.config.id2label

# ================================================================
# SER INFERENCE
# ================================================================

def run_ser(audio_path: Path) -> list[dict]:
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    results = []
    for seg in segments:
        s     = int(seg["start"] * sr)
        e     = int(seg["end"]   * sr)
        chunk = waveform[:, s:e]

        if chunk.numel() == 0:
            results.append({"id": seg["id"], "predicted_emotion": "neutral", "confidence": 0.0})
            continue

        inputs = feature_extractor(
            chunk.squeeze().numpy(), sampling_rate=sr, return_tensors="pt"
        )
        with torch.no_grad():
            probs = torch.softmax(ser_model(**inputs).logits, dim=-1)

        conf, pred_id = torch.max(probs, dim=-1)
        raw   = id2label[pred_id.item()].lower()
        label = SUPERB_TO_STD.get(raw, raw)

        results.append({
            "id":                seg["id"],
            "predicted_emotion": label,
            "confidence":        round(conf.item(), 4),
        })
    return results


# ================================================================
# AGREEMENT METRICS
# ================================================================

def compute_agreement(target_results: list[dict]) -> dict:
    total   = len(src_segments)
    matches = 0
    neutral = 0
    per_total   = defaultdict(int)
    per_correct = defaultdict(int)
    confusion   = defaultdict(lambda: defaultdict(int))

    for src, tgt in zip(src_segments, target_results):
        sl = src["emotion"]
        tl = tgt["predicted_emotion"]
        per_total[sl] += 1
        confusion[sl][tl] += 1
        if sl == tl:
            matches += 1
            per_correct[sl] += 1
        if tl == "neutral":
            neutral += 1

    return {
        "total_segments":        total,
        "label_agreement":       round(matches / total, 4) if total else 0.0,
        "neutral_collapse_rate": round(neutral / total, 4) if total else 0.0,
        "per_emotion_recall":    {
            emo: round(per_correct[emo] / per_total[emo], 4) if per_total[emo] else 0.0
            for emo in per_total
        },
        "confusion_matrix": {src: dict(tgt_c) for src, tgt_c in confusion.items()},
    }


# ================================================================
# MAIN
# ================================================================

all_agreement = {}

for strategy in STRATEGIES:
    audio_path   = BASE_DIR / f"final_{strategy}.wav"
    per_seg_path = BASE_DIR / f"acoustic_drift_{strategy}.json"

    # Per-strategy skip
    if per_seg_path.exists() and per_seg_path.stat().st_size >= MIN_JSON_BYTES:
        try:
            with open(per_seg_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            all_agreement[strategy] = cached["metrics"]
            print(f"[Phase 5] SKIP {strategy} — cached results loaded.")
            continue
        except Exception:
            pass   # fall through to re-run

    print(f"\n[Phase 5] Evaluating: {strategy}")
    target_results = run_ser(audio_path)
    agreement      = compute_agreement(target_results)
    all_agreement[strategy] = agreement

    with open(per_seg_path, "w", encoding="utf-8") as f:
        json.dump(
            {"short_id": SHORT_ID, "strategy": strategy,
             "segments": target_results, "metrics": agreement},
            f, indent=2, ensure_ascii=False,
        )
    print(f"  Agreement: {agreement['label_agreement']:.4f}  "
          f"Neutral collapse: {agreement['neutral_collapse_rate']:.4f}")

# ================================================================
# COMPARISON + SAVE
# ================================================================

comparison = {
    "short_id":   SHORT_ID,
    "n_segments": len(segments),
    "strategies": {s: all_agreement[s] for s in STRATEGIES},
}
with open(COMP_PATH, "w", encoding="utf-8") as f:
    json.dump(comparison, f, indent=2, ensure_ascii=False)

# ================================================================
# PRINT SUMMARY
# ================================================================

print("\n" + "=" * 58)
print("PHASE 5 — ACOUSTIC EMOTION VERIFICATION")
print("=" * 58)
print(f"{'Strategy':<14} {'Agreement':>12} {'Neutral Collapse':>18}")
print("-" * 58)
for strategy in STRATEGIES:
    a = all_agreement[strategy]
    print(f"{strategy:<14} {a['label_agreement']:12.4f} {a['neutral_collapse_rate']:18.4f}")
print("=" * 58)
print("\nNote: acoustic drift is a diagnostic metric. Text-level")
print("optimization targets emotion–meaning coupled drift, not")
print("acoustic emotion realization in TTS output.")
print(f"\n[Phase 5 COMPLETE]  Comparison → {COMP_PATH}")