"""
3_full.py — Phase 3: Translation with Drift Evaluation (German → English)
=========================================================================

Produces three parallel English translation outputs per segment:
  - Baseline:   single-pass, temperature=0.0
  - Iterative:  metric-guided iterative regeneration (N=3 rounds)
  - Candidate:  N=3 independent candidates, best selected by penalty

Source emotion distribution is derived as a ONE-HOT from the gold
EmoDB emotion label (loaded from evaluation_data/sample_metadata.json).
This avoids running an English-only classifier on German source text.

Fallback: if metadata is unavailable, the Phase 2 SER label is used.

Skip logic: if all output JSON and TXT files already exist and are valid,
            this phase is skipped. Otherwise, per-row checkpointing
            (3_checkpoint.json) allows resuming after crashes.

Outputs:
  segments_baseline.json / translated_baseline.txt
  segments_iterative.json / translated_iterative.txt
  segments_candidate.json / translated_candidate.txt
  drift_comparison.json
  3_checkpoint.json  (safe to delete after success)

Usage:
  python 3_full.py <SHORT_ID>
"""

import json
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

from drift_eval import (
    get_emotion_dist,
    make_source_dist,
    translate_baseline,
    run_iterative,
    run_candidate,
    EMOTION_ORDER,
)

if len(sys.argv) < 2:
    print("Usage: python 3_full.py <SHORT_ID>")
    sys.exit(1)

SHORT_ID = sys.argv[1]
BASE_DIR = Path("dataset/shorts") / SHORT_ID

TRANSCRIPT_PATH = BASE_DIR / "transcript.json"
EMOTION_PATH    = BASE_DIR / "emotion.json"
METADATA_PATH   = Path("evaluation_data") / "sample_metadata.json"


OUT_BASELINE_JSON  = BASE_DIR / "segments_baseline.json"
OUT_ITERATIVE_JSON = BASE_DIR / "segments_iterative.json"
OUT_CANDIDATE_JSON = BASE_DIR / "segments_candidate.json"
OUT_BASELINE_TXT   = BASE_DIR / "translated_baseline.txt"
OUT_ITERATIVE_TXT  = BASE_DIR / "translated_iterative.txt"
OUT_CANDIDATE_TXT  = BASE_DIR / "translated_candidate.txt"
OUT_COMPARISON     = BASE_DIR / "drift_comparison.json"
CHECKPOINT_PATH    = BASE_DIR / "3_checkpoint.json"

for p in [TRANSCRIPT_PATH, EMOTION_PATH]:
    if not p.exists():
        raise FileNotFoundError(f"[Phase 3] Missing: {p}")

# all outputs must exist and be non-trivial

def _all_outputs_valid() -> bool:
    required = [
        OUT_BASELINE_JSON, OUT_ITERATIVE_JSON, OUT_CANDIDATE_JSON,
        OUT_BASELINE_TXT,  OUT_ITERATIVE_TXT,  OUT_CANDIDATE_TXT,
        OUT_COMPARISON,
    ]
    for path in required:
        if not path.exists() or path.stat().st_size < 10:
            return False
    # Validate json integrity
    for path in [OUT_BASELINE_JSON, OUT_ITERATIVE_JSON, OUT_CANDIDATE_JSON, OUT_COMPARISON]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and len(data) == 0:
                return False
        except Exception:
            return False
    return True


if _all_outputs_valid():
    with open(OUT_COMPARISON, "r") as f:
        comp = json.load(f)
    n = comp.get("n_segments", "?")
    print(f"[Phase 3] SKIP — all outputs valid ({n} segments). Delete 3_checkpoint.json to re-run.")
    sys.exit(0)

# loading data
with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
    transcript = json.load(f)

with open(EMOTION_PATH, "r", encoding="utf-8") as f:
    emotion_data = json.load(f)

t_segs = transcript["segments"]
e_segs = emotion_data["segments"]

if len(t_segs) != len(e_segs):
    raise ValueError(f"[Phase 3] Segment count mismatch: {len(t_segs)} vs {len(e_segs)}")

segments = []
for t, e in zip(t_segs, e_segs):
    if t["id"] != e["id"]:
        raise ValueError(f"[Phase 3] Segment ID mismatch: {t['id']} vs {e['id']}")
    segments.append({
        "id":        t["id"],
        "start":     t["start"],
        "end":       t["end"],
        "text":      t["text"].strip(),
        "emotion":   e["emotion"],
        "intensity": e["intensity"],
    })

print(f"[Phase 3] {len(segments)} segments loaded for {SHORT_ID}")

# what is the source emotion label

gold_emotion: str | None = None

if METADATA_PATH.exists():
    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        gold_emotion = meta.get(SHORT_ID, {}).get("emotion")
        if gold_emotion:
            print(f"[Phase 3] Gold emotion from metadata: {gold_emotion}")
    except Exception as e:
        print(f"[Phase 3] WARN: Could not load metadata: {e}")

if gold_emotion is None:
    # fallback: use the most common SER label across segments
    from collections import Counter
    label_counts = Counter(s["emotion"] for s in segments)
    gold_emotion = label_counts.most_common(1)[0][0]
    print(f"[Phase 3] WARN: No metadata — using SER majority label: {gold_emotion}")

# build source emotion distribution
print("[Phase 3] Building source emotion distribution ...")
_, label_order = get_emotion_dist("I feel happy.")   # dummy English — only for label_order
source_dist = make_source_dist(gold_emotion, label_order)

print(f"[Phase 3] Source dist — dominant: {label_order[int(np.argmax(source_dist))]}")
print(f"          Vector: {dict(zip(label_order, source_dist.round(3)))}")

RATE_DELAY = 0.8   

def save_checkpoint(baseline_rows, iterative_rows, candidate_rows):
    data = {
        "baseline":  baseline_rows,
        "iterative": iterative_rows,
        "candidate": candidate_rows,
    }
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, default=str)


def load_checkpoint():
    if CHECKPOINT_PATH.exists():
        print(f"[Phase 3] Checkpoint found — resuming ...")
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        b = data.get("baseline",  [])
        i = data.get("iterative", [])
        c = data.get("candidate", [])
        print(f"  baseline={len(b)}  iterative={len(i)}  candidate={len(c)}")
        return b, i, c
    return [], [], []


baseline_rows, iterative_rows, candidate_rows = load_checkpoint()

done_baseline  = {r["id"] for r in baseline_rows}
done_iterative = {r["id"] for r in iterative_rows}
done_candidate = {r["id"] for r in candidate_rows}

# ---- BASELINE ----
if len(baseline_rows) < len(segments):
    print("\n[Phase 3] === BASELINE ===")
    for seg in tqdm(segments, desc="baseline"):
        sid = seg["id"]
        if sid in done_baseline:
            continue

        translation = translate_baseline(seg["text"], seg["emotion"], delay=RATE_DELAY)

        baseline_rows.append({
            "id":          sid,
            "start":       seg["start"],
            "end":         seg["end"],
            "source_text": seg["text"],
            "emotion":     seg["emotion"],
            "intensity":   seg["intensity"],
            "translation": translation,
        })
        save_checkpoint(baseline_rows, iterative_rows, candidate_rows)
else:
    print("[Phase 3] Baseline complete — skipping")

if len(iterative_rows) < len(segments):
    print("\n[Phase 3] === ITERATIVE REGENERATION ===")
    for seg in tqdm(segments, desc="iterative"):
        sid = seg["id"]
        if sid in done_iterative:
            continue

        result = run_iterative(
            source_text  = seg["text"],
            emotion      = seg["emotion"],
            source_dist  = source_dist,
            label_order  = label_order,
            max_attempts = 3,
            delay        = RATE_DELAY,
        )

        iterative_rows.append({
            "id":          sid,
            "start":       seg["start"],
            "end":         seg["end"],
            "source_text": seg["text"],
            "emotion":     seg["emotion"],
            "intensity":   seg["intensity"],
            "translation": result["best_translation"],
            "metrics":     result["best_metrics"],
            "attempts":    result["attempts"],
        })
        save_checkpoint(baseline_rows, iterative_rows, candidate_rows)
else:
    print("[Phase 3] Iterative complete — skipping")

if len(candidate_rows) < len(segments):
    print("\n[Phase 3] === CANDIDATE GENERATION ===")
    for seg in tqdm(segments, desc="candidate"):
        sid = seg["id"]
        if sid in done_candidate:
            continue

        result = run_candidate(
            source_text  = seg["text"],
            emotion      = seg["emotion"],
            source_dist  = source_dist,
            label_order  = label_order,
            n_candidates = 3,
            delay        = RATE_DELAY,
        )

        candidate_rows.append({
            "id":          sid,
            "start":       seg["start"],
            "end":         seg["end"],
            "source_text": seg["text"],
            "emotion":     seg["emotion"],
            "intensity":   seg["intensity"],
            "translation": result["best_translation"],
            "metrics":     result["best_metrics"],
            "candidates":  result["candidates"],
            "early_exit":  result["early_exit"],
        })
        save_checkpoint(baseline_rows, iterative_rows, candidate_rows)
else:
    print("[Phase 3] Candidate complete — skipping")

# sort and save

baseline_rows.sort(key=lambda r: r["id"])
iterative_rows.sort(key=lambda r: r["id"])
candidate_rows.sort(key=lambda r: r["id"])


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    print(f"[saved] {path}")


save_json(OUT_BASELINE_JSON,  baseline_rows)
save_json(OUT_ITERATIVE_JSON, iterative_rows)
save_json(OUT_CANDIDATE_JSON, candidate_rows)

# concatenated translation for coqui
for path, rows in [
    (OUT_BASELINE_TXT,  baseline_rows),
    (OUT_ITERATIVE_TXT, iterative_rows),
    (OUT_CANDIDATE_TXT, candidate_rows),
]:
    text = " ".join(r["translation"] for r in rows if r.get("translation"))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[saved] {path}  ({len(text)} chars)")

#comparison
def _agg(rows):
    metrics = [r.get("metrics") for r in rows if r.get("metrics")]
    if not metrics:
        return {}
    return {
        "n":              len(metrics),
        "mean_D_JS":      float(np.mean([m["D_JS"]      for m in metrics])),
        "mean_semantic":  float(np.mean([m["semantic"]   for m in metrics])),
        "flip_rate":      float(np.mean([m["D_flip"]     for m in metrics])),
        "mean_D_entropy": float(np.mean([m["D_entropy"]  for m in metrics])),
        "mean_penalty":   float(np.mean([m["penalty"]    for m in metrics])),
    }

comparison = {
    "short_id":     SHORT_ID,
    "gold_emotion": gold_emotion,
    "n_segments":   len(segments),
    "iterative":    _agg(iterative_rows),
    "candidate":    _agg(candidate_rows),
}
save_json(OUT_COMPARISON, comparison)

#summary

print("\n" + "=" * 60)
print(f"PHASE 3 COMPLETE — {SHORT_ID}  (German → English)")
print(f"  Gold emotion: {gold_emotion}")
print(f"  Segments:     {len(segments)}")

for name, agg in [("Iterative", comparison["iterative"]),
                  ("Candidate", comparison["candidate"])]:
    if agg:
        print(f"\n  {name}:")
        print(f"    Semantic  : {agg['mean_semantic']:.4f}")
        print(f"    D_JS      : {agg['mean_D_JS']:.4f}")
        print(f"    Flip rate : {agg['flip_rate']*100:.1f}%")
        print(f"    D_entropy : {agg['mean_D_entropy']:.4f}")

print(f"\n  Checkpoint: {CHECKPOINT_PATH}  (safe to delete)")
print("=" * 60)