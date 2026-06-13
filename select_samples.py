"""
select_samples.py — EmoDB Stratified Sample Selector
=====================================================

Reads EmoDB wav files, maps their emotion codes to the 7-class system,
performs stratified sampling (5 per class, 30 total), copies files to
evaluation_data/, and writes sample_metadata.json for pipeline use.

EmoDB filename format: {speaker:2d}{sentence:3s}{emotion:1s}{version:1s}.wav
  e.g. 03a01Fa.wav → speaker=03, sentence=a01, emotion=F(joy), version=a

EmoDB emotion codes:
  W = Wut        → anger
  E = Ekel       → disgust
  A = Angst      → fear
  F = Freude     → joy
  T = Trauer     → sadness
  N = Neutral    → neutral
  L = Langeweile → neutral  (boredom — mapped to neutral)

Note: EmoDB has no surprise class. The 7-class system's surprise slot
      will not be populated. This is a known dataset limitation.

Usage:
  1. Place EmoDB wav files in emodb_data/ (any subdirectory depth)
  2. python select_samples.py
  3. Copy the printed EVAL_SAMPLES list into run_evaluation.py
"""

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

SOURCE_ROOT       = Path("C:\emodb_data")   # put EmoDB .wav files here
DEST              = Path("C:\emodb_data\evaluation_data")
METADATA_OUT      = DEST / "sample_metadata.json"
SAMPLES_PER_CLASS = 5                    # 5 × 6 emotion classes = 30 total
RANDOM_SEED       = 42                   # research invariant

# EmoDB single-char emotion code → 7-class label
EMODB_MAP = {
    "W": "anger",
    "E": "disgust",
    "A": "fear",
    "F": "joy",
    "T": "sadness",
    "N": "neutral",
    "L": "neutral",   # boredom → neutral
}

# j-hartmann label order (must match drift_eval.py EMOTION_ORDER)
JHART_ORDER = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]


# ============================================================
# HELPERS
# ============================================================

def extract_emodb_emotion(stem: str) -> str | None:
    """
    Extract emotion label from EmoDB filename stem.
    EmoDB stem length is always 7 chars: 2 (speaker) + 3 (sentence) + 1 (emotion) + 1 (version)
    Emotion code is at index 5.
    """
    if len(stem) < 6:
        return None
    code = stem[5].upper()
    return EMODB_MAP.get(code)


def validate_audio(path: Path, min_bytes: int = 4096) -> bool:
    """Basic silent-failure check: file exists and is non-trivially sized."""
    return path.exists() and path.stat().st_size >= min_bytes


# ============================================================
# MAIN
# ============================================================

def stratify_sample() -> None:
    DEST.mkdir(exist_ok=True)

    # --- Discover wav files ---
    all_wavs = list(SOURCE_ROOT.rglob("*.wav"))
    if not all_wavs:
        print(f"[ERROR] No .wav files found under: {SOURCE_ROOT}")
        print("        Create the folder and place EmoDB wav files there.")
        return

    print(f"[select_samples] Found {len(all_wavs)} total wav files.")

    # --- Categorize by emotion ---
    categorized: dict[str, list[Path]] = defaultdict(list)
    unrecognized = 0

    for f in all_wavs:
        emo = extract_emodb_emotion(f.stem)
        if emo:
            categorized[emo].append(f)
        else:
            unrecognized += 1

    if unrecognized:
        print(f"[WARN] {unrecognized} files skipped (unrecognized emotion code)")

    print("\nFiles available per emotion class:")
    for emo in sorted(categorized):
        print(f"  {emo:<10}: {len(categorized[emo])}")

    # --- Stratified sampling ---
    random.seed(RANDOM_SEED)
    selected_list: list[tuple[str, str]] = []
    metadata: dict[str, dict] = {}

    for emo in sorted(categorized):
        files = categorized[emo]

        if len(files) < SAMPLES_PER_CLASS:
            print(f"[WARN] {emo}: only {len(files)} files — taking all of them")
            picked = files
        else:
            picked = random.sample(files, SAMPLES_PER_CLASS)

        # De-duplicate emotion-slot counter for neutral (L + N both → neutral)
        emo_counter = sum(1 for sid, _ in selected_list if sid.startswith(f"eval_{emo}_"))

        for f in picked:
            emo_counter += 1
            sid       = f"eval_{emo}_{emo_counter}"
            dest_path = DEST / f"{sid}.wav"

            # Copy and validate
            shutil.copy(f, dest_path)
            if not validate_audio(dest_path):
                print(f"[WARN] Copied file appears invalid: {dest_path}")

            selected_list.append((sid, str(dest_path)))

            metadata[sid] = {
                "emotion":      emo,
                "source_file":  str(f),
                "source_stem":  f.stem,
                "lang":         "de",
                "jhart_index":  JHART_ORDER.index(emo) if emo in JHART_ORDER else -1,
            }

    # --- Save metadata ---
    with open(METADATA_OUT, "w", encoding="utf-8") as mf:
        json.dump(metadata, mf, indent=2, ensure_ascii=False)

    print(f"\n[OK] {len(selected_list)} samples written to: {DEST}")
    print(f"[OK] Metadata written to: {METADATA_OUT}")

    # --- Validation summary ---
    print("\nSampled emotion distribution:")
    dist_check: dict[str, int] = defaultdict(int)
    for sid, _ in selected_list:
        dist_check[metadata[sid]["emotion"]] += 1
    for emo in sorted(dist_check):
        print(f"  {emo:<10}: {dist_check[emo]}")

    total = len(selected_list)
    print(f"\nTotal samples: {total}  (target ≤ 30)")
    if total > 30:
        print("[WARN] Total exceeds 30 — check SAMPLES_PER_CLASS")

    # --- Print EVAL_SAMPLES snippet ---
    print("\n" + "=" * 60)
    print("COPY AND PASTE THIS INTO run_evaluation.py:")
    print("=" * 60)
    print("EVAL_SAMPLES = [")
    for sid, path in selected_list:
        safe_path = path.replace("\\", "/")
        print(f'    ("{sid}", "{safe_path}"),')
    print("]")


if __name__ == "__main__":
    stratify_sample()