"""
run_evaluation.py — Orchestrator for Real-World Audio Pipeline
==============================================================

Usage:
  python run_evaluation.py <VIDEO_URL> <SHORT_ID>

Example:
  python run_evaluation.py "https://youtube.com/shorts/xyz" my_test_short
"""

import json
import subprocess
import sys
import os
from pathlib import Path

# ============================================================
# EXECUTION INVARIANTS
# ============================================================

PYTHON_MAIN       = r"C:\Users\Sesha Sai\emotion_s2st\venv\Scripts\python.exe"
PYTHON_CHATTERBOX = r"C:\Users\Sesha Sai\emotion_s2st\venv_chatterbox\Scripts\python.exe"
PYTHON_PHASE3     = PYTHON_MAIN

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "") 

# ============================================================
# VALIDATION THRESHOLDS
# ============================================================

MIN_WAV_BYTES  = 4_096    
MIN_JSON_BYTES = 10       
MIN_TXT_BYTES  = 2        

# ============================================================
# FILE-EXISTENCE SKIP SPECS
# ============================================================

def phase_skip_spec(short_id: str) -> dict[str, list[tuple[str, int]]]:
    return {
        "phase0": [
            ("vocals_raw.wav",  MIN_WAV_BYTES),
            ("vocals_coqui.wav",MIN_WAV_BYTES),
            ("bgm.wav",         MIN_WAV_BYTES),
        ],
        "phase1": [("transcript.json", MIN_JSON_BYTES)],
        "phase2": [("emotion.json",    MIN_JSON_BYTES)],
        "phase3": [
            ("segments_baseline.json",  MIN_JSON_BYTES),
            ("segments_iterative.json", MIN_JSON_BYTES),
            ("segments_candidate.json", MIN_JSON_BYTES),
            ("translated_baseline.txt",  MIN_TXT_BYTES),
            ("translated_iterative.txt", MIN_TXT_BYTES),
            ("translated_candidate.txt", MIN_TXT_BYTES),
            ("drift_comparison.json",    MIN_JSON_BYTES),
        ],
        "phase4a": [
            ("tts_coqui_baseline.wav",  MIN_WAV_BYTES),
            ("tts_coqui_iterative.wav", MIN_WAV_BYTES),
            ("tts_coqui_candidate.wav", MIN_WAV_BYTES),
        ],
        "phase4b": [
            ("tts_chatterbox_baseline.wav",  MIN_WAV_BYTES),
            ("tts_chatterbox_iterative.wav", MIN_WAV_BYTES),
            ("tts_chatterbox_candidate.wav", MIN_WAV_BYTES),
        ],
        "phase44": [
            ("final_baseline.wav",  MIN_WAV_BYTES),
            ("final_iterative.wav", MIN_WAV_BYTES),
            ("final_candidate.wav", MIN_WAV_BYTES),
        ],
        "phase5": [("acoustic_comparison.json", MIN_JSON_BYTES)],
    }

def phase_complete(short_id: str, phase: str) -> bool:
    base = Path("dataset/shorts") / short_id
    specs = phase_skip_spec(short_id).get(phase, [])
    return all(
        (base / fname).exists() and (base / fname).stat().st_size >= min_bytes
        for fname, min_bytes in specs
    )

def validate_phase(short_id: str, phase: str, halt_on_fail: bool = True) -> bool:
    base   = Path("dataset/shorts") / short_id
    specs  = phase_skip_spec(short_id).get(phase, [])
    issues = []

    for fname, min_bytes in specs:
        fpath = base / fname
        if not fpath.exists():
            issues.append(f"MISSING: {fname}")
        elif fpath.stat().st_size < min_bytes:
            issues.append(f"EMPTY/TINY: {fname} ({fpath.stat().st_size} bytes < {min_bytes})")
        elif fname.endswith(".json"):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list) and len(data) == 0:
                    issues.append(f"EMPTY LIST: {fname}")
            except Exception as e:
                issues.append(f"INVALID JSON: {fname} ({e})")

    if issues:
        print(f"\n[VALIDATION FAIL] {short_id} / {phase}:")
        for issue in issues:
            print(f"  ✗ {issue}")
        if halt_on_fail:
            print("[FATAL] Silent failure detected. Halting pipeline.")
            sys.exit(1)
        return False
    return True

def run_cmd(command: list, env: dict | None = None) -> None:
    print(f"\n>>> {' '.join(str(c) for c in command)}")
    result = subprocess.run(command, env=env)
    if result.returncode != 0:
        print(f"\n[FATAL] Command failed: {' '.join(str(c) for c in command)}")
        sys.exit(1)

# ============================================================
# PIPELINE EXECUTION
# ============================================================

def process_sample(video_url: str, short_id: str) -> None:
    print("\n" + "=" * 64)
    print(f"TARGET: {short_id}  |  {video_url}")
    print("=" * 64)

    env = os.environ.copy()
    if GROQ_API_KEY:
        env["GROQ_API_KEY"] = GROQ_API_KEY

    # --- Phase 0 (Using 0.py for URL extraction) ---
    if phase_complete(short_id, "phase0"):
        print(f"[SKIP] Phase 0 — outputs already valid")
    else:
        run_cmd([PYTHON_MAIN, "0.py", video_url, short_id])
        validate_phase(short_id, "phase0")

    # --- Phase 1 ---
    if phase_complete(short_id, "phase1"):
        print(f"[SKIP] Phase 1 — transcript.json already valid")
    else:
        run_cmd([PYTHON_MAIN, "1.py", short_id])
        validate_phase(short_id, "phase1")

    # --- Phase 2 ---
    if phase_complete(short_id, "phase2"):
        print(f"[SKIP] Phase 2 — emotion.json already valid")
    else:
        run_cmd([PYTHON_MAIN, "2.py", short_id])
        validate_phase(short_id, "phase2")

    # --- Phase 3 ---
    if phase_complete(short_id, "phase3"):
        print(f"[SKIP] Phase 3 — all translation outputs already valid")
    else:
        run_cmd([PYTHON_PHASE3, "3.py", short_id], env=env)
        validate_phase(short_id, "phase3")

    # --- Phase 4A (XTTS) ---
    if phase_complete(short_id, "phase4a"):
        print(f"[SKIP] Phase 4A — all tts_coqui_*.wav already valid")
    else:
        run_cmd([PYTHON_MAIN, "4.1.py", short_id])
        validate_phase(short_id, "phase4a")

    # --- Phase 4B (Chatterbox) ---
    if phase_complete(short_id, "phase4b"):
        print(f"[SKIP] Phase 4B — all tts_chatterbox_*.wav already valid")
    else:
        run_cmd([PYTHON_CHATTERBOX, "4.2.py", short_id])
        validate_phase(short_id, "phase4b")

    # --- Phase 4.4 (Loudness + Mix) ---
    if phase_complete(short_id, "phase44"):
        print(f"[SKIP] Phase 4.4 — all final_*.wav already valid")
    else:
        run_cmd([PYTHON_MAIN, "4.4.py", short_id])
        validate_phase(short_id, "phase44")

    # --- Phase 5 (Acoustic verification) ---
    if phase_complete(short_id, "phase5"):
        print(f"[SKIP] Phase 5 — acoustic_comparison.json already valid")
    else:
        run_cmd([PYTHON_MAIN, "5.py", short_id])
        validate_phase(short_id, "phase5")

    print(f"\n[DONE] {short_id} Pipeline Execution Complete.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python run_evaluation.py <VIDEO_URL> <SHORT_ID>")
        sys.exit(1)

    VIDEO_URL = sys.argv[1]
    SHORT_ID = sys.argv[2]

    process_sample(VIDEO_URL, SHORT_ID)