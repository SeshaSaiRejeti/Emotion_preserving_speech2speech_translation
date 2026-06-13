"""
4_1_multi.py — Phase 4A: XTTS Synthesis for All Three Strategies (English Output)
==================================================================================

Synthesizes English speech from all three translation strategies using
Coqui XTTS-v2. Speaker reference audio is the German EmoDB vocal extract
(vocals_coqui.wav) — XTTS cross-lingual voice cloning handles this natively.

Language code: "en"  (target is now English)

Skip logic: if all three tts_coqui_*.wav files already exist and are
            non-trivially sized (> 4KB), this phase is skipped.

Usage:
  python 4_1_multi.py <SHORT_ID>
"""

import sys
import torch
import torchaudio
from pathlib import Path
from TTS.api import TTS

# ================================================================
# INPUT
# ================================================================

if len(sys.argv) < 2:
    print("Usage: python 4_1_multi.py <SHORT_ID>")
    sys.exit(1)

SHORT_ID    = sys.argv[1]
TARGET_LANG = "en"   # English — EmoDB pipeline always targets English

BASE_DIR     = Path("dataset/shorts") / SHORT_ID
AUDIO_PROMPT = BASE_DIR / "vocals_coqui.wav"
STRATEGIES   = ["baseline", "iterative", "candidate"]

# ================================================================
# VALIDATION — required inputs
# ================================================================

if not AUDIO_PROMPT.exists():
    raise FileNotFoundError(f"[Phase 4A] Missing speaker reference: {AUDIO_PROMPT}")

for strategy in STRATEGIES:
    txt_path = BASE_DIR / f"translated_{strategy}.txt"
    if not txt_path.exists():
        raise FileNotFoundError(
            f"[Phase 4A] Missing: {txt_path}. Run 3_full.py first."
        )

# ================================================================
# SKIP CHECK — all three outputs must exist and be non-trivially sized
# ================================================================

MIN_AUDIO_BYTES = 4096   # < 4KB → silent failure

def _all_outputs_valid() -> bool:
    for strategy in STRATEGIES:
        out = BASE_DIR / f"tts_coqui_{strategy}.wav"
        if not out.exists() or out.stat().st_size < MIN_AUDIO_BYTES:
            return False
    return True


if _all_outputs_valid():
    print("[Phase 4A] SKIP — all tts_coqui_*.wav files already exist and are valid.")
    sys.exit(0)

# ================================================================
# LOAD TEXTS
# ================================================================

texts = {}
for strategy in STRATEGIES:
    path = BASE_DIR / f"translated_{strategy}.txt"
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        raise ValueError(f"[Phase 4A] Empty translation file: {path}")
    texts[strategy] = text
    print(f"[Phase 4A] {strategy}: {len(text)} chars")

# ================================================================
# LOAD COQUI XTTS-v2
# ================================================================

print("[Phase 4A] Loading Coqui XTTS-v2 ...")
device = "cuda" if torch.cuda.is_available() else "cpu"

tts = TTS(
    model_name="tts_models/multilingual/multi-dataset/xtts_v2",
    progress_bar=True,
    gpu=(device == "cuda"),
)

# ================================================================
# SYNTHESIZE — strategy by strategy
# ================================================================

for strategy in STRATEGIES:
    output_path = BASE_DIR / f"tts_coqui_{strategy}.wav"

    # Per-strategy skip (allows partial completion)
    if output_path.exists() and output_path.stat().st_size >= MIN_AUDIO_BYTES:
        print(f"[Phase 4A] SKIP {strategy} — output already exists.")
        continue

    print(f"\n[Phase 4A] Synthesizing: {strategy} (lang={TARGET_LANG}) ...")

    wav = tts.tts(
        text=texts[strategy],
        speaker_wav=str(AUDIO_PROMPT),
        language=TARGET_LANG,
    )

    wav_tensor = torch.tensor(wav).unsqueeze(0)

    torchaudio.save(str(output_path), wav_tensor, 24000)

    sz       = output_path.stat().st_size
    duration = wav_tensor.shape[1] / 24000
    print(f"  → {output_path}  ({duration:.1f}s, {sz//1024} KB)")

    # Silent failure check
    if sz < MIN_AUDIO_BYTES:
        print(f"  [WARN] Output suspiciously small ({sz} bytes) — possible silent failure")

# ================================================================
# SUMMARY
# ================================================================

print("\n[Phase 4A COMPLETE]")
for strategy in STRATEGIES:
    path   = BASE_DIR / f"tts_coqui_{strategy}.wav"
    status = f"{path.stat().st_size // 1024} KB" if path.exists() else "MISSING"
    print(f"  {strategy:<12} → {path}  [{status}]")