"""
4_4_multi.py — Phase 4.4: Audio Reconstruction for All Strategies
=================================================================

Applies loudness normalization (LUFS matching) and BGM mixing to all
three Chatterbox outputs.

Skip logic: if all three final_*.wav outputs already exist and are
            non-trivially sized, phase is skipped. Per-strategy skip
            also applied.

Usage:
  python 4_4_multi.py <SHORT_ID>
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import pyloudnorm as pyln
import librosa
from pydub import AudioSegment

STRATEGIES      = ["baseline", "iterative", "candidate"]
MIN_AUDIO_BYTES = 4096

# ================================================================
# INPUT
# ================================================================

if len(sys.argv) < 2:
    print("Usage: python 4_4_multi.py <SHORT_ID>")
    sys.exit(1)

SHORT_ID        = sys.argv[1]
BASE_DIR        = Path("dataset/shorts") / SHORT_ID
REFERENCE_AUDIO = BASE_DIR / "vocals_raw.wav"
BGM_AUDIO       = BASE_DIR / "bgm.wav"

# ================================================================
# VALIDATION
# ================================================================

if not REFERENCE_AUDIO.exists():
    raise FileNotFoundError(f"[Phase 4.4] Missing reference audio: {REFERENCE_AUDIO}")

for strategy in STRATEGIES:
    p = BASE_DIR / f"tts_chatterbox_{strategy}.wav"
    if not p.exists():
        raise FileNotFoundError(f"[Phase 4.4] Missing: {p}. Run 4_2_multi.py first.")

# ================================================================
# GLOBAL SKIP CHECK
# ================================================================

def _all_outputs_valid() -> bool:
    for strategy in STRATEGIES:
        out = BASE_DIR / f"final_{strategy}.wav"
        if not out.exists() or out.stat().st_size < MIN_AUDIO_BYTES:
            return False
    return True


if _all_outputs_valid():
    print("[Phase 4.4] SKIP — all final_*.wav files already exist and are valid.")
    sys.exit(0)

# ================================================================
# LOAD REFERENCE AUDIO
# ================================================================

ref_audio, ref_sr = sf.read(str(REFERENCE_AUDIO))
print(f"[Phase 4.4] Reference: {REFERENCE_AUDIO.name}  (SR={ref_sr})")

# ================================================================
# BGM
# ================================================================

has_bgm = BGM_AUDIO.exists()
bgm     = AudioSegment.from_file(BGM_AUDIO) if has_bgm else None
if has_bgm:
    print(f"[Phase 4.4] BGM loaded: {BGM_AUDIO.name}")
else:
    print("[Phase 4.4] No BGM — speech-only outputs")

# ================================================================
# HELPERS
# ================================================================

def to_mono(x: np.ndarray) -> np.ndarray:
    return x if x.ndim == 1 else np.mean(x, axis=1)


def resample_ref(audio: np.ndarray, src_sr: int, tgt_sr: int) -> np.ndarray:
    if src_sr == tgt_sr:
        return audio
    if audio.ndim == 2:
        audio = audio.T
    out = librosa.resample(audio, orig_sr=src_sr, target_sr=tgt_sr)
    return out.T if out.ndim == 2 else out


# ================================================================
# PROCESS ONE STRATEGY
# ================================================================

def process_strategy(strategy: str) -> None:
    output_path = BASE_DIR / f"final_{strategy}.wav"
    temp_path   = BASE_DIR / f"_tmp_loudness_{strategy}.wav"

    # Per-strategy skip
    if output_path.exists() and output_path.stat().st_size >= MIN_AUDIO_BYTES:
        print(f"[Phase 4.4] SKIP {strategy} — output already exists.")
        return

    print(f"\n[Phase 4.4] {strategy} ...")

    speech_path              = BASE_DIR / f"tts_chatterbox_{strategy}.wav"
    speech_audio, speech_sr  = sf.read(str(speech_path))
    ref_resampled            = resample_ref(ref_audio, ref_sr, speech_sr)

    # LUFS measurement + gain
    meter       = pyln.Meter(speech_sr)
    ref_lufs    = meter.integrated_loudness(to_mono(ref_resampled))
    speech_lufs = meter.integrated_loudness(to_mono(speech_audio))
    gain_db     = ref_lufs - speech_lufs

    print(f"  Reference LUFS : {ref_lufs:.2f}")
    print(f"  Speech LUFS    : {speech_lufs:.2f}")
    print(f"  Gain applied   : {gain_db:.2f} dB")

    # Apply gain + clip guard
    gain_linear = 10 ** (gain_db / 20.0)
    speech_adj  = np.clip(speech_audio * gain_linear, -1.0, 1.0)
    sf.write(str(temp_path), speech_adj, speech_sr)

    speech_seg = AudioSegment.from_file(temp_path)

    if not has_bgm:
        speech_seg.export(str(output_path), format="wav")
    else:
        speech_ms = len(speech_seg)
        bgm_loop  = bgm
        while len(bgm_loop) < speech_ms:
            bgm_loop = bgm_loop + bgm
        bgm_loop  = bgm_loop[:speech_ms]
        speech_seg.overlay(bgm_loop).export(str(output_path), format="wav")

    temp_path.unlink(missing_ok=True)

    sz = output_path.stat().st_size
    print(f"  → {output_path}  [{sz//1024} KB]")

    if sz < MIN_AUDIO_BYTES:
        print(f"  [WARN] Output suspiciously small ({sz} bytes) — possible silent failure")


# ================================================================
# MAIN
# ================================================================

for strategy in STRATEGIES:
    process_strategy(strategy)

print("\n[Phase 4.4 COMPLETE]")
for strategy in STRATEGIES:
    path   = BASE_DIR / f"final_{strategy}.wav"
    status = f"{path.stat().st_size // 1024} KB" if path.exists() else "MISSING"
    print(f"  final_{strategy}.wav  [{status}]")