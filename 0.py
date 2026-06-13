"""
0.py — Phase 0: YouTube/Instagram Ingestion & Demucs Separation
===============================================================
Downloads a video via yt-dlp, extracts audio, separates vocals/BGM,
and enforces the 16kHz mono invariant for the rest of the pipeline.
"""

import sys
import subprocess
from pathlib import Path
import shutil
import torch
import torchaudio

# ============================================================
# INPUT VALIDATION
# ============================================================
if len(sys.argv) < 3:
    print("Usage: python 0.py <VIDEO_URL> <SHORT_ID>")
    sys.exit(1)

VIDEO_URL = sys.argv[1]
SHORT_ID = sys.argv[2]

def detect_platform(url: str) -> str:
    if "instagram.com" in url: return "instagram"
    if "youtube.com" in url or "youtu.be" in url: return "youtube"
    return "unknown"

print(f"[Phase 0] Detected source platform: {detect_platform(VIDEO_URL)}")

# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path("dataset/shorts") / SHORT_ID
BASE_DIR.mkdir(parents=True, exist_ok=True)

RAW_AUDIO    = BASE_DIR / "raw_audio.wav"
TMP_16K      = BASE_DIR / "_tmp_16k.wav"
VOCALS_RAW   = BASE_DIR / "vocals_raw.wav"
VOCALS_COQUI = BASE_DIR / "vocals_coqui.wav"
BGM          = BASE_DIR / "bgm.wav"

# ============================================================
# PHASE 0 — DOWNLOAD RAW AUDIO
# ============================================================
print(f"[Phase 0] Downloading audio → {SHORT_ID}")
subprocess.run(
    [
        "yt-dlp",
        "-f", "bestaudio",
        "--extract-audio",
        "--audio-format", "wav",
        "-o", str(RAW_AUDIO),
        VIDEO_URL,
    ],
    check=True,
)

# ============================================================
# PHASE 0 — RESAMPLE TO MONO 16kHz (INTERNAL CONTRACT)
# ============================================================
print("[Phase 0] Converting raw audio to mono 16kHz...")
subprocess.run(
    [
        "ffmpeg", "-y", "-i", str(RAW_AUDIO),
        "-ac", "1", "-ar", "16000", "-vn", str(TMP_16K),
    ],
    check=True,
)

# ============================================================
# PHASE 0.5 — DEMUCS SEPARATION
# ============================================================
print("[Phase 0.5] Separating vocals and background music...")
demucs_tmp = BASE_DIR / "demucs_tmp"
subprocess.run(
    [
        sys.executable, "-m", "demucs", 
        "-n", "htdemucs", 
        "--two-stems", "vocals",
        "-o", str(demucs_tmp), 
        str(TMP_16K),
    ],
    check=True,
)

demucs_out = demucs_tmp / "htdemucs" / TMP_16K.stem
vocals_tmp = demucs_out / "vocals.wav"
bgm_tmp    = demucs_out / "no_vocals.wav"

if not vocals_tmp.exists() or not bgm_tmp.exists():
    raise RuntimeError("Demucs failed to produce vocals/BGM")

# ============================================================
# PHASE 0.5 — CANONICAL VOCALS & NORMALIZATION
# ============================================================
waveform, sr = torchaudio.load(vocals_tmp)

if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

if sr != 16000:
    waveform = torchaudio.functional.resample(waveform, sr, 16000)

torchaudio.save(VOCALS_RAW, waveform.float(), 16000)
bgm_tmp.replace(BGM)

print("[Phase 0.5] Preparing Coqui-friendly vocals...")
rms = torch.sqrt(torch.mean(waveform ** 2) + 1e-8)
gain = torch.clamp(0.06 / rms, 10 ** (-3 / 20), 10 ** (3 / 20))

torchaudio.save(VOCALS_COQUI, waveform * gain, 16000)

# ============================================================
# CLEANUP
# ============================================================
shutil.rmtree(demucs_tmp, ignore_errors=True)
TMP_16K.unlink(missing_ok=True)
RAW_AUDIO.unlink(missing_ok=True) # Optional: remove original raw download to save space

print("[Phase 0 + 0.5 COMPLETE]")