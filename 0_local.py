import sys
import subprocess
from pathlib import Path
import shutil
import torch
import torchaudio

# ============================================================
# INPUT CONTRACT (LOCAL FILE)
# Usage: python 0_local.py <LOCAL_AUDIO_PATH> <SHORT_ID>
# ============================================================

if len(sys.argv) < 3:
    print("Usage: python 0_local.py <LOCAL_AUDIO_PATH> <SHORT_ID>")
    sys.exit(1)

LOCAL_AUDIO_PATH = Path(sys.argv[1])
SHORT_ID = sys.argv[2]

if not LOCAL_AUDIO_PATH.exists():
    raise FileNotFoundError(f"Source audio not found: {LOCAL_AUDIO_PATH}")

print(f"[Phase 0] Ingesting local audio: {LOCAL_AUDIO_PATH.name} -> {SHORT_ID}")

# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path("dataset/shorts") / SHORT_ID
BASE_DIR.mkdir(parents=True, exist_ok=True)

RAW_AUDIO = BASE_DIR / "raw_audio.wav"
TMP_16K = BASE_DIR / "_tmp_16k.wav"
VOCALS_RAW = BASE_DIR / "vocals_raw.wav"
VOCALS_COQUI = BASE_DIR / "vocals_coqui.wav"
BGM = BASE_DIR / "bgm.wav"

# ============================================================
# PHASE 0 — COPY LOCAL AUDIO
# ============================================================
shutil.copy(LOCAL_AUDIO_PATH, RAW_AUDIO)
print(f"[OK] Audio copied to structural invariant -> {RAW_AUDIO}")

# ============================================================
# PHASE 0 — RESAMPLE TO MONO 16kHz
# ============================================================
print("[Phase 0] Converting raw audio to mono 16kHz...")
subprocess.run(
    ["ffmpeg", "-y", "-i", str(RAW_AUDIO), "-ac", "1", "-ar", "16000", "-vn", str(TMP_16K)],
    check=True,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)

# ============================================================
# PHASE 0.5 — DEMUCS SEPARATION (Explicit Path Handoff)
# ============================================================
print("[Phase 0.5] Separating vocals and background music...")
demucs_tmp = BASE_DIR / "demucs_tmp"

# Using sys.executable ensures we use the exact same python.exe 
# that is currently running 0_local.py, bypassing PATH issues.
subprocess.run(
    [
        sys.executable, "-m", "demucs", 
        "-n", "htdemucs", 
        "--two-stems", "vocals", 
        "-o", str(demucs_tmp), 
        str(TMP_16K)
    ],
    check=True
)

demucs_out = demucs_tmp / "htdemucs" / TMP_16K.stem
vocals_tmp = demucs_out / "vocals.wav"
bgm_tmp = demucs_out / "no_vocals.wav"

if not vocals_tmp.exists() or not bgm_tmp.exists():
    raise RuntimeError("Demucs failed to produce vocals/BGM")

# ============================================================
# PHASE 0.5 — CANONICAL VOCALS & COQUI NORMALIZATION
# ============================================================
waveform, sr = torchaudio.load(vocals_tmp)
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)
if sr != 16000:
    waveform = torchaudio.functional.resample(waveform, sr, 16000)

waveform = waveform.float()
torchaudio.save(VOCALS_RAW, waveform, 16000)

# Save BGM
shutil.copy(bgm_tmp, BGM)

# Coqui Normalization
print("[Phase 0.5] Preparing Coqui-friendly vocals...")
rms = torch.sqrt(torch.mean(waveform ** 2) + 1e-8)
gain = torch.clamp(0.06 / rms, 10 ** (-3 / 20), 10 ** (3 / 20))
torchaudio.save(VOCALS_COQUI, waveform * gain, 16000)

# Cleanup
shutil.rmtree(demucs_tmp, ignore_errors=True)
TMP_16K.unlink(missing_ok=True)

print("[Phase 0 + 0.5 COMPLETE]")