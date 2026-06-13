"""
4_2_multi.py — Phase 4B: Chatterbox Expressive TTS for All Strategies
======================================================================

Runs Chatterbox Turbo on all three translation strategies.
Speaker prompt is extracted per-block from the Coqui reference audio.

Skip logic: if all three tts_chatterbox_*.wav outputs already exist and
            are non-trivially sized, this phase is skipped.

Per-strategy skip: individual strategies are also skipped if their output
                   already exists, allowing partial completion.

Usage:
  python 4_2_multi.py <SHORT_ID>
"""

import sys
import json
import shutil
from pathlib import Path

import torch
import torchaudio
import peft  # noqa: F401

from chatterbox.tts_turbo import ChatterboxTurboTTS

# ================================================================
# CONFIG
# ================================================================

MAX_CHARS         = 500
EST_MULT          = 1.2
MAX_PROMPT_SEC    = 8.0
MIN_PROMPT_SEC    = 5.0
MAX_TOTAL_SILENCE = 5.0
MIN_GAP_SILENCE   = 0.5
MIN_AUDIO_BYTES   = 4096

STRATEGIES = ["baseline", "iterative", "candidate"]

# ================================================================
# INPUT
# ================================================================

if len(sys.argv) < 2:
    print("Usage: python 4_2_multi.py <SHORT_ID>")
    sys.exit(1)

SHORT_ID = sys.argv[1]
BASE_DIR = Path("dataset/shorts") / SHORT_ID
RAW_VOCALS_PATH = BASE_DIR / "vocals_raw.wav"

# ================================================================
# VALIDATION
# ================================================================

if not RAW_VOCALS_PATH.exists():
    raise FileNotFoundError(f"[Phase 4B] Missing: {RAW_VOCALS_PATH}")

for strategy in STRATEGIES:
    for fname in [f"segments_{strategy}.json", f"tts_coqui_{strategy}.wav"]:
        if not (BASE_DIR / fname).exists():
            raise FileNotFoundError(
                f"[Phase 4B] Missing: {BASE_DIR / fname}. "
                f"Run previous phases first."
            )

# ================================================================
# GLOBAL SKIP CHECK
# ================================================================

def _all_outputs_valid() -> bool:
    for strategy in STRATEGIES:
        out = BASE_DIR / f"tts_chatterbox_{strategy}.wav"
        if not out.exists() or out.stat().st_size < MIN_AUDIO_BYTES:
            return False
    return True


if _all_outputs_valid():
    print("[Phase 4B] SKIP — all tts_chatterbox_*.wav files already exist and are valid.")
    sys.exit(0)

# ================================================================
# LOAD CHATTERBOX
# ================================================================

print("[Phase 4B] Loading Chatterbox Turbo ...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model  = ChatterboxTurboTTS.from_pretrained(device=device)
SR     = model.sr

# ================================================================
# LOAD RAW VOCALS (for duration reference)
# ================================================================

raw_vocals, raw_sr = torchaudio.load(RAW_VOCALS_PATH)
T_ORIG = raw_vocals.shape[1] / raw_sr

# ================================================================
# HELPERS
# ================================================================

def ensure_min_duration(audio_path: str, min_sec: float = 5.1) -> str:
    """Loop audio if shorter than min_sec to satisfy Chatterbox invariant."""
    wav, sr = torchaudio.load(audio_path)
    duration = wav.shape[1] / sr
    if duration >= min_sec:
        return audio_path

    print(f"[Phase 4B] Reference too short ({duration:.1f}s). Looping ...")
    repeats     = int(min_sec // duration) + 1
    looped      = wav.repeat(1, repeats)
    looped_path = str(audio_path).replace(".wav", "_looped.wav")
    torchaudio.save(looped_path, looped, sr)
    return looped_path


def estimate_chars(text: str) -> int:
    return int(len(text) * EST_MULT)


def duration_sec(wav: torch.Tensor) -> float:
    return wav.shape[1] / SR


def slice_speaker_prompt(
    coqui_wav: torch.Tensor, coqui_sr: int,
    start_sec: float, end_sec: float,
    block_id: int, tmp_dir: Path,
) -> Path | None:
    center = 0.5 * (start_sec + end_sec)
    half   = max(MIN_PROMPT_SEC / 2, (end_sec - start_sec) / 2)
    win_s  = max(0.0, center - half)
    win_e  = min(center + half, coqui_wav.shape[1] / coqui_sr)

    if win_e - win_s < MIN_PROMPT_SEC:
        return None
    if win_e - win_s > MAX_PROMPT_SEC:
        win_e = win_s + MAX_PROMPT_SEC

    s, e = int(win_s * coqui_sr), int(win_e * coqui_sr)
    wav  = coqui_wav[:, s:e]
    if wav.shape[1] / coqui_sr < MIN_PROMPT_SEC:
        return None

    p = tmp_dir / f"_cb_prompt_{block_id}.wav"
    torchaudio.save(str(p), wav, coqui_sr)
    return p


def make_fallback_prompt(coqui_wav: torch.Tensor, coqui_sr: int, tmp_dir: Path) -> Path:
    wav  = coqui_wav[:, : int(5.5 * coqui_sr)]
    path = tmp_dir / "_cb_prompt_fallback.wav"
    torchaudio.save(str(path), wav, coqui_sr)
    return path


def stitch(chunks: list[torch.Tensor], t_orig: float) -> torch.Tensor:
    t_gen     = sum(duration_sec(w) for w in chunks)
    t_silence = min(MAX_TOTAL_SILENCE, max(0.0, t_orig - t_gen))

    if t_silence <= 0:
        return torch.cat(chunks, dim=1)

    sil_per_gap = t_silence / (len(chunks) + 1)
    if sil_per_gap < MIN_GAP_SILENCE:
        return torch.cat(chunks, dim=1)

    silence  = torch.zeros(1, int(sil_per_gap * SR))
    stitched = []
    for c in chunks:
        stitched.append(silence)
        stitched.append(c)
    stitched.append(silence)
    return torch.cat(stitched, dim=1)


# ================================================================
# PROCESS ONE STRATEGY
# ================================================================

def process_strategy(strategy: str) -> None:
    output_path = BASE_DIR / f"tts_chatterbox_{strategy}.wav"

    # Per-strategy skip
    if output_path.exists() and output_path.stat().st_size >= MIN_AUDIO_BYTES:
        print(f"[Phase 4B] SKIP {strategy} — output already exists.")
        return

    print(f"\n[Phase 4B] Processing: {strategy}")

    with open(BASE_DIR / f"segments_{strategy}.json", "r", encoding="utf-8") as f:
        seg_rows = json.load(f)
    seg_rows.sort(key=lambda r: r["id"])

    coqui_wav, coqui_sr = torchaudio.load(BASE_DIR / f"tts_coqui_{strategy}.wav")
    if coqui_wav.shape[0] > 1:
        coqui_wav = coqui_wav.mean(dim=0, keepdim=True)

    tmp_dir = BASE_DIR / f"_cb_tmp_{strategy}"
    tmp_dir.mkdir(exist_ok=True)

    fallback_prompt: Path | None = None
    final_audio: list[torch.Tensor] = []
    current_block: list[dict] = []
    char_sum: int = 0

    def flush_block(block: list[dict]) -> None:
        nonlocal fallback_prompt
        if not block:
            return

        block_text = " ".join(r["translation"] for r in block if r.get("translation"))
        if not block_text.strip():
            return

        prompt_path = slice_speaker_prompt(
            coqui_wav, coqui_sr,
            block[0]["start"], block[-1]["end"],
            block[0]["id"], tmp_dir,
        )

        if prompt_path is None:
            if fallback_prompt is None:
                fallback_prompt = make_fallback_prompt(coqui_wav, coqui_sr, tmp_dir)
            prompt_path = fallback_prompt

        safe_prompt = ensure_min_duration(str(prompt_path))

        with torch.no_grad():
            wav = model.generate(text=block_text, audio_prompt_path=safe_prompt)
        final_audio.append(wav)

    for seg in seg_rows:
        translation = seg.get("translation", "")
        if not translation:
            continue
        est = estimate_chars(translation)
        if char_sum + est > MAX_CHARS:
            flush_block(current_block)
            current_block = []
            char_sum      = 0
        current_block.append(seg)
        char_sum += est

    flush_block(current_block)

    if not final_audio:
        print(f"  [WARN] No audio generated for: {strategy}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return

    final_wav = stitch(final_audio, T_ORIG)
    torchaudio.save(str(output_path), final_wav, SR)

    sz       = output_path.stat().st_size
    duration = duration_sec(final_wav)
    print(f"  → {output_path}  ({duration:.1f}s, {sz//1024} KB)")

    if sz < MIN_AUDIO_BYTES:
        print(f"  [WARN] Output suspiciously small ({sz} bytes) — possible silent failure")

    shutil.rmtree(tmp_dir, ignore_errors=True)


# ================================================================
# MAIN
# ================================================================

for strategy in STRATEGIES:
    process_strategy(strategy)

print("\n[Phase 4B COMPLETE]")
for strategy in STRATEGIES:
    path   = BASE_DIR / f"tts_chatterbox_{strategy}.wav"
    status = f"{path.stat().st_size // 1024} KB" if path.exists() else "MISSING"
    print(f"  {strategy:<12} → {path}  [{status}]")