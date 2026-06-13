import sys
import shutil
import importlib

print("=" * 60)
print(" EMOTION-PRESERVING S2ST — ENVIRONMENT CHECK")
print("=" * 60)

# ----------------------------
# 1. Python version check
# ----------------------------
print("\n[1] Python Version Check")
major, minor = sys.version_info[:2]
print(f"Detected Python version: {major}.{minor}")

if major == 3 and minor == 10:
    print("✅ Python version OK (3.10)")
else:
    print("❌ Python version NOT OK — required: Python 3.10.x")
    sys.exit(1)

# ----------------------------
# 2. PyTorch check (CPU only)
# ----------------------------
print("\n[2] PyTorch Check")
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print("⚠️ CUDA detected — but this project is CPU-only (not fatal)")
    else:
        print("✅ CPU-only PyTorch detected (correct)")

except Exception as e:
    print("❌ PyTorch not working")
    print(e)
    sys.exit(1)

# ----------------------------
# 3. Whisper check
# ----------------------------
print("\n[3] Whisper Check")
try:
    import whisper
    print("✅ Whisper imported successfully")
except Exception as e:
    print("❌ Whisper import failed")
    print(e)
    sys.exit(1)

# ----------------------------
# 4. ffmpeg check
# ----------------------------
print("\n[4] ffmpeg Check")
ffmpeg_path = shutil.which("ffmpeg")

if ffmpeg_path is None:
    print("❌ ffmpeg not found in PATH")
    sys.exit(1)
else:
    print(f"✅ ffmpeg found at: {ffmpeg_path}")

# ----------------------------
# 5. Transformers check
# ----------------------------
print("\n[5] Transformers Check")
try:
    import transformers
    print(f"Transformers version: {transformers.__version__}")
    print("✅ Transformers imported successfully")
except Exception as e:
    print("❌ Transformers import failed")
    print(e)
    sys.exit(1)

# ----------------------------
# 6. Audio libraries check
# ----------------------------
print("\n[6] Audio Libraries Check")
audio_libs = ["numpy", "scipy", "soundfile", "librosa"]

for lib in audio_libs:
    try:
        importlib.import_module(lib)
        print(f"✅ {lib} imported")
    except Exception as e:
        print(f"❌ {lib} failed to import")
        print(e)
        sys.exit(1)

# ----------------------------
# 7. Torch CPU sanity test
# ----------------------------
print("\n[7] Torch CPU Sanity Test")
try:
    x = torch.randn(500, 500)
    y = x @ x
    print("✅ Torch CPU computation successful")
except Exception as e:
    print("❌ Torch CPU computation failed")
    print(e)
    sys.exit(1)

# ----------------------------
# FINAL RESULT
# ----------------------------
print("\n" + "=" * 60)
print("🎉 ENVIRONMENT CHECK PASSED")
print("Your laptop is READY for the project.")
print("=" * 60)
