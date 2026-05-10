"""Tunable constants for the voice isolation pipeline.

A note on rates: DeepFilterNet3 operates natively at 48 kHz, so the pipeline
denoises at 48 kHz first and only downsamples to 16 kHz for the analysis path
(Silero VAD + ECAPA). The final output is the 48 kHz denoised audio gated by
the state machine — no upsample needed at the end.
"""

# Sample rates
INPUT_SAMPLE_RATE = 48000   # mic + DeepFilterNet
MODEL_SAMPLE_RATE = 16000   # Silero VAD + ECAPA
OUTPUT_SAMPLE_RATE = 48000  # what we hand to VB-Cable

# Block sizing — Silero VAD requires exactly 512 samples at 16 kHz (32 ms).
BLOCK_MS = 32
BLOCK_SAMPLES_MODEL = BLOCK_MS * MODEL_SAMPLE_RATE // 1000   # 512
BLOCK_SAMPLES_INPUT = BLOCK_MS * INPUT_SAMPLE_RATE // 1000   # 1536
BLOCK_SAMPLES_OUTPUT = BLOCK_MS * OUTPUT_SAMPLE_RATE // 1000  # 1536

# ECAPA buffering — accumulate this much 16 kHz audio before running speaker
# verification.
ECAPA_BUFFER_MS = 512
ECAPA_BUFFER_BLOCKS = ECAPA_BUFFER_MS // BLOCK_MS   # 16
ECAPA_BUFFER_SAMPLES = ECAPA_BUFFER_BLOCKS * BLOCK_SAMPLES_MODEL  # 8192

# Speculative output buffer — pre-allocated to hold one ECAPA window worth
# of 48 kHz denoised frames (released or zeroed when ECAPA decides).
SPECULATIVE_BUFFER_SAMPLES = ECAPA_BUFFER_BLOCKS * BLOCK_SAMPLES_INPUT  # 24576

# Re-verify the speaker every ~800 ms while CONFIRMED.
RECHECK_BLOCKS = 25

# Drop back to SILENCE after this many consecutive non-speech blocks (~200 ms).
SILENCE_DEBOUNCE_BLOCKS = 7

# Thresholds — tune empirically after enrollment.
VAD_THRESHOLD = 0.5
SPEAKER_THRESHOLD = 0.25

# Audio device indices — set after running enroll.py / pipeline.py to see the
# device list. None means "use system default" (warned about on startup).
MIC_DEVICE_INDEX = None
OUT_DEVICE_INDEX = None

# Paths
ENROLLMENT_PATH = "enrollment.pt"
ECAPA_SAVEDIR = "pretrained_models/spkrec-ecapa-voxceleb"
