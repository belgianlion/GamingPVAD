# GamingPVAD

Personal VAD virtual microphone for games. Captures your mic, denoises it
with DeepFilterNet, gates it on **your** voice using Silero VAD plus an
ECAPA-TDNN speaker check, and pipes the result into a virtual mic so any
game (including those without preprocessing hooks like REPO) receives only
your cleaned voice.

## Pipeline

```
Mic (48 kHz)
  -> DeepFilterNet3 denoise (48 kHz)
  -> Resample to 16 kHz (analysis path)
  -> Silero VAD  ──► CONFIRMED? ──► state machine gate
  -> ECAPA-TDNN  ──┘
  -> Output: gated 48 kHz denoised audio -> VB-Cable virtual mic
```

The state machine has four states (SILENCE / BUFFERING / CONFIRMED /
REJECTED) and re-checks the speaker every ~800 ms while CONFIRMED.

## Requirements

- Windows 10/11.
- [UV](https://docs.astral.sh/uv/) for Python dependency management.
- [VB-Audio Virtual Cable](https://vb-audio.com/Cable/) installed.

## Setup

1. Install VB-Audio Virtual Cable.
2. Clone this repo and `cd` into it.
3. `uv sync` — creates `.venv/` and installs locked dependencies (PyTorch
   download is ~200 MB on first run).
4. `uv run python enroll.py` — records 8 s of your voice and saves
   `enrollment.pt`. The first run also downloads the ECAPA model from
   Hugging Face.
5. Open `config.py` and set `MIC_DEVICE_INDEX` and `OUT_DEVICE_INDEX` from
   the device list printed by step 4. `OUT_DEVICE_INDEX` left as `None` is
   auto-detected by name "CABLE Input" — fine if you only have one VB-Cable.
6. In Windows Sound Settings, set "CABLE Output" as the default microphone.
7. `uv run python pipeline.py` — runs the live pipeline. Ctrl+C to stop.

## Tests

```
uv run pytest
```

Covers `audio_utils.py` (resampler shapes, circular buffer wraparound) and
`state_machine.py` (every transition, with a mocked verifier).

## Tuning

All thresholds live in `config.py`.

- **My own voice gets rejected**: lower `SPEAKER_THRESHOLD` toward 0.15.
- **Partner / TV bleeds through**: raise `SPEAKER_THRESHOLD` toward 0.35.
- **First syllable gets clipped**: lower `VAD_THRESHOLD`. Note that the
  first ~512 ms of a fresh utterance is BUFFERING — the pipeline passes
  audio through speculatively during that window, so any bleed in
  REJECTED outcomes is bounded by `ECAPA_BUFFER_MS`.
- A longer or cleaner enrollment clip improves ECAPA accuracy
  significantly. Re-record (`uv run python enroll.py`) any time you want
  to refresh.

## Layout

```
config.py           tunable constants
audio_utils.py      Resampler, CircularBuffer (no heap alloc on hot path)
models.py           SileroVAD, SpeakerVerifier, Denoiser wrappers
state_machine.py    SILENCE / BUFFERING / CONFIRMED / REJECTED
enroll.py           one-time 8 s enrollment
pipeline.py         main entry point, opens sounddevice.Stream
tests/              pytest coverage of pure-logic modules
```

## Notes / known limitations

- DeepFilterNet 0.5.6 still imports from `torchaudio.backend`, removed in
  torchaudio 2.1. `pyproject.toml` pins torch and torchaudio to the 2.0.x
  line until upstream releases a fix.
- SpeechBrain 1.0.x calls `huggingface_hub.hf_hub_download(use_auth_token=...)`,
  removed in `huggingface_hub` 0.24+. Pinned to <0.24 for the same reason.
- On Windows, SpeechBrain falls back to copying ECAPA assets out of the HF
  cache (no symlinks without dev-mode/admin) — handled automatically via
  `LocalStrategy.COPY`.
- The pipeline is currently Windows-only. Linux/PipeWire support is out of
  scope for the initial release.
- DeepFilterNet3 runs at 48 kHz natively, so the pipeline denoises at the
  input rate and only downsamples for the VAD/ECAPA analysis path. (The
  original `plan.md` placed the resampler before the denoiser, which would
  feed DFN at the wrong rate — this implementation deviates intentionally.)
