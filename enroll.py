"""One-time enrollment: record 8 s of the user's voice and save its embedding.

Run once before `pipeline.py` so the pipeline has a target speaker to gate
on. Saves to `config.ENROLLMENT_PATH` (`enrollment.pt` by default).
"""

from __future__ import annotations

import sys

import numpy as np
import sounddevice as sd
import torch

from audio_utils import Resampler
from config import (
    ENROLLMENT_PATH,
    INPUT_SAMPLE_RATE,
    MIC_DEVICE_INDEX,
    MODEL_SAMPLE_RATE,
    OUTPUT_SAMPLE_RATE,
)
from models import SpeakerVerifier

ENROLL_SECONDS = 8


def list_input_devices() -> None:
    print("Available input devices:")
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            print(f"  [{idx}] {dev['name']}  ({int(dev['default_samplerate'])} Hz)")


def main() -> int:
    list_input_devices()
    device = MIC_DEVICE_INDEX
    if device is None:
        print(
            "\n[!] MIC_DEVICE_INDEX is None in config.py — using the system default.\n"
            "    Set it explicitly to the index of the mic you want to enroll.\n"
        )

    input(f"Press Enter to start a {ENROLL_SECONDS}-second enrollment recording... ")
    print("Recording...")
    recording = sd.rec(
        frames=ENROLL_SECONDS * INPUT_SAMPLE_RATE,
        samplerate=INPUT_SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=device,
    )
    sd.wait()
    audio_48k = recording[:, 0]
    print(f"Captured {audio_48k.shape[0]} samples @ {INPUT_SAMPLE_RATE} Hz "
          f"(peak {float(np.max(np.abs(audio_48k))):.3f}).")

    if float(np.max(np.abs(audio_48k))) < 0.01:
        print("[!] Recording is nearly silent — check the mic and try again.")
        return 1

    resampler = Resampler(INPUT_SAMPLE_RATE, MODEL_SAMPLE_RATE, OUTPUT_SAMPLE_RATE)
    audio_16k = np.clip(resampler.resample_input_to_model(audio_48k), -1.0, 1.0)

    print("Loading ECAPA model (first run downloads weights)...")
    verifier = SpeakerVerifier()
    embedding = verifier.embed(audio_16k)
    self_sim = verifier.compare(embedding, embedding)
    print(f"Self-similarity (sanity check): {self_sim:.4f}  (expected ~1.0)")

    torch.save(embedding, ENROLLMENT_PATH)
    print(f"Saved enrollment embedding to {ENROLLMENT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
