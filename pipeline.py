"""Real-time voice isolation pipeline.

Wires the mic → DeepFilterNet → resample → Silero VAD → ECAPA gate → output
chain inside a `sounddevice.Stream`. Run after `python enroll.py`.

Architecture note: DeepFilterNet3 runs natively at 48 kHz, so we denoise
first at the input rate and only downsample the analysis path. The output
is the gated 48 kHz denoised audio — no upsample stage at the end.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd

from audio_utils import Resampler
from config import (
    BLOCK_SAMPLES_INPUT,
    ENROLLMENT_PATH,
    INPUT_SAMPLE_RATE,
    MIC_DEVICE_INDEX,
    MODEL_SAMPLE_RATE,
    OUTPUT_SAMPLE_RATE,
    OUT_DEVICE_INDEX,
)
from models import Denoiser, SileroVAD, SpeakerVerifier
from state_machine import StateMachine


def print_devices() -> None:
    print("Audio devices:")
    for idx, dev in enumerate(sd.query_devices()):
        kind = []
        if dev["max_input_channels"] > 0:
            kind.append("in")
        if dev["max_output_channels"] > 0:
            kind.append("out")
        print(f"  [{idx}] ({'/'.join(kind):>5})  {dev['name']}")


def find_vbcable_output() -> int | None:
    """Return the device index of VB-Cable's input ("CABLE Input"), or None."""
    for idx, dev in enumerate(sd.query_devices()):
        name = dev["name"].lower()
        if dev["max_output_channels"] > 0 and "cable input" in name:
            return idx
    return None


def raise_priority() -> None:
    """Bump the process priority on Windows; best-effort, log on failure."""
    try:
        import psutil

        psutil.Process().nice(psutil.HIGH_PRIORITY_CLASS)
    except Exception as exc:  # pragma: no cover — depends on host OS
        print(f"[!] Could not raise process priority: {exc}")


def main() -> int:
    if not Path(ENROLLMENT_PATH).exists():
        print(
            f"[!] Enrollment file '{ENROLLMENT_PATH}' not found.\n"
            "    Run `uv run python enroll.py` first to record your voice."
        )
        return 1

    print_devices()

    out_device = OUT_DEVICE_INDEX
    if out_device is None:
        detected = find_vbcable_output()
        if detected is not None:
            print(f"\n[i] Auto-detected VB-Cable input at device index {detected}.")
            out_device = detected
        else:
            print(
                "\n[!] OUT_DEVICE_INDEX is None and 'CABLE Input' was not found.\n"
                "    Install VB-Audio Virtual Cable from https://vb-audio.com/Cable/\n"
                "    or set OUT_DEVICE_INDEX in config.py to an explicit index."
            )
            return 1

    if MIC_DEVICE_INDEX is None:
        print(
            "\n[!] MIC_DEVICE_INDEX is None in config.py — using the system default.\n"
            "    Set it to the index of your mic for reliable results."
        )
    in_device = MIC_DEVICE_INDEX

    print("\nLoading models (first run downloads weights)...")
    denoiser = Denoiser()
    vad = SileroVAD()
    verifier = SpeakerVerifier()
    enrollment = SpeakerVerifier.load_enrollment(ENROLLMENT_PATH)
    state_machine = StateMachine(verifier, enrollment)
    resampler = Resampler(INPUT_SAMPLE_RATE, MODEL_SAMPLE_RATE, OUTPUT_SAMPLE_RATE)
    print("Models loaded.")

    raise_priority()

    stop_event = threading.Event()

    def callback(indata, outdata, frames, time, status):  # noqa: ARG001
        if status:
            # Underflows / overflows surface here; print but do not raise so
            # the stream keeps running.
            print(f"[stream status] {status}", file=sys.stderr)

        # Mono float32 — `indata` is shape (frames, 1).
        mic = indata[:, 0]
        denoised_48k = denoiser.process(mic)
        analysis_16k = resampler.resample_input_to_model(denoised_48k)
        # Silero requires exactly 512 samples at 16 kHz; trim/pad if the
        # resampler returned a slightly different length due to filter edges.
        if analysis_16k.shape[0] > 512:
            analysis_16k = analysis_16k[:512]
        elif analysis_16k.shape[0] < 512:
            padded = np.zeros(512, dtype=np.float32)
            padded[: analysis_16k.shape[0]] = analysis_16k
            analysis_16k = padded

        vad_conf = vad.predict(analysis_16k)
        out_48k = state_machine.process(denoised_48k, analysis_16k, vad_conf)
        outdata[:, 0] = out_48k

    print(
        f"\nOpening stream:\n"
        f"  in  device : {in_device}\n"
        f"  out device : {out_device}\n"
        f"  block      : {BLOCK_SAMPLES_INPUT} samples ({BLOCK_SAMPLES_INPUT / INPUT_SAMPLE_RATE * 1000:.0f} ms)"
    )
    try:
        with sd.Stream(
            samplerate=INPUT_SAMPLE_RATE,
            blocksize=BLOCK_SAMPLES_INPUT,
            dtype="float32",
            channels=1,
            device=(in_device, out_device),
            callback=callback,
        ):
            print("Pipeline running. Ctrl+C to stop.")
            try:
                stop_event.wait()
            except KeyboardInterrupt:
                pass
    finally:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    # Quieten DeepFilterNet's INFO logging during runtime.
    os.environ.setdefault("DF_LOG_LEVEL", "WARNING")
    sys.exit(main())
