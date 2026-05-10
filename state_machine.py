"""State machine that decides whether to pass each output frame through.

The four states model the lifecycle of a single speech burst:

    SILENCE   — VAD quiet; output muted.
    BUFFERING — VAD fired but we haven't yet confirmed the speaker. We pass
                the (already-denoised) audio through speculatively so the
                onset of the user's speech is not clipped. If ECAPA later
                rejects, audio gets muted from that block onwards (we cannot
                rewind audio already handed to the OS).
    CONFIRMED — ECAPA accepted the speaker. Audio flows freely; we re-run
                ECAPA on a rolling window every RECHECK_BLOCKS blocks to
                catch a speaker change mid-burst.
    REJECTED  — ECAPA rejected the speaker. Output muted until VAD fires
                again (a new speech onset gets a fresh chance).
"""

from __future__ import annotations

import enum
from typing import Protocol

import numpy as np
import torch

from audio_utils import CircularBuffer
from config import (
    ECAPA_BUFFER_BLOCKS,
    ECAPA_BUFFER_SAMPLES,
    RECHECK_BLOCKS,
    SILENCE_DEBOUNCE_BLOCKS,
    SPEAKER_THRESHOLD,
    VAD_THRESHOLD,
)


class State(enum.Enum):
    SILENCE = "SILENCE"
    BUFFERING = "BUFFERING"
    CONFIRMED = "CONFIRMED"
    REJECTED = "REJECTED"


class _VerifierProtocol(Protocol):
    def embed(self, audio_16k: np.ndarray) -> torch.Tensor: ...

    @staticmethod
    def compare(a: torch.Tensor, b: torch.Tensor) -> float: ...


class StateMachine:
    """Per-block decision logic.

    Owns no audio I/O; the caller provides each block and gets back either
    the same block (pass) or zeros (mute).
    """

    def __init__(
        self,
        verifier: _VerifierProtocol,
        enrollment: torch.Tensor,
        speaker_threshold: float = SPEAKER_THRESHOLD,
        vad_threshold: float = VAD_THRESHOLD,
    ) -> None:
        self.verifier = verifier
        self.enrollment = enrollment
        self.speaker_threshold = speaker_threshold
        self.vad_threshold = vad_threshold

        self.state = State.SILENCE
        self._ecapa_buf = CircularBuffer(ECAPA_BUFFER_SAMPLES)
        self._buffering_blocks = 0    # blocks since BUFFERING started
        self._recheck_counter = 0     # blocks since last CONFIRMED-mode check
        self._silence_counter = 0     # consecutive sub-threshold blocks in CONFIRMED
        self.last_similarity: float = 0.0   # exposed for diagnostics

    # ---- main entry point -------------------------------------------------

    def process(
        self,
        frame_out: np.ndarray,
        frame_16k: np.ndarray,
        vad_confidence: float,
    ) -> np.ndarray:
        """Decide whether to pass `frame_out` through or mute it.

        `frame_out` is the (denoised) output-rate frame; `frame_16k` is the
        16 kHz analysis frame fed to VAD/ECAPA. Returns either `frame_out`
        unchanged or a zero array of the same shape.
        """
        speech = vad_confidence >= self.vad_threshold

        # Keep the rolling 16 kHz window topped up. CONFIRMED rechecks read
        # the most recent ECAPA_BUFFER_SAMPLES from this buffer.
        self._ecapa_buf.write(frame_16k)

        if self.state is State.SILENCE:
            return self._step_silence(frame_out, speech)
        if self.state is State.BUFFERING:
            return self._step_buffering(frame_out, speech)
        if self.state is State.CONFIRMED:
            return self._step_confirmed(frame_out, speech)
        if self.state is State.REJECTED:
            return self._step_rejected(frame_out, speech)
        raise AssertionError(f"unreachable state: {self.state!r}")

    # ---- per-state handlers ----------------------------------------------

    def _step_silence(self, frame_out: np.ndarray, speech: bool) -> np.ndarray:
        if speech:
            self._enter_buffering()
            # First block of speech — speculative pass-through.
            return frame_out
        return np.zeros_like(frame_out)

    def _step_buffering(self, frame_out: np.ndarray, speech: bool) -> np.ndarray:
        self._buffering_blocks += 1
        if self._buffering_blocks >= ECAPA_BUFFER_BLOCKS:
            # We have enough audio to verify.
            decision = self._run_ecapa()
            if decision:
                self.state = State.CONFIRMED
                self._recheck_counter = 0
                self._silence_counter = 0
                return frame_out
            self.state = State.REJECTED
            return np.zeros_like(frame_out)
        # Still buffering — keep passing audio through speculatively.
        return frame_out

    def _step_confirmed(self, frame_out: np.ndarray, speech: bool) -> np.ndarray:
        if speech:
            self._silence_counter = 0
        else:
            self._silence_counter += 1
            if self._silence_counter >= SILENCE_DEBOUNCE_BLOCKS:
                self.state = State.SILENCE
                self._reset_counters()
                return np.zeros_like(frame_out)

        self._recheck_counter += 1
        if self._recheck_counter >= RECHECK_BLOCKS:
            self._recheck_counter = 0
            if not self._run_ecapa():
                self.state = State.REJECTED
                return np.zeros_like(frame_out)
        return frame_out

    def _step_rejected(self, frame_out: np.ndarray, speech: bool) -> np.ndarray:
        if speech:
            self._enter_buffering()
            return frame_out
        return np.zeros_like(frame_out)

    # ---- helpers ----------------------------------------------------------

    def _enter_buffering(self) -> None:
        self.state = State.BUFFERING
        self._buffering_blocks = 1   # the current block counts as block #1
        # We deliberately do NOT clear _ecapa_buf — keeping the recent 16 k
        # tail means the upcoming ECAPA check sees a smooth window.

    def _reset_counters(self) -> None:
        self._buffering_blocks = 0
        self._recheck_counter = 0
        self._silence_counter = 0

    def _run_ecapa(self) -> bool:
        if self._ecapa_buf.filled < ECAPA_BUFFER_SAMPLES:
            # Not enough audio yet; treat as "not yet decided" → keep
            # buffering. Caller transitions us only when the buffer is full,
            # so this guard is mostly defensive.
            return False
        window = self._ecapa_buf.read_last_n(ECAPA_BUFFER_SAMPLES)
        embedding = self.verifier.embed(window)
        sim = self.verifier.compare(embedding, self.enrollment)
        self.last_similarity = sim
        return sim >= self.speaker_threshold
