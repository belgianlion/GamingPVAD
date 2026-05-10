"""Tests for the speaker-gating state machine.

We mock the speaker verifier so we can drive ECAPA decisions
deterministically — the goal is to exercise transitions and counters, not
the embedding model itself.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from config import (
    BLOCK_SAMPLES_INPUT,
    BLOCK_SAMPLES_MODEL,
    ECAPA_BUFFER_BLOCKS,
    RECHECK_BLOCKS,
    SILENCE_DEBOUNCE_BLOCKS,
)
from state_machine import State, StateMachine


class FakeVerifier:
    """Stand-in for SpeakerVerifier with a programmable decision."""

    def __init__(self, scores: list[float] | None = None, default: float = 1.0) -> None:
        self.calls = 0
        self.default = default
        self.scores = list(scores) if scores else []

    def embed(self, audio_16k: np.ndarray) -> torch.Tensor:
        # Return any deterministic 192-vector; only `compare` matters here.
        return torch.zeros(192)

    def compare(self, a: torch.Tensor, b: torch.Tensor) -> float:
        self.calls += 1
        if self.scores:
            return self.scores.pop(0)
        return self.default


@pytest.fixture
def enrollment() -> torch.Tensor:
    return torch.zeros(192)


def make_frame_out() -> np.ndarray:
    return np.full(BLOCK_SAMPLES_INPUT, 0.5, dtype=np.float32)


def make_frame_16k() -> np.ndarray:
    return np.full(BLOCK_SAMPLES_MODEL, 0.5, dtype=np.float32)


def is_zero(frame: np.ndarray) -> bool:
    return np.all(frame == 0.0)


# ---------------------------------------------------------------------------
# SILENCE
# ---------------------------------------------------------------------------

def test_silence_outputs_zeros_when_no_speech(enrollment: torch.Tensor) -> None:
    sm = StateMachine(FakeVerifier(default=1.0), enrollment)
    out = sm.process(make_frame_out(), make_frame_16k(), vad_confidence=0.1)
    assert is_zero(out)
    assert sm.state is State.SILENCE


def test_silence_to_buffering_on_speech(enrollment: torch.Tensor) -> None:
    sm = StateMachine(FakeVerifier(default=1.0), enrollment)
    out = sm.process(make_frame_out(), make_frame_16k(), vad_confidence=0.9)
    # Speculative pass-through on the first block of speech.
    np.testing.assert_array_equal(out, make_frame_out())
    assert sm.state is State.BUFFERING


# ---------------------------------------------------------------------------
# BUFFERING
# ---------------------------------------------------------------------------

def test_buffering_to_confirmed_on_pass(enrollment: torch.Tensor) -> None:
    verifier = FakeVerifier(default=0.9)   # >= SPEAKER_THRESHOLD
    sm = StateMachine(verifier, enrollment)
    # Drive enough blocks to fill the ECAPA window.
    for _ in range(ECAPA_BUFFER_BLOCKS):
        out = sm.process(make_frame_out(), make_frame_16k(), vad_confidence=0.9)
        assert not is_zero(out)   # speculative pass-through throughout
    assert sm.state is State.CONFIRMED
    assert verifier.calls == 1


def test_buffering_to_rejected_on_fail(enrollment: torch.Tensor) -> None:
    verifier = FakeVerifier(default=0.0)   # below threshold
    sm = StateMachine(verifier, enrollment)
    last = None
    for _ in range(ECAPA_BUFFER_BLOCKS):
        last = sm.process(make_frame_out(), make_frame_16k(), vad_confidence=0.9)
    assert sm.state is State.REJECTED
    # The block on which rejection fired is muted.
    assert is_zero(last)


# ---------------------------------------------------------------------------
# CONFIRMED
# ---------------------------------------------------------------------------

def _drive_to_confirmed(verifier: FakeVerifier, enrollment: torch.Tensor) -> StateMachine:
    sm = StateMachine(verifier, enrollment)
    for _ in range(ECAPA_BUFFER_BLOCKS):
        sm.process(make_frame_out(), make_frame_16k(), vad_confidence=0.9)
    assert sm.state is State.CONFIRMED
    return sm


def test_confirmed_passes_audio_through(enrollment: torch.Tensor) -> None:
    sm = _drive_to_confirmed(FakeVerifier(default=0.9), enrollment)
    out = sm.process(make_frame_out(), make_frame_16k(), vad_confidence=0.9)
    np.testing.assert_array_equal(out, make_frame_out())
    assert sm.state is State.CONFIRMED


def test_confirmed_drops_to_silence_after_debounce(enrollment: torch.Tensor) -> None:
    sm = _drive_to_confirmed(FakeVerifier(default=0.9), enrollment)
    # Below-threshold blocks for silence_debounce_blocks consecutive blocks.
    for _ in range(SILENCE_DEBOUNCE_BLOCKS - 1):
        out = sm.process(make_frame_out(), make_frame_16k(), vad_confidence=0.0)
        assert sm.state is State.CONFIRMED
        np.testing.assert_array_equal(out, make_frame_out())
    out = sm.process(make_frame_out(), make_frame_16k(), vad_confidence=0.0)
    assert sm.state is State.SILENCE
    assert is_zero(out)


def test_confirmed_recheck_keeps_state_when_pass(enrollment: torch.Tensor) -> None:
    verifier = FakeVerifier(default=0.9)
    sm = _drive_to_confirmed(verifier, enrollment)
    initial_calls = verifier.calls
    for _ in range(RECHECK_BLOCKS):
        sm.process(make_frame_out(), make_frame_16k(), vad_confidence=0.9)
    assert sm.state is State.CONFIRMED
    assert verifier.calls == initial_calls + 1


def test_confirmed_recheck_drops_to_rejected_on_fail(enrollment: torch.Tensor) -> None:
    # Pass once to confirm, then fail on the recheck.
    verifier = FakeVerifier(scores=[0.9, 0.0])
    sm = _drive_to_confirmed(verifier, enrollment)
    out = None
    for _ in range(RECHECK_BLOCKS):
        out = sm.process(make_frame_out(), make_frame_16k(), vad_confidence=0.9)
    assert sm.state is State.REJECTED
    assert is_zero(out)


# ---------------------------------------------------------------------------
# REJECTED
# ---------------------------------------------------------------------------

def test_rejected_stays_muted_without_speech(enrollment: torch.Tensor) -> None:
    verifier = FakeVerifier(default=0.0)
    sm = StateMachine(verifier, enrollment)
    for _ in range(ECAPA_BUFFER_BLOCKS):
        sm.process(make_frame_out(), make_frame_16k(), vad_confidence=0.9)
    assert sm.state is State.REJECTED
    out = sm.process(make_frame_out(), make_frame_16k(), vad_confidence=0.0)
    assert is_zero(out)
    assert sm.state is State.REJECTED


def test_rejected_to_buffering_on_new_speech(enrollment: torch.Tensor) -> None:
    verifier = FakeVerifier(default=0.0)
    sm = StateMachine(verifier, enrollment)
    for _ in range(ECAPA_BUFFER_BLOCKS):
        sm.process(make_frame_out(), make_frame_16k(), vad_confidence=0.9)
    assert sm.state is State.REJECTED
    # A new speech block should re-enter BUFFERING and pass through.
    out = sm.process(make_frame_out(), make_frame_16k(), vad_confidence=0.9)
    assert sm.state is State.BUFFERING
    np.testing.assert_array_equal(out, make_frame_out())
