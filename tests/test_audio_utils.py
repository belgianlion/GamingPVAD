import numpy as np
import pytest

from audio_utils import CircularBuffer, Resampler


# ---------------------------------------------------------------------------
# Resampler
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def resampler() -> Resampler:
    return Resampler(input_rate=48000, model_rate=16000, output_rate=48000)


def test_resampler_down_length_and_dtype(resampler: Resampler) -> None:
    audio = np.random.uniform(-0.5, 0.5, size=1536).astype(np.float32)
    out = resampler.resample_input_to_model(audio)
    assert out.dtype == np.float32
    # 48k -> 16k is a 1:3 reduction; allow ±2 samples for filter edge handling.
    assert abs(out.shape[0] - 512) <= 2


def test_resampler_up_length_and_dtype(resampler: Resampler) -> None:
    audio = np.random.uniform(-0.5, 0.5, size=512).astype(np.float32)
    out = resampler.resample_model_to_output(audio)
    assert out.dtype == np.float32
    assert abs(out.shape[0] - 1536) <= 2


def test_resampler_preserves_silence(resampler: Resampler) -> None:
    out = resampler.resample_input_to_model(np.zeros(1536, dtype=np.float32))
    assert np.max(np.abs(out)) < 1e-6


def test_resampler_identity_when_rates_match() -> None:
    r = Resampler(input_rate=16000, model_rate=16000, output_rate=16000)
    audio = np.random.uniform(-1.0, 1.0, size=512).astype(np.float32)
    np.testing.assert_array_equal(r.resample_input_to_model(audio), audio)
    np.testing.assert_array_equal(r.resample_model_to_output(audio), audio)


# ---------------------------------------------------------------------------
# CircularBuffer
# ---------------------------------------------------------------------------

def test_circular_buffer_invalid_capacity() -> None:
    with pytest.raises(ValueError):
        CircularBuffer(0)
    with pytest.raises(ValueError):
        CircularBuffer(-1)


def test_circular_buffer_partial_fill() -> None:
    buf = CircularBuffer(10)
    buf.write(np.array([1, 2, 3, 4], dtype=np.float32))
    assert buf.filled == 4
    assert not buf.is_full
    np.testing.assert_array_equal(
        buf.read_last_n(4), np.array([1, 2, 3, 4], dtype=np.float32)
    )


def test_circular_buffer_full_fill() -> None:
    buf = CircularBuffer(5)
    buf.write(np.arange(5, dtype=np.float32))
    assert buf.is_full
    np.testing.assert_array_equal(buf.read_last_n(5), np.arange(5, dtype=np.float32))


def test_circular_buffer_wraparound_keeps_most_recent() -> None:
    buf = CircularBuffer(5)
    buf.write(np.arange(8, dtype=np.float32))   # 0..7, only last 5 retained
    assert buf.is_full
    np.testing.assert_array_equal(
        buf.read_last_n(5), np.array([3, 4, 5, 6, 7], dtype=np.float32)
    )


def test_circular_buffer_multiple_writes_wrap_correctly() -> None:
    buf = CircularBuffer(5)
    buf.write(np.array([1, 2, 3], dtype=np.float32))
    buf.write(np.array([4, 5, 6, 7], dtype=np.float32))
    np.testing.assert_array_equal(
        buf.read_last_n(5), np.array([3, 4, 5, 6, 7], dtype=np.float32)
    )


def test_circular_buffer_oversized_write_keeps_tail() -> None:
    buf = CircularBuffer(4)
    buf.write(np.arange(10, dtype=np.float32))
    np.testing.assert_array_equal(
        buf.read_last_n(4), np.array([6, 7, 8, 9], dtype=np.float32)
    )


def test_circular_buffer_clear_resets_state() -> None:
    buf = CircularBuffer(4)
    buf.write(np.arange(4, dtype=np.float32))
    buf.clear()
    assert buf.filled == 0
    assert not buf.is_full
    with pytest.raises(ValueError):
        buf.read_last_n(1)


def test_circular_buffer_read_more_than_filled_raises() -> None:
    buf = CircularBuffer(10)
    buf.write(np.array([1, 2, 3], dtype=np.float32))
    with pytest.raises(ValueError):
        buf.read_last_n(4)


def test_circular_buffer_read_zero_returns_empty() -> None:
    buf = CircularBuffer(4)
    out = buf.read_last_n(0)
    assert out.shape == (0,)
    assert out.dtype == np.float32
