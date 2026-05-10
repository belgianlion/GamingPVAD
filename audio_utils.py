"""Audio helpers used by the real-time pipeline.

Both classes are designed to allocate at construction time only, so they are
safe to call from inside the `sounddevice` audio callback.
"""

from __future__ import annotations

import numpy as np
import torch
import torchaudio.functional as AF


class Resampler:
    """Wraps `torchaudio.functional.resample` for the two rates we care about."""

    def __init__(self, input_rate: int, model_rate: int, output_rate: int) -> None:
        self.input_rate = input_rate
        self.model_rate = model_rate
        self.output_rate = output_rate

    @staticmethod
    def _to_tensor(audio: np.ndarray) -> torch.Tensor:
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32, copy=False)
        return torch.from_numpy(audio)

    def resample_input_to_model(self, audio: np.ndarray) -> np.ndarray:
        """Resample mic-rate audio (e.g. 48 kHz) down to the model rate (16 kHz)."""
        if self.input_rate == self.model_rate:
            return audio.astype(np.float32, copy=False)
        out = AF.resample(self._to_tensor(audio), self.input_rate, self.model_rate)
        return out.numpy().astype(np.float32, copy=False)

    def resample_model_to_output(self, audio: np.ndarray) -> np.ndarray:
        """Resample model-rate audio (16 kHz) up to the output rate (e.g. 48 kHz)."""
        if self.model_rate == self.output_rate:
            return audio.astype(np.float32, copy=False)
        out = AF.resample(self._to_tensor(audio), self.model_rate, self.output_rate)
        return out.numpy().astype(np.float32, copy=False)


class CircularBuffer:
    """Fixed-capacity FIFO over a pre-allocated float32 numpy array.

    Used to accumulate ~512 ms of audio for ECAPA without allocating in the
    audio callback.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._buf = np.zeros(capacity, dtype=np.float32)
        self._write_idx = 0
        self._filled = 0

    @property
    def filled(self) -> int:
        """Number of valid samples currently stored (capped at capacity)."""
        return self._filled

    @property
    def is_full(self) -> bool:
        return self._filled >= self.capacity

    def clear(self) -> None:
        self._write_idx = 0
        self._filled = 0

    def write(self, frames: np.ndarray) -> None:
        """Append `frames` to the buffer, overwriting oldest data on wrap."""
        n = frames.shape[0]
        if n == 0:
            return
        if n >= self.capacity:
            # Only the last `capacity` samples are retained.
            self._buf[:] = frames[-self.capacity:].astype(np.float32, copy=False)
            self._write_idx = 0
            self._filled = self.capacity
            return

        end = self._write_idx + n
        if end <= self.capacity:
            self._buf[self._write_idx:end] = frames
        else:
            first = self.capacity - self._write_idx
            self._buf[self._write_idx:] = frames[:first]
            self._buf[:n - first] = frames[first:]
        self._write_idx = end % self.capacity
        self._filled = min(self.capacity, self._filled + n)

    def read_last_n(self, n: int) -> np.ndarray:
        """Return the most recent `n` samples in chronological order.

        Raises ValueError if `n` exceeds the number of samples filled.
        """
        if n > self._filled:
            raise ValueError(f"requested {n} samples but only {self._filled} filled")
        if n == 0:
            return np.zeros(0, dtype=np.float32)

        # The most recent sample is at (write_idx - 1) mod capacity.
        # Walk backwards `n` samples from there.
        start = (self._write_idx - n) % self.capacity
        end = start + n
        if end <= self.capacity:
            return self._buf[start:end].copy()
        first = self.capacity - start
        out = np.empty(n, dtype=np.float32)
        out[:first] = self._buf[start:]
        out[first:] = self._buf[:n - first]
        return out
