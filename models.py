"""Wrappers around the three models the pipeline uses.

Each wrapper holds persistent state (LSTM hidden state for Silero, DFState for
DeepFilterNet) and exposes a small per-frame API the audio callback can call.
Initialisation is heavy (downloads + model load); never re-init mid-stream.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from config import ECAPA_SAVEDIR, INPUT_SAMPLE_RATE, MODEL_SAMPLE_RATE


class SileroVAD:
    """Silero VAD wrapper. Stateful — feed it 32 ms blocks @ 16 kHz."""

    def __init__(self) -> None:
        model, _utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            verbose=False,
        )
        self.model = model
        # Internal LSTM state lives inside `model` and persists across calls.

    def predict(self, frame_16k: np.ndarray) -> float:
        """Return speech confidence in [0, 1] for a 512-sample @ 16 kHz frame."""
        tensor = torch.from_numpy(np.ascontiguousarray(frame_16k, dtype=np.float32))
        with torch.no_grad():
            conf = self.model(tensor, MODEL_SAMPLE_RATE)
        return float(conf.item())

    def reset(self) -> None:
        """Reset the LSTM state. Only call this between unrelated streams."""
        self.model.reset_states()


class SpeakerVerifier:
    """ECAPA-TDNN speaker embedding + cosine similarity."""

    def __init__(self, savedir: str = ECAPA_SAVEDIR) -> None:
        # Imported here so the heavy speechbrain import only happens when we
        # actually instantiate the verifier.
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy

        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=savedir,
            # Windows can't symlink without dev-mode/admin; copy the files
            # locally instead.
            local_strategy=LocalStrategy.COPY,
        )

    def embed(self, audio_16k: np.ndarray) -> torch.Tensor:
        """Return a [192] float tensor for the supplied 16 kHz audio."""
        # Clip to [-1, 1] to avoid NaN embeddings on out-of-range input.
        clipped = np.clip(audio_16k, -1.0, 1.0).astype(np.float32, copy=False)
        tensor = torch.from_numpy(np.ascontiguousarray(clipped))[None, :]   # [1, N]
        with torch.no_grad():
            emb = self.model.encode_batch(tensor)   # [1, 1, 192]
        return emb.squeeze().detach()

    @staticmethod
    def compare(emb_a: torch.Tensor, emb_b: torch.Tensor) -> float:
        """Cosine similarity in [-1, 1] between two 1-D embedding tensors."""
        sim = torch.nn.functional.cosine_similarity(
            emb_a.flatten().unsqueeze(0),
            emb_b.flatten().unsqueeze(0),
        )
        return float(sim.item())

    @staticmethod
    def load_enrollment(path: str) -> torch.Tensor:
        return torch.load(path, map_location="cpu", weights_only=False)


class Denoiser:
    """DeepFilterNet3 stateful streaming denoiser. Runs natively at 48 kHz."""

    def __init__(self) -> None:
        from df.enhance import init_df, enhance as df_enhance

        model, df_state, _ = init_df()
        self.model = model
        self.df_state = df_state
        self._enhance = df_enhance
        self.sample_rate = df_state.sr()
        if self.sample_rate != INPUT_SAMPLE_RATE:
            # If a future DFN release changes the rate, we want to fail loudly
            # rather than silently send mismatched audio through the model.
            raise RuntimeError(
                f"DeepFilterNet expected {INPUT_SAMPLE_RATE} Hz but reports "
                f"{self.sample_rate} Hz; pipeline rates need adjusting."
            )

    def process(self, frame_48k: np.ndarray) -> np.ndarray:
        """Denoise one block of 48 kHz audio; preserves persistent state."""
        tensor = torch.from_numpy(
            np.ascontiguousarray(frame_48k, dtype=np.float32)
        )[None, :]   # [1, N]
        with torch.no_grad():
            out = self._enhance(self.model, self.df_state, tensor)
        return out.squeeze(0).numpy().astype(np.float32, copy=False)
