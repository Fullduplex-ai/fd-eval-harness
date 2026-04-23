from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np


@dataclass
class AudioSession:
    audio: np.ndarray
    sample_rate: int
    input_channel_indices: list[int]
    target_channel_indices: list[int]

    def __post_init__(self):
        # Implicitly treat 1D array as (N, 1)
        if self.audio.ndim == 1:
            self.audio = self.audio.reshape(-1, 1)

        num_channels = self.audio.shape[1]

        in_set = set(self.input_channel_indices)
        tgt_set = set(self.target_channel_indices)

        if not in_set.isdisjoint(tgt_set):
            raise ValueError(f"Input and target channels overlap: {in_set.intersection(tgt_set)}")

        for ch in in_set.union(tgt_set):
            if not (0 <= ch < num_channels):
                raise ValueError(
                    f"Channel index {ch} out of bounds for audio with {num_channels} channels."
                )

    def stream(self, chunk_ms: int = 20) -> Iterator[np.ndarray]:
        """Yields audio in chunks of chunk_ms."""
        if chunk_ms <= 0:
            raise ValueError("chunk_ms must be greater than 0")

        chunk_samples = int(self.sample_rate * chunk_ms / 1000)
        if chunk_samples == 0:
            chunk_samples = 1  # Avoid infinite loop if sample rate or chunk_ms is very small

        total_samples = self.audio.shape[0]
        for i in range(0, total_samples, chunk_samples):
            yield self.audio[i : i + chunk_samples]
