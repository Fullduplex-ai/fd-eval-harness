import json
from pathlib import Path

import numpy as np
import soundfile as sf


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Load audio from file and return it as a NumPy array with the sample rate.

    Returns
    -------
    audio : np.ndarray
        Audio data (shape: (frames, channels)).
    sample_rate : int
        Sample rate.
    """
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return audio, sr


def load_labels(path: str | Path) -> list[dict]:
    """Load JSON labels from file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of dicts in {path}, got {type(data).__name__}")
    return data
