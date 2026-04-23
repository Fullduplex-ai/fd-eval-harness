import numpy as np

from .audio_session import AudioSession


def collect_all(session: AudioSession) -> np.ndarray:
    """
    Consume the entire chunk iterator from an AudioSession and return a single array.
    This provides an escape hatch for v0.1 offline models.
    """
    chunks = list(session.stream())
    if not chunks:
        # Return an empty array with the correct number of channels
        has_multi_dim = hasattr(session, "audio") and session.audio.ndim > 1
        num_channels = session.audio.shape[1] if has_multi_dim else 1
        return np.empty((0, num_channels))
    return np.concatenate(chunks, axis=0)
