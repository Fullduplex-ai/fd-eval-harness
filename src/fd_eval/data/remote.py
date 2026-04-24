import logging
from pathlib import Path
from typing import Optional, Sequence

try:
    from huggingface_hub import snapshot_download
    _HAS_HF_HUB = True
except ImportError:
    _HAS_HF_HUB = False

logger = logging.getLogger(__name__)


def download_hf_dataset(
    repo_id: str,
    *,
    repo_type: str = "dataset",
    revision: Optional[str] = None,
    allow_patterns: Optional[Sequence[str]] = None,
    ignore_patterns: Optional[Sequence[str]] = None,
) -> Path:
    """Download a dataset from Hugging Face Hub to the local cache.

    This function relies on `huggingface_hub.snapshot_download` to fetch files.
    It returns the local path to the cached dataset directory. If the dataset
    is already fully cached, it will return the cached path instantly.

    Args:
        repo_id: The Hugging Face repository ID (e.g., 'Fullduplex-ai/example-benchmark').
        repo_type: The type of the repository, usually 'dataset' or 'model'.
        revision: The git revision (branch, tag, or commit hash) to download.
        allow_patterns: A list of glob patterns to restrict which files are downloaded
            (e.g., `["*.wav", "*.json"]`). Useful for large datasets.
        ignore_patterns: A list of glob patterns to exclude from downloading.

    Returns:
        Path: The absolute path to the local cached directory.

    Raises:
        ImportError: If the `huggingface_hub` package is not installed.
    """
    if not _HAS_HF_HUB:
        raise ImportError(
            "The `huggingface_hub` package is required to download remote datasets. "
            "Please install it with `pip install huggingface_hub`."
        )

    logger.info(f"Downloading dataset '{repo_id}' from Hugging Face Hub...")
    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )
    
    logger.info(f"Dataset '{repo_id}' cached at: {local_dir}")
    return Path(local_dir)
