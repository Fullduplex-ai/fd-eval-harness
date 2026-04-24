import pytest
from pathlib import Path
from fd_eval.data import download_hf_dataset, load_audio

@pytest.mark.network
def test_remote_data_loading_e2e():
    """
    E2E Test for Remote Data Loading (v0.3).
    Downloads a small public dataset from Hugging Face and verifies
    that our existing `load_audio` pipeline can ingest it seamlessly.
    """
    repo_id = "Narsil/asr_dummy"
    
    # We restrict to just one specific file to make the test fast
    dataset_path = download_hf_dataset(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=["1.flac"]
    )
    
    assert dataset_path.exists(), "Dataset directory should exist"
    assert dataset_path.is_dir(), "Dataset path should be a directory"
    
    audio_file = dataset_path / "1.flac"
    assert audio_file.exists(), "Target audio file should exist in the cache"
    
    # Verify the audio can be loaded properly
    audio_array, sr = load_audio(audio_file)
    
    assert sr == 16000, "Expected sample rate to be 16kHz for this specific dataset"
    assert len(audio_array.shape) >= 1, "Audio should be loaded as a numpy array"
    assert audio_array.shape[0] > 0, "Audio array should not be empty"
