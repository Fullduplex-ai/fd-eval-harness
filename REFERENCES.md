# fd-eval-harness — External References

Reading list for contributors and for the Cursor agent building v0.1.

## Structural references (study the pattern, do not copy code)

- **EleutherAI `lm-evaluation-harness`** — https://github.com/EleutherAI/lm-evaluation-harness. Study for the task registry plugin pattern, CLI ergonomics, and how tasks are versioned independently of the harness core.
- **Hugging Face `evaluate`** — https://github.com/huggingface/evaluate. Study for the metric plugin pattern and how community-contributed metrics are discovered.
- **SpeechBrain** — https://github.com/speechbrain/speechbrain. Study for audio loading conventions and sample-rate handling.
- **ESPnet and ESPnet-SDS** — https://github.com/espnet/espnet. Study for speech dialogue system evaluation conventions.
- **torchmetrics** — https://github.com/Lightning-AI/torchmetrics. Study for streaming-friendly metric accumulation patterns that work without holding entire prediction arrays in memory.

## Target models for v0.1 adapter

- **Kyutai Moshi** — https://github.com/kyutai-labs/moshi. Open-weights FD STS model. Primary target for the v0.1 `local-hf-model` adapter.
- **Sesame CSM** — https://github.com/SesameAILabs/csm. License review required before shipping an adapter. Target for v0.2.
- **NVIDIA PersonaPlex** — https://developer.nvidia.com/. Target for v0.2.

## Benchmarks in the adjacent space (for orientation only)

- **Full-Duplex-Bench v1 / v2 / v3** — arXiv identifiers 2503.04721, 2510.07838, and successors. Relevant prior work on full-duplex evaluation; noted here for context, not for copying.
- **HumDial** — ICASSP 2026 accepted benchmark for dialogue modeling.
- **Big Bench Audio** — commercial audio reasoning benchmark tracked by Artificial Analysis.

## Audio processing libraries

- **soundfile** — https://pysoundfile.readthedocs.io/. Primary choice for WAV load and save.
- **librosa** — https://librosa.org/. Resample only; do not use for broader audio processing.
- **torchaudio** — https://pytorch.org/audio/. Optional, lazy-imported.
- **numpy** — https://numpy.org/. Standard array backend.

## Python packaging references

- **pyproject.toml PEPs**: PEP 517, PEP 518, PEP 621, PEP 660. Use `pyproject.toml` with either setuptools or hatchling as the build backend.
- **ruff** — https://github.com/astral-sh/ruff. Primary linter and formatter. Configure in `pyproject.toml`.

## Related internal project documents

These are in the parent `oto-blog-project/` repo, not in this subdirectory.

- `30-drafts/07-benchmark-landscape/` — blog article on the STS benchmark landscape for broader context.
- `30-drafts/08-why-new-benchmarks/` — blog article on why new benchmarks are needed.
