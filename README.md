# fd-eval-harness

**A shared execution layer for full duplex voice evaluation.**

`fd-eval-harness` is not another benchmark. It is a benchmark-agnostic executor that helps voice AI teams measure real-time conversation quality in a reproducible way. Instead of building a custom evaluation stack for each model, teams can run shared tests for interruption handling, turn-taking, and tool use, letting many benchmarks test many models in one consistent way.

## Development Install

Requires Python 3.11+.

```bash
git clone https://github.com/fullduplexai/fd-eval-harness.git
cd fd-eval-harness
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

> [!WARNING]
> You must run `pip install -e .` (or `.[dev]`) to test the harness. Running `pytest` purely via `PYTHONPATH=.` will cause tests relying on plugin registry (`entry_points`) to fail, as the entry points are only registered upon package installation.

## Usage

The harness provides a standardized evaluation pipeline for full-duplex models. It supports both **Observer Mode** (listening to a pre-recorded conversation) and **Participant Mode** (interacting as one of the speakers).

Think of this as the streaming-audio sibling of EleutherAI's [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). Just as `lm-eval-harness` became the de facto executor for language model benchmarks, `fd-eval-harness` aims to become the shared executor for full-duplex voice model benchmarks.

## What this is

A Python library and CLI that:

- Loads two-channel full-duplex audio data (48kHz or 24kHz, channel-separated, 2-channel) and per-session reference labels.
- Wraps FD STS models (local checkpoints, API-based systems) behind a uniform adapter interface.
- Loads task plugins at runtime and scores model outputs against reference labels.
- Emits standardized JSON results with full environment, data, and model provenance.

## What this is NOT

- **Not the benchmark dataset itself.** Audio data and reference labels are distributed separately.
- **Not a training or fine-tuning framework.**
- **Not a model zoo.** Model weights are not hosted here.
- **Not a web leaderboard.** The leaderboard is a separate project; this harness produces the scores it consumes.

## Status

Pre-alpha. v0.3 targeted for 2026-Q3.

Licensed under Apache-2.0.

## Quick start

```bash
pip install fd-eval-harness

fd-eval \
  --tasks voice_activity_detection \
  --adapter energy_vad \
  --in-channels 0,1 \
  --tgt-channels "" \
  --audio-path path/to/session.wav \
  --labels-path path/to/labels.json \
  --output results.json
```

For programmatic execution in Python, see [`examples/run_evaluation_programmatically.py`](examples/run_evaluation_programmatically.py).

## Documentation

- [`CURSOR_PROMPT.md`](CURSOR_PROMPT.md) — handoff brief for a Cursor/Claude agent building v0.1.
- [`docs/DESIGN.md`](docs/DESIGN.md) — architecture and design decisions.
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — versioned scope plan (v0.1 → v1.0).
- [`docs/TASKS.md`](docs/TASKS.md) — task plugin authoring guide.
- [`REFERENCES.md`](REFERENCES.md) — external reading.

## Citation

If you use fd-eval-harness in published research, please cite:

```
@software{fdevalharness2026,
  title   = {fd-eval-harness: Evaluation harness for full-duplex speech-to-speech models},
  author  = {Fullduplex.ai},
  year    = {2026},
  url     = {https://github.com/fullduplexai/fd-eval-harness}
}
```

## License

Apache-2.0. See `LICENSE`.
