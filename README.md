# fd-eval-harness

**Evaluation harness for full-duplex speech-to-speech (FD STS) models.**

A benchmark-agnostic executor. Specific benchmark task suites are distributed as separate plugin packages.

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

Pre-alpha. v0.1 targeted for 2026-Q3.

Licensed under Apache-2.0.

## Quick start

```bash
pip install fd-eval-harness

fd-eval run \
  --model local-hf-model:kyutai/moshiko-pytorch-bf16 \
  --tasks voice-activity-detection,speaker-change-detection,laughter-detection \
  --data path/to/session \
  --output results.json
```

See `--help` for the full CLI.

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
