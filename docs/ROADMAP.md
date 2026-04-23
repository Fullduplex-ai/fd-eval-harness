# fd-eval-harness — Roadmap

Versioned scope plan. Each version has an explicit scope and explicit out-of-scope list.

## Core v0.1 Infrastructure (Slices 1-7): Complete as of 2026-04-24

Scope: a minimum viable harness that runs a small set of demonstration tasks against at least 1 model adapter on 1 sample session, end-to-end, locally.

- **Completed Slices**:
  1. Project setup and core interfaces
  2. Syntactic validation for configuration
  3. Base `Task` and `ModelAdapter` interfaces
  4. Dummy Model Adapter (`moshi` stub)
  5. Metrics and `TaskResult` format
  6. Plugin Registry and CLI
  7. DataLoader and Evaluation Loop

- **Deliverables achieved**:
  - Apache-2.0 license.
  - Python package structure ready.
  - Core interfaces (`AudioSession`, `PredictionStream`, `TaskResult`).
  - Dynamic plugin registry (`entry_points`).
  - CLI: `fd-eval` for offline batch execution.
  - Synthetic test fixtures and programmatic execution examples.
  - Full GitHub Actions CI with 70%+ branch coverage.

## v0.1.1 Small Follow-ups

- Slice 7.1: `soundfile` lazy imports, `Task.version` interface enforcement, and documentation for development installs.

## Core v0.2 (Slice A, B, Realtime Adapter): Complete as of 2026-04-24

Focus shifts to the "Shared Execution Layer" positioning, prioritizing high-business-value tasks over generic observer measurements.

- **Completed Slices**:
  - **Slice A**: `tool_use_under_disfluency` (algorithmic scoring)
  - **Slice B**: `interruption_state_update` (llm-judge scoring)
  - **Realtime API**: `OpenAIRealtimeAdapter` implementation

- **ADRs Ratified**:
  - D013 (Task interface extensions)
  - D014 (Shared Execution Layer)
  - D015 (LLM-as-judge scoring protocol)
  - D016 (Real-Time Pacing & Server-Side VAD for Realtime API)

## Stretch (v0.3+)

- Real-model smoke tests against actual Moshi checkpoints.
- Remote data loading (Hugging Face datasets).
- Streaming evaluation mode (`--streaming`).
- Multilingual task variants (e.g. Japanese backchannel taxonomy).

## Explicitly out of scope for core

The following legacy observer tasks have been removed from the core harness scope and are welcomed as **external plugins**:
- `speaker_change_detection`
- `laughter_detection`
- `disfluency_detection`

The core harness will not carry bundled references or reference adapters for these. They should be implemented by benchmark plugins that actually require them.

## Versioning policy

- **v0.x** — pre-stable, breaking changes allowed between minor versions. Called out explicitly in release notes.
- **v1.x** — stable, strict semver.
- **Task versions are independent of harness versions.** A single task can reach v2.0.0 while the harness is still at v1.0. The harness output records both versions.
- **Adapter versions are independent** of harness and task versions for the same reason.
