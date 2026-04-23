# Cursor Agent Prompt: fd-eval-harness v0.1

This document is the initial briefing to hand to a Cursor or Claude Code agent for the v0.1 build of `fd-eval-harness`.

---

## How to use this document

1. Open the `fd-eval-harness/` directory in Cursor.
2. Paste the section below marked **PROMPT TO CURSOR** into a new Cursor chat.
3. Ensure the agent has read access to `docs/DESIGN.md`, `docs/ROADMAP.md`, `docs/TASKS.md`, and `REFERENCES.md` before it begins.
4. Review the agent's first planning response before approving execution. Do not let it start writing code before it has produced an implementation plan.
5. If the agent asks clarifying questions, answer them in writing and update the docs so future runs of the same prompt benefit.

---

## PROMPT TO CURSOR

You are building **fd-eval-harness**, a Python evaluation harness for full-duplex speech-to-speech (FD STS) models. This is the v0.1 build.

### Required reading before writing any code

Read the following files in order. Confirm in your first response that you have read each one.

1. `README.md` — public-facing description of the project.
2. `docs/DESIGN.md` — architecture and design decisions. This is the source of truth for all structural questions.
3. `docs/ROADMAP.md` — phased build plan. You are working on **v0.1 only**. Do not build v0.2 or v0.3 features.
4. `docs/TASKS.md` — task plugin authoring guide. v0.1 ships a small set of public-literature example tasks listed in `docs/ROADMAP.md`.
5. `REFERENCES.md` — external reading. Study EleutherAI's `lm-evaluation-harness` for the task registry pattern, but do NOT copy its code.

### Your objective

Ship a Python package that satisfies all acceptance criteria below.

### Coding standards

- Python 3.11 or newer.
- Type hints on every public function and method. Use `from __future__ import annotations` at the top of every module.
- `ruff` for linting and formatting. Configure in `pyproject.toml`.
- `pytest` for tests. Every public function has at least one test.
- `pyproject.toml` for packaging, with `setuptools` or `hatchling` as the build backend. No `setup.py`.
- Docstrings in Google style on all public APIs.
- No mutable default arguments.
- No module-level mutable state. Pass explicit context objects through function calls.
- Use `logging.getLogger(__name__)` for all logging. No `print` statements in library code.
- Use `pathlib.Path` for all file paths. No raw strings for paths.

### Architecture constraints (from DESIGN.md)

- **Plugin-based task registry.** Each task inherits from a `Task` base class and registers via a decorator. Tasks are discoverable via `importlib.metadata` entry points.
- **Plugin-based model adapter.** Each adapter inherits from a `ModelAdapter` base class.
- **Streaming-aware interfaces** even where v0.1 implementations are offline. APIs expose `PredictionStream` iterators; offline implementations yield all items eagerly. Do not bake batch-only assumptions into the public API.
- **Two-channel audio is the default.** Never reshape to mono without an explicit mixdown call.
- **Sample-rate conversions** go through exactly one module (`fd_eval.audio.resample`). No other module calls `librosa.resample` or equivalents directly.

### Acceptance criteria for v0.1

All must pass:

1. `pip install -e .` succeeds in a clean virtualenv on Python 3.11 and 3.12.
2. `fd-eval --help` prints usage text listing all top-level subcommands.
3. `fd-eval list-tasks` prints the v0.1 example tasks (see `docs/ROADMAP.md`) with name, category, version.
4. `fd-eval list-adapters` prints at least 1 model adapter.
5. `fd-eval run --model <adapter> --tasks <list> --data <path> --output <file>` executes end-to-end on the synthetic sample session included in `tests/fixtures/` and produces a JSON file matching the schema in `docs/DESIGN.md`.
6. `pytest` passes with at least 70 percent branch coverage.
7. The Quick Start command in `README.md` works end-to-end on a fresh clone (with synthetic fixture data).
8. `ruff check` and `ruff format --check` both pass.
9. No confidential or third-party-spec-specific content is checked into this repo. Benchmark-specific task taxonomies live in separate plugin packages.

### Out of scope for v0.1

Do not implement any of the following. These are planned for later versions.

- Streaming evaluation mode. Offline batch is sufficient.
- Web leaderboard submission.
- Audio preprocessing pipelines. Accept pre-preprocessed audio input.
- Training or fine-tuning of any kind.
- Task plugins beyond the small public-literature example set listed in `docs/ROADMAP.md`.
- Benchmark-specific task suites (these belong in separate plugin packages, not in the core harness).
- Multi-language task variants beyond what the sample data requires.
- Real OpenAI Realtime API integration. A stub returning fixed outputs is acceptable.
- Concurrent or distributed evaluation.
- Docker or container packaging.

### Anti-patterns to avoid

- Do not copy source code from `lm-evaluation-harness` or other libraries. Study the structure and write fresh code.
- Do not add shell scripts. All automation goes through Python entry points.
- Do not write a custom logging framework. Use the standard `logging` module.
- Do not add a Python dependency without justifying it in the commit message or PR description.
- Do not commit real audio data. Test fixtures are synthetic, short, and under 1 MB total.
- Do not create documentation files beyond what the roadmap specifies.
- Do not introduce abstraction layers that are not exercised by v0.1 code. YAGNI applies.

### Required deliverables (PR structure)

Organize your work into small commits. A reasonable PR sequence:

1. Scaffold: `pyproject.toml`, directory layout, empty `__init__.py` files, `ruff` and `pytest` configs.
2. Core abstractions: `Task`, `ModelAdapter`, `DataLoader`, `Scorer`, `Reporter` base classes and protocols.
3. Data loading: two-channel full-duplex audio loader with 48kHz / 24kHz WAV and JSON label support.
4. First task implementation: one of the public-literature example tasks listed in `docs/ROADMAP.md` (e.g., `voice_activity_detection`).
5. First model adapter: `local-hf-model`.
6. CLI: `fd-eval run`, `list-tasks`, `list-adapters`, `version`.
7. Remaining v0.1 example tasks from `docs/ROADMAP.md`, each behind its own entry point.
8. Stub adapter: `openai-realtime-api` (mocked responses, no network calls).
9. End-to-end integration test using synthetic fixtures.
10. README polish, CLI help text review, final coverage pass.

Each commit passes `ruff` and `pytest`.

### Process

1. **Read** all required documents listed above.
2. **Produce a one-page implementation plan** as your first response. Include:
   - Confirmation you have read each required document.
   - Proposed module layout (directory tree).
   - Order of PRs with rough scope of each.
   - List of open questions or ambiguities you need answered before writing code.
3. **Wait for approval** before writing any code.
4. **Work in small commits.** Each commit passes `ruff` and `pytest`.
5. **Open a PR for v0.1** when all acceptance criteria are met.

### Questions

If anything in the docs is ambiguous or contradictory, list the ambiguities as numbered questions in your planning response. Do not guess.

---

## Notes for the human reviewer (not for Cursor)

### Before pasting

- Confirm the repo has been initialized with `git init` and a suitable `.gitignore` for Python.
- Confirm a clean Python 3.11+ virtualenv is ready.
- Consider whether to add an `AGENTS.md` or `.cursorrules` file with coding conventions so the agent picks them up automatically.

### Reviewing the agent's first response

Look for:

- Explicit acknowledgment that each required doc has been read.
- A module layout that respects the "plugin-first, streaming-ready" constraints.
- Honest listing of ambiguities. If the agent produces no questions, be suspicious.
- No premature coding.

### Iterating this prompt

This prompt is v1. Keep a changelog at the bottom of this file as the prompt evolves across Cursor runs. Record which parts worked, which parts the agent misinterpreted, and which ambiguities surfaced.

---

## Changelog

- 2026-04-23 — v1 draft created. Not yet run against Cursor.
