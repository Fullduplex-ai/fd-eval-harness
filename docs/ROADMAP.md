# fd-eval-harness — Roadmap

Versioned scope plan. Each version has an explicit scope and explicit out-of-scope list.

## v0.1 — first runnable harness (target: 2026-Q3, 6-week build)

Scope: a minimum viable harness that runs a small set of demonstration tasks against at least 1 model adapter on 1 sample session, end-to-end, locally.

### v0.1 example task plugins (public-literature only)

The v0.1 reference distribution ships a small set of public-literature task plugins, intended purely to prove the harness runs end-to-end. These are authoring examples, not a benchmark.

- `voice_activity_detection`
- `speaker_change_detection`
- `laughter_detection`
- `disfluency_detection`

Benchmark-specific task suites (such as lab-authored evaluation taxonomies) are authored as separate plugin packages and are not part of the core harness distribution.

### v0.1 model adapters

- `local-hf-model` — real implementation, target: Kyutai Moshi open weights.
- `openai-realtime-api` — stub. Returns fixed mock outputs. Real API integration deferred to v0.2.

### v0.1 deliverables

- Python package published to TestPyPI under `fd-eval-harness`.
- Apache-2.0 license.
- CLI: `fd-eval run`, `fd-eval list-tasks`, `fd-eval list-adapters`, `fd-eval version`.
- JSON output schema v0.1 with JSON Schema file.
- Full documentation set in place (README, DESIGN, ROADMAP, TASKS, REFERENCES).
- `pytest` suite with 70 percent branch coverage minimum.
- Synthetic test fixtures (under 1 MB total).
- GitHub Actions CI: lint, format-check, test, matrix over Python 3.11 and 3.12.

### Explicitly out of scope for v0.1

- Streaming evaluation. Offline batch only.
- Leaderboard submission.
- Real API calls to OpenAI, Gemini, or Anthropic.
- Additional task plugins beyond the small public-literature example set.
- Multi-language support beyond fixture data.
- Remote data loading (Hugging Face datasets, S3, etc.).
- Docker or container packaging.
- Distributed or concurrent evaluation.

## v0.2 — full task coverage plus real APIs (target: 2026-Q4, 10-week build)

Scope expansion:

- Expanded example task plugin set (additional public-literature tasks, numbers and titles to be decided during v0.2 planning).
- Streaming evaluation mode behind `--streaming` flag.
- Real OpenAI Realtime API adapter.
- Real Google Gemini Live adapter.
- Sesame CSM adapter (subject to license review).
- Hugging Face datasets integration for remote data loading.
- Leaderboard submission CLI: `fd-eval submit`.

### v0.2 deliverables

- Published on PyPI proper (dropping the TestPyPI-only status).
- Integration tests against all three real APIs, gated behind secrets in CI.
- JSON schema v0.2 (backward-compatible with v0.1 readers, additive fields only).
- At least one published FD benchmarking paper uses the harness.

### Out of scope for v0.2

- Curated multi-benchmark bundles shipped from the harness core (benchmark task suites continue to live in separate plugin packages).
- Per-language task variants.
- Plugin registry UI.
- Reproducibility helper CLI.

## v0.3 — multilingual plus reproducibility (target: 2027-Q1, 12-week build)

Scope expansion:

- Multi-language variants. First addition: Japanese addendum for backchannel taxonomy (Clancy 1996 / Stivers 2009 mapping).
- Concurrent multi-session evaluation.
- Docker image on Docker Hub.
- `fd-eval reproduce <run.json>` CLI: re-runs from an environment manifest, producing a new `run.json` and comparing.
- Plugin registry for community-contributed tasks and adapters.

## v1.0 — stability (target: 2027-Q3)

Stability guarantee:

- JSON output schema frozen.
- CLI frozen.
- Core task implementations frozen at v1.0.0 each.
- Semver from here forward.
- Breaking changes require v2.0.0.

## Versioning policy

- **v0.x** — pre-stable, breaking changes allowed between minor versions. Called out explicitly in release notes.
- **v1.x** — stable, strict semver.
- **Task versions are independent of harness versions.** A single task can reach v2.0.0 while the harness is still at v1.0. The harness output records both versions.
- **Adapter versions are independent** of harness and task versions for the same reason.

## Release cadence

- v0.x: every 6 to 12 weeks, aligned to deliverable milestones.
- v1.x: quarterly, with LTS every 12 months.
