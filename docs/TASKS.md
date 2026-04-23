# fd-eval-harness — Task Plugin Authoring Guide

`fd-eval-harness` is **benchmark-agnostic**. It does not ship with a fixed task taxonomy. Instead, each benchmark specification is authored as a task plugin package that the harness loads at runtime (see D004 in `_internal/DECISIONS.md`).

This document describes how to write task plugins. Specific benchmark implementations (which tasks to include, how they are labeled, which metric tolerances to use) live in separate plugin packages.

## What is a task plugin

A task plugin is a Python package that registers one or more `Task` subclasses with the harness. A plugin package contains:

- A `Task` subclass for each evaluation dimension it implements.
- A metric function per task (or a reference to a shared metric in `fd_eval.metrics`).
- Metadata: task name, version, evaluation mode, scoring method, reference dataset.
- Optional: plugin-level configuration and defaults.

Plugins are discovered at runtime via `importlib.metadata` entry points under the `fd_eval.tasks` group.

## Minimal task plugin skeleton

A minimal plugin looks like this:

```python
# my_fd_tasks/voice_activity.py
from fd_eval.core import Task

class VoiceActivityDetection(Task):
    name = "voice_activity_detection"
    version = "1.0.0"
    mode = "observer"
    scoring_method = "algorithmic"

    def evaluate(self, session, predictions, references):
        # Compute event detection F1 with the tolerance declared by the task.
        ...
```

And in the plugin package's `pyproject.toml`:

```toml
[project.entry-points."fd_eval.tasks"]
voice_activity_detection = "my_fd_tasks.voice_activity:VoiceActivityDetection"
```

## Task interface contract

A task plugin must:

1. Inherit from `fd_eval.core.Task` (an `ABC`).
2. Declare a `scoring_method` class attribute. It must be one of the values in the `ScoringMethod` literal (see D008 in `_internal/DECISIONS.md`):
   - `"algorithmic"` — deterministic programmatic scoring (WER, F1, latency in seconds).
   - `"llm-judge"` — an LLM scores the output against a rubric.
   - `"human-mos"` — human raters produce MOS-style ratings.
   - `"hybrid"` — two or more of the above combined by a declared rule.
   - `"other"` — requires also setting `scoring_method_detail` to a non-empty string.
3. Declare `name`, `version`, and `mode` (one of `"observer"` or `"participant"`, see D009).
4. Implement an `evaluate` method whose exact signature is under review in v0.1; see "Evaluate signature" below.
5. Be deterministic given the same inputs and configuration.
6. Not hold mutable module-level state.

`Task.__init_subclass__` validates the `scoring_method` and `scoring_method_detail` declarations at class-definition time. A plugin whose class is malformed fails at import, not at run.

## Evaluation modes

Two evaluation modes are supported (D009):

- **Observer mode** — the model analyzes a pre-recorded two-channel session. The adapter's `process(session)` returns a `PredictionStream`; the task scores that stream against per-session references.
- **Participant mode** — the model listens to input channels and generates audio on target channels. Channel roles are declared on `AudioSession.input_channel_indices` / `target_channel_indices` (D007). The task scores properties of the generated output (for example, turn-taking latency) rather than its content.

A single task is one mode, not both. The mode is fixed at the class level and recorded in `run.json`.

## Evaluate signature

The v0.1 core defines `Task` with validated declarative attributes but does not yet finalize the `evaluate` method signature. Early task plugins should treat `evaluate(session, predictions, references)` as the working shape, passing the `AudioSession` (for adapter-independent context such as sample rate and channel layout), the adapter's `PredictionStream`, and a reference structure defined by the task plugin itself.

The exact signature will be finalized in a later ADR once at least two tasks with different scoring regimes (algorithmic observer and algorithmic participant) are implemented end-to-end.

## Metric implementations

The harness ships reference implementations of common audio evaluation metrics in `fd_eval.metrics` (planned; not all present in the v0.1 skeleton):

- `event_detection_f1(predictions, references, tolerance_sec)` — time-tolerant event F1.
- `multiclass_classification_accuracy(predictions, references)` — classification accuracy.
- `boundary_detection_f1(predictions, references, tolerance_sec)` — segment boundary F1.
- `bootstrap_confidence_interval(scores, n_resamples=1000, confidence=0.95)` — confidence interval for any score list.

Task plugins are encouraged to use these where possible. When a plugin needs a custom metric, it declares the metric inline and documents the algorithm in the plugin's docstring.

## Tolerance handling

Temporal tolerances for event-detection-style tasks are **configurable per run**. A task plugin declares a default tolerance; the harness records the actual tolerance used in `run.json`. This makes cross-run comparability explicit.

## Versioning

Task versions are independent of the harness version and independent of other tasks. A task can be at v2.0.0 while the harness is at v0.1 and another task in the same plugin is at v1.3.2. The harness records all three in the output.

Breaking changes to a task's scoring (different metric, different tolerance semantics, different output format) require a major version bump on the task.

## v0.1 reference tasks

The v0.1 reference distribution includes **two** public-literature task plugins for demonstration, one per mode. They are **not** a benchmark. They exist to prove the harness runs end-to-end in both modes (D009) and to serve as authoring examples. The scope was narrowed from an earlier five-task plan; the reasoning and the list of deferred tasks is at the end of this section.

### 1. `voice_activity_detection`
- **Mode**: observer.
- **Scoring method**: `algorithmic` (event-detection F1 with configurable time tolerance).
- **Reference dataset**: any two-channel conversation with per-channel speech/non-speech intervals. Public options: AMI (interval-aligned headset mix), CHiME-6 (with CHiME-6 style per-speaker VAD labels).
- **Scope**: per-channel binary speech vs. non-speech detection. No classification of speech type.
- **Reference adapter**: `energy_vad` (RMS-threshold-based observer adapter).

### 2. `turn_taking_latency`
- **Mode**: participant (the single v0.1 participant task per D009).
- **Scoring method**: `algorithmic` (timing-only, content-blind).
- **Reference dataset**: any two-channel conversational corpus where the input-channel side can be played to the model and the target-channel onset can be measured. Public options: AMI headset mix, Switchboard two-channel recordings.
- **Scope**: measure the time from end-of-speech on the input channel to first-audible-generation on the target channel. Does not score what the model said, only when it started saying it. Endpoint detection methodology and silence threshold are declared inside the plugin and recorded in `run.json`.
- **Reference adapter**: `moshi` with `emit_as="turn_taking"` (Participant-Only per D012).

These two tasks together exercise both modes and the two most common event-matching shapes (symmetric tolerance F1 for observer VAD; asymmetric forward-walk pairing for participant latency). The `scoring_method` declaration is uniform across v0.1 (`"algorithmic"`) because v0.1 scope does not include LLM-judge or human-MOS tasks.

### Shipped in v0.2 (Business-Value Priorities)

Based on D014 (Positioning as a Shared Execution Layer), the harness prioritizes plugins that address critical failure modes in real-world conversational agents. The following participant-mode tasks were completed in v0.2:

#### 3. `interruption_state_update`
- **Mode**: participant.
- **Scoring method**: `llm-judge` (Evaluating the semantic correctness of the model's final response after the interruption).
- **Goal**: Measure whether the model properly abandons an old instruction and follows a new one when the user interrupts it mid-turn.
- **Business Value**: Demonstrates robustness against mid-turn context shifts, addressing a common and highly visible failure mode in full-duplex agents.

#### 4. `tool_use_under_disfluency`
- **Mode**: participant.
- **Scoring method**: `algorithmic` (Exact or subset JSON matching of emitted tool-call payloads).
- **Goal**: Measure if the model correctly emits a tool-call payload despite user hesitations, repetitions, or self-corrections mid-sentence.
- **Business Value**: Connects voice evaluation directly to enterprise API execution reliability.

#### Out of scope for fd-eval-harness core (welcomed as external plugins)

Three further observer tasks were in the earlier v0.1 plan:

- `speaker_change_detection` — boundary-detection F1 on AMI / DIHARD III. Could produce a partial baseline with an energy-differential adapter on 2-channel audio, but requires its own reference adapter beyond `energy_vad`. The harness core will not carry bundled references for this task; it can be implemented by external benchmark plugins as needed.
- `laughter_detection` — event-detection F1 on Switchboard / ICSI laughter subsets. Energy thresholding cannot distinguish speech from laughter; the resulting scores in v0.1 would be degenerate (empty stream or RMS-peak false positives). The harness core will not carry bundled references for this task; it can be implemented by external benchmark plugins as needed.
- `disfluency_detection` — event-detection F1 on Switchboard NXT / CallHome disfluency subsets. Same structural constraint as laughter: requires VUV or filled-pause-specific features. The harness core will not carry bundled references for this task; it can be implemented by external benchmark plugins as needed.

## Writing a plugin for a specific benchmark

Benchmark specifications (for example, a particular lab's evaluation taxonomy) should be authored as **separate plugin packages**, either public or private. This keeps the harness core neutral and lets benchmark owners retain control over their taxonomy, naming, and licensing.

A benchmark plugin package typically contains:

- One `Task` subclass per evaluation dimension in the benchmark.
- A top-level configuration specifying which tasks are included at which version.
- Documentation of the benchmark's data format requirements.
- Optionally, a reference dataset download stub.

Benchmark plugins can be:

- **Public**: distributed on PyPI under the benchmark owner's namespace.
- **Private**: distributed internally, installed from a private index or git URL.

The harness treats public and private plugins identically.

## Testing a task plugin

Every task plugin should include:

- Unit tests for the metric function on synthetic inputs with known expected outputs.
- A round-trip integration test: construct a synthetic `AudioSession`, run the task, verify the output schema.
- Edge case tests: empty predictions, empty references, identical predictions and references, predictions shifted by more than tolerance.

## Contributing a task plugin

Task plugins contributed to the core harness distribution follow the general contribution guidelines in the repository root. External plugins are self-governed by their authors.
