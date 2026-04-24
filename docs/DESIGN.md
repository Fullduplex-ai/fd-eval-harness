# fd-eval-harness — Design Document

## 1. Problem statement

Full-duplex speech-to-speech (FD STS) models are shipping at a steady cadence: Moshi (Kyutai), Sesame CSM, NVIDIA PersonaPlex, OpenAI GPT-4o Realtime, Google Gemini Live, Amazon Nova Sonic, and a growing list of proprietary systems. Researchers evaluating or comparing these models face three gaps:

1. **No standardized metric implementations.** Every paper reports its own turn-taking F1, with implementation details buried in appendices. Results are not cross-comparable.
2. **No shared model adapter layer.** Each research team writes custom plumbing to call Moshi locally, OpenAI via API, and a local checkpoint. Effort is duplicated.
3. **No shared result schema.** Public leaderboards cannot aggregate scores because result formats differ across teams.

`fd-eval-harness` closes these three gaps. It is not a new benchmark. It is a benchmark-agnostic executor. Benchmark task suites are authored as separate plugin packages, public or private, and loaded into the harness at runtime.

## 2. Design principles

1. **Benchmark-agnostic core.** The architecture supports any benchmark spec whose tasks can be expressed as `Task` plugin subclasses. The core harness ships only a small set of public-literature example tasks, intended to prove the end-to-end pipeline. Benchmark-specific task suites live in separate plugin packages.
2. **Streaming-ready interfaces, offline-OK implementations.** v0.1 processes audio files offline, but the public API shapes data as iterable streams so a streaming v0.2 implementation can slot in without breaking callers.
3. **Plugin-first.** Tasks, adapters, and metrics are discoverable plugins. Researchers add a new task by adding a file, not by editing the core.
4. **Reproducibility is load-bearing.** Every run records environment, git commit, seeds, model checksum, data checksum, and the full CLI invocation.
5. **Small public surface.** Fewer features done well over many features done partially.

## 3. Non-goals

- Not a training or fine-tuning framework.
- Not a dataset distribution channel. Audio and reference labels live in separate repos.
- Not a model hosting service. The harness calls models, does not host them.
- Not a web UI in v0.1. A leaderboard is a separate project that consumes this harness's JSON output.

## 4. Core abstractions

Four abstractions define the public contract. Base classes live in `fd_eval.core` and concrete implementations live in `fd_eval.tasks`, `fd_eval.adapters`, etc.

Two evaluation modes are supported (see D009):

- **Observer mode** — the model is evaluated on its own analysis of a pre-recorded two-channel conversation. Example: voice activity detection over both channels.
- **Participant mode** — the model listens to the input channel(s) and generates into the target channel(s). Example: turn-taking latency measured from input silence to target onset.

Channel roles are declared explicitly per-session via `input_channel_indices` and `target_channel_indices` on `AudioSession` (see D007). Observer tasks typically set both lists to the same two channels; participant tasks partition them.

### 4.1 Task

Represents one evaluation dimension. A task declares how its outputs will be scored so that leaderboards can guarantee like-for-like aggregation (see D008).

```python
from typing import Literal

ScoringMethod = Literal["algorithmic", "llm-judge", "human-mos", "hybrid", "other"]

class Task(ABC):
    scoring_method: ScoringMethod
    scoring_method_detail: str = ""  # required when scoring_method == "other"

    # Concrete tasks also declare:
    #   name: str
    #   version: str  (independent of harness version)
    #   mode: Literal["observer", "participant"]
```

`Task.__init_subclass__` validates at class-definition time that `scoring_method` is set to a value in the `ScoringMethod` literal, and that `scoring_method_detail` is non-empty when `scoring_method == "other"`. Task plugins that fail this check raise `ValueError` at import time, not at run time.

### 4.2 FDModelAdapter

Wraps a model behind a uniform interface. Adapters handle authentication, input format conversion, streaming boundaries, and output parsing.

The v0.1 signature is deliberately minimal (see D005):

```python
class FDModelAdapter(ABC):
    @abstractmethod
    def process(self, session: AudioSession) -> PredictionStream:
        """Process an AudioSession and yield prediction events."""
```

Adapter-specific configuration (voice variant, sampling temperature, API credentials) is passed to the adapter's `__init__`, not to `process`. This keeps the per-session call site uniform across adapters.

v0.1 target adapters:
- `local-hf-model`: loads a Hugging Face model checkpoint, runs inference locally. Primary target: Kyutai Moshi family. Ships both Moshiko and Moshika voice variants, selected via a required `voice: Literal["moshiko", "moshika"]` constructor argument (see D010).
- `openai-realtime-api`: Fully implemented in v0.2. Streams via WebSockets using server-side VAD (1x pacing).

### 4.3 DataLoader

Loads two-channel full-duplex audio and per-session reference labels, and constructs `AudioSession` instances with explicit channel-role assignments.

v0.1 input contract:
- Audio: 48kHz or 24kHz WAV, channel-separated, at least 2 channels. The `AudioSession` carries the full multi-channel array; `input_channel_indices` and `target_channel_indices` select which channels play which role for a given task.
- Labels: JSON file with time-intervals. Schema defined in `fd_eval.data.schemas`.
- Metadata: YAML per session, schema defined in `fd_eval.data.schemas`. Benchmark plugin packages may extend this schema with benchmark-specific fields.

An invariant enforced on `AudioSession` construction: input and target channel sets must be disjoint, and every declared index must be in range for the loaded audio (see D007 for rationale).

### 4.4 Scorer and Reporter

`Scorer` combines per-task results into a session-level summary. `Reporter` writes the standardized JSON output.

## 5. Data flow

```
AudioSession + SessionMetadata
       ↓
DataLoader  →  (audio_streams, reference_labels, metadata)
       ↓
ModelAdapter.process()  →  PredictionStream
       ↓
Task.evaluate(predictions, references, config)  →  TaskResult  (one per task)
       ↓
Scorer  →  SessionSummary  (aggregates TaskResults)
       ↓
Reporter  →  run.json + human-readable summary
```

## 6. Output format (JSON schema v0.1)

Abbreviated for clarity. Full JSON Schema file lives at `fd_eval/schemas/run_v0.1.json`.

```json
{
  "schema_version": "0.1",
  "run_id": "uuid-v4",
  "timestamp": "ISO-8601 UTC",
  "environment": {
    "python_version": "3.11.x",
    "package_versions": {"fd-eval-harness": "0.1.0", "torch": "2.x", "...": "..."},
    "git_commit": "short-sha",
    "git_dirty": false,
    "seed": 42,
    "hostname_hash": "sha256:..."
  },
  "model": {
    "adapter": "local-hf-model",
    "name": "kyutai/moshiko-pytorch-bf16",
    "license": "CC-BY-4.0",
    "checksum": "sha256:..."
  },
  "session": {
    "id": "sess_001",
    "duration_sec": 720.4,
    "sample_rate": 48000,
    "channels": 2,
    "conversation_type": "casual-spontaneous",
    "data_checksum": "sha256:..."
  },
  "tasks": {
    "voice_activity_detection": {
      "task_version": "1.0.0",
      "score": 0.74,
      "confidence_interval": [0.71, 0.77],
      "details": {"precision": 0.76, "recall": 0.72, "n_events": 182}
    },
    "speaker_change_detection": {"...": "..."}
  },
  "errors": []
}
```

## 7. Streaming vs batch

v0.1 and v0.2 process audio offline but expose streaming-shaped interfaces so a true remote streaming mode (v0.3) can land without breaking callers. Two pieces of the public contract define this shape.

**Audio chunk iterator on `AudioSession`** (see D006):

```python
def stream(self, chunk_ms: int = 20) -> Iterator[np.ndarray]:
    """Yields audio in chunks of chunk_ms. Default 20ms matches typical FD STS frame rate."""
```

Adapters that want the whole array at once use the `collect_all(session)` helper from `fd_eval.core.helpers`, which consumes the iterator and concatenates. Offline adapters may use this escape hatch freely; streaming adapters like OpenAI Realtime consume the iterator chunk-by-chunk without changing the caller's session construction.

**Prediction events** (see D005):

```python
@dataclass
class PredictionEvent:
    timestamp_s: float

PredictionStream = Iterator[PredictionEvent]
```

`PredictionStream` is a type alias, not a Protocol class. Concrete adapters yield `PredictionEvent` instances (or dataclass subclasses that extend `PredictionEvent` with task-specific fields such as `audio_chunk`, `label`, or `confidence`). Callers must treat the stream as forward-only and must not assume it is exhaustible, rewindable, or re-iterable.

An offline adapter materializes all predictions first, then yields them. A streaming adapter (like OpenAI Realtime) yields incrementally as audio arrives. Callers must treat both identically.

## 8. Two-channel audio handling

Channel separation is the default assumption throughout.

- Audio is loaded as `np.ndarray` with shape `(n_samples, 2)` or as a typed `StereoWaveform` dataclass with explicit `speaker_a` and `speaker_b` fields.
- Never reshape to mono without an explicit `mixdown()` call. A mixdown call must record its strategy (sum, average, left-only, etc.) in the run output.

## 9. Sample-rate handling

Only `fd_eval.audio.resample` may call `librosa.resample` or equivalent. All other modules assume the sample rate they are handed. Resampling operations are logged and recorded in the run output.

## 10. Reproducibility

Every `run.json` records:

- Python version and full package version manifest (via `importlib.metadata`).
- Git commit hash plus a dirty-tree flag.
- All random seeds used, across Python, NumPy, and (when applicable) PyTorch.
- Model weights checksum. For API-based adapters, the model name string returned by the API.
- Data file checksums (per audio file, per label file).
- Full command-line invocation, with secrets redacted.
- Hardware summary (CPU model, GPU model if present, RAM bucket).

## 11. Error handling

- Task failures do not abort the run. They are logged in the `errors` array of the run output, and the affected task's result is `null`.
- Adapter failures abort the run with a non-zero exit code. An adapter failure is not recoverable mid-run.
- Data loading errors abort the run.
- All error messages include the session ID, task name (if applicable), and a correlation ID.

## 12. Open design questions (resolve before v0.1 code starts)

1. **Audio I/O library.** Recommendation: `soundfile` for load and save, `librosa` for resample only, `torchaudio` optional and lazy-imported. Open: confirm `soundfile` covers all WAV variants in the test fixture data.
2. **Remote data loading.** Deferred to v0.3.
3. **Sample data in repo.** Only synthetic 5-second fixtures for tests. Real session data is distributed through benchmark plugin packages, not from this repo.
4. **Task versioning.** Each task has its own `version` string independent of harness version. A task can reach v2.0.0 while the harness is still v0.1. The harness output records both.
5. **Model license handling.** Some model licenses forbid benchmarking. Proposal: adapter authors declare the license string; the harness records it in output and prints a warning if the license is known-restrictive. The harness does not enforce license compliance; that is the researcher's responsibility.

## 13. Structural decisions deferred to v0.3 or later

- Live streaming ingestion from API endpoints.
- Leaderboard submission integration.
- Concurrent multi-session evaluation.
- Docker and container packaging.
- Per-language task variants (e.g., Japanese aizuchi specialization).
- Curated multi-benchmark bundles shipped from the harness core. Benchmark-specific task suites continue to live in separate plugin packages.

## 14. Reference implementations to study

These are structural references, not code to copy.

- EleutherAI `lm-evaluation-harness` for task registry and plugin pattern.
- SpeechBrain for audio loading conventions.
- ESPnet-SDS for speech dialogue system evaluation conventions.
- Hugging Face `evaluate` library for metric plugin pattern.
- `torchmetrics` for streaming-friendly metric accumulation patterns.

Study structure, write fresh code.

## 15. Decision records referenced by this document

Each load-bearing choice above has a corresponding ADR in the project's decision log. Consult the ADR for rationale, alternatives considered, and dissent.

- **D001** Python 3.11+
- **D002** Apache-2.0 license
- **D003** GitHub org `fullduplexai`
- **D004** Benchmark-agnostic core; benchmark suites as plugin packages
- **D005** `FDModelAdapter.process(session) -> PredictionStream`
- **D006** Hybrid streaming contract (20 ms chunk default, `collect_all` helper)
- **D007** `AudioSession` with explicit `input_channel_indices` and `target_channel_indices`
- **D008** `Task.scoring_method` required as a Literal, `"other"` escape hatch
- **D009** v0.1 ships both modes; participant task is Turn-Taking Latency
- **D010** Moshi adapter exposes both voice variants via a `voice` parameter

