"""FastAPI backend for the fd-eval-harness UI prototype.

This is a Claude-authored exploratory UI, separate from the Antigravity-owned
`src/fd_eval/` code path. It is not intended for git commit.

Run:
    pip install -r _ui_prototype/requirements.txt
    pip install -e '.[dev]'        # make entry_points visible
    uvicorn _ui_prototype.app:app --reload

Open http://localhost:8000
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="fd-eval-harness UI", version="0.0.1-ui")

_HERE = Path(__file__).parent
_STATIC_DIR = _HERE / "static"
_SAMPLES_DIR = _HERE / "samples"


# ---------- Harness integration (lazy) ----------

def _harness() -> dict[str, Any]:
    """Lazy import so the UI can boot even if fd-eval isn't installed yet."""
    from fd_eval.core.audio_session import AudioSession
    from fd_eval.core.registry import (
        get_adapter,
        get_task,
        list_adapters,
        list_tasks,
    )
    from fd_eval.data import load_audio, load_labels

    return {
        "AudioSession": AudioSession,
        "get_task": get_task,
        "get_adapter": get_adapter,
        "list_tasks": list_tasks,
        "list_adapters": list_adapters,
        "load_audio": load_audio,
        "load_labels": load_labels,
    }


def _task_meta(task_name: str) -> dict[str, Any]:
    h = _harness()
    cls = h["get_task"](task_name)
    return {
        "name": task_name,
        "version": getattr(cls, "version", "unknown"),
        "scoring_method": getattr(cls, "scoring_method", "unknown"),
        "mode": getattr(cls, "mode", "unknown"),
        "doc": (cls.__doc__ or "").strip().split("\n")[0] if cls.__doc__ else "",
    }


def _adapter_meta(adapter_name: str) -> dict[str, Any]:
    h = _harness()
    cls = h["get_adapter"](adapter_name)
    return {
        "name": adapter_name,
        "doc": (cls.__doc__ or "").strip().split("\n")[0] if cls.__doc__ else "",
    }


# ---------- Sample generation (startup) ----------
#
# Per-task sample pairs. Each entry describes what a canned demo looks like
# for a given task:
#   * audio: a tiny 2-channel wav (speech is simulated with sine tones)
#   * labels: a labels.json shaped for the task's ``parse_references``
#   * adapter: the adapter the UI should auto-pick with this sample
#   * adapter_args: matching default kwargs for that adapter
#   * in_channels / tgt_channels: matching channel config
# Tasks without a canned sample are listed in ``_SAMPLES_UNAVAILABLE`` with
# a human-readable reason, surfaced in the UI.

_SAMPLES_UNAVAILABLE: dict[str, str] = {
    "interruption_state_update": (
        "This task needs an adapter that emits TranscriptPredictionEvent. "
        "No such adapter ships with v0.2 (OpenAIRealtimeAdapter is still a stub). "
        "Provide your own adapter, or hand-craft a transcript prediction payload."
    ),
}


def _write_wav_two_channel_tone(
    audio_path: Path,
    *,
    sr: int = 24000,
    duration_s: float = 2.0,
    ch0_window: tuple[float, float] = (0.5, 1.5),
    ch0_hz: float = 220.0,
    ch1_window: tuple[float, float] = (1.2, 1.8),
    ch1_hz: float = 330.0,
) -> None:
    """Stamp a 2-channel WAV with tone bursts, used as a dummy speech proxy."""
    import numpy as np
    import soundfile as sf

    t = np.arange(0, int(duration_s * sr)) / sr
    ch0 = np.zeros_like(t, dtype="float32")
    m0 = (t >= ch0_window[0]) & (t < ch0_window[1])
    ch0[m0] = 0.3 * np.sin(2 * np.pi * ch0_hz * t[m0]).astype("float32")
    ch1 = np.zeros_like(t, dtype="float32")
    m1 = (t >= ch1_window[0]) & (t < ch1_window[1])
    ch1[m1] = 0.3 * np.sin(2 * np.pi * ch1_hz * t[m1]).astype("float32")
    audio = np.stack([ch0, ch1], axis=-1)
    sf.write(audio_path, audio, sr)


def _ensure_samples() -> None:
    """Generate canned per-task sample pairs if missing."""
    _SAMPLES_DIR.mkdir(exist_ok=True)

    # ----- voice_activity_detection (legacy default) -----
    vad_audio = _SAMPLES_DIR / "vad__audio.wav"
    vad_labels = _SAMPLES_DIR / "vad__labels.json"
    if not vad_audio.exists():
        _write_wav_two_channel_tone(vad_audio)
    if not vad_labels.exists():
        vad_labels.write_text(
            json.dumps(
                [
                    {"timestamp_s": 0.5, "channel": 0, "is_speech": True},
                    {"timestamp_s": 1.5, "channel": 0, "is_speech": False},
                    {"timestamp_s": 1.2, "channel": 1, "is_speech": True},
                    {"timestamp_s": 1.8, "channel": 1, "is_speech": False},
                ],
                indent=2,
            )
        )

    # Legacy filenames kept as aliases so older sessions that hard-coded
    # the VAD filenames continue to work.
    legacy_audio = _SAMPLES_DIR / "sample_audio.wav"
    legacy_labels = _SAMPLES_DIR / "sample_labels.json"
    if not legacy_audio.exists():
        legacy_audio.write_bytes(vad_audio.read_bytes())
    if not legacy_labels.exists():
        legacy_labels.write_bytes(vad_labels.read_bytes())

    # ----- tool_use_under_disfluency -----
    # Uses the ``tool_use_stub`` adapter's built-in default tool call
    # (no stub_file arg required) and a labels.json that subset-matches it.
    tool_audio = _SAMPLES_DIR / "tool_use__audio.wav"
    tool_labels = _SAMPLES_DIR / "tool_use__labels.json"
    if not tool_audio.exists():
        _write_wav_two_channel_tone(tool_audio, duration_s=1.0)
    if not tool_labels.exists():
        tool_labels.write_text(
            json.dumps(
                [{"tool_name": "weather", "arguments": {"location": "Tokyo"}}],
                indent=2,
            )
        )


# Registry of per-task sample bundles. Keys are task names.
_SAMPLE_BUNDLES: dict[str, dict[str, Any]] = {
    "voice_activity_detection": {
        "audio": "vad__audio.wav",
        "labels": "vad__labels.json",
        "adapter": "energy_vad",
        "adapter_args": {"threshold": 0.02},
        "in_channels": "0,1",
        "tgt_channels": "",
        "blurb": (
            "2-channel sine burst + matching VAD state-change labels. "
            "EnergyVAD emits onset/offset events; the task scores segment overlap."
        ),
    },
    "tool_use_under_disfluency": {
        "audio": "tool_use__audio.wav",
        "labels": "tool_use__labels.json",
        "adapter": "tool_use_stub",
        "adapter_args": {},
        "in_channels": "0",
        "tgt_channels": "1",
        "blurb": (
            "ToolUseStubAdapter's built-in default emits weather(location=Tokyo). "
            "The label file asks for the same call, so score should be 1.0."
        ),
    },
}


@app.on_event("startup")
def _on_startup() -> None:
    try:
        _ensure_samples()
    except Exception as e:  # pragma: no cover
        # Sample generation is non-fatal; UI can still work with user uploads.
        print(f"[fd-eval UI] Warning: sample generation failed: {e}")


# ---------- Routes ----------

@app.get("/")
def index():
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/api/meta")
def meta():
    """Return registry contents plus a harness-installed flag."""
    try:
        h = _harness()
        tasks = [_task_meta(name) for name in h["list_tasks"]()]
        adapters = [_adapter_meta(name) for name in h["list_adapters"]()]

        # Decorate with sample availability so the UI can disclaim mismatches.
        for t in tasks:
            bundle = _SAMPLE_BUNDLES.get(t["name"])
            if bundle is not None:
                t["sample"] = {
                    "available": True,
                    "adapter": bundle["adapter"],
                    "adapter_args": bundle["adapter_args"],
                    "in_channels": bundle["in_channels"],
                    "tgt_channels": bundle["tgt_channels"],
                    "blurb": bundle["blurb"],
                }
            else:
                t["sample"] = {
                    "available": False,
                    "reason": _SAMPLES_UNAVAILABLE.get(
                        t["name"],
                        "No canned sample ships for this task in the prototype UI.",
                    ),
                }

        return {
            "harness_installed": True,
            "tasks": tasks,
            "adapters": adapters,
        }
    except Exception as e:
        return {
            "harness_installed": False,
            "error": str(e),
            "hint": "From the project root, run `pip install -e '.[dev]'` so entry-points register.",
            "tasks": [],
            "adapters": [],
        }


# Files allowed to be served from the samples directory. Kept as a fixed
# allowlist so the endpoint cannot be abused for arbitrary file reads.
_ALLOWED_SAMPLE_FILES = {
    # legacy aliases
    "sample_audio.wav",
    "sample_labels.json",
    # VAD bundle
    "vad__audio.wav",
    "vad__labels.json",
    # tool_use bundle
    "tool_use__audio.wav",
    "tool_use__labels.json",
}


@app.get("/api/samples/{name}")
def sample(name: str):
    if name not in _ALLOWED_SAMPLE_FILES:
        raise HTTPException(404, "Unknown sample file.")
    path = _SAMPLES_DIR / name
    if not path.exists():
        raise HTTPException(404, "Sample not yet generated.")
    return FileResponse(path)


@app.post("/api/evaluate")
async def evaluate(
    task: str = Form(...),
    adapter: str = Form(...),
    adapter_args: str = Form("{}"),
    in_channels: str = Form("0"),
    tgt_channels: str = Form("1"),
    task_args: str = Form("{}"),
    audio: UploadFile = File(...),
    labels: UploadFile = File(...),
):
    # 1. Load harness
    try:
        h = _harness()
    except Exception as e:
        raise HTTPException(500, detail=f"fd-eval-harness is not importable: {e}")

    # 2. Resolve task & adapter
    try:
        task_cls = h["get_task"](task)
    except KeyError:
        raise HTTPException(400, detail=f"Task '{task}' not found in registry.")
    try:
        adapter_cls = h["get_adapter"](adapter)
    except KeyError:
        raise HTTPException(400, detail=f"Adapter '{adapter}' not found in registry.")

    # 3. Parse kwargs
    try:
        adapter_kwargs = json.loads(adapter_args) if adapter_args.strip() else {}
        task_kwargs = json.loads(task_args) if task_args.strip() else {}
    except json.JSONDecodeError as e:
        raise HTTPException(400, detail=f"Invalid JSON kwargs: {e}")

    # 4. Parse channels
    try:
        in_ch = [int(c.strip()) for c in in_channels.split(",") if c.strip()]
        tgt_ch = [int(c.strip()) for c in tgt_channels.split(",") if c.strip()]
    except ValueError as e:
        raise HTTPException(400, detail=f"Invalid channel spec: {e}")

    # 5. Stage uploads to tempfiles (loaders expect paths)
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        audio_p = td_p / (audio.filename or "audio.wav")
        labels_p = td_p / (labels.filename or "labels.json")
        audio_p.write_bytes(await audio.read())
        labels_p.write_bytes(await labels.read())

        try:
            audio_arr, sr = h["load_audio"](audio_p)
            raw_labels = h["load_labels"](labels_p)
        except Exception as e:
            raise HTTPException(400, detail=f"Input loading failed: {e}")

    # 6. Build AudioSession
    try:
        session = h["AudioSession"](
            audio=audio_arr,
            sample_rate=sr,
            input_channel_indices=in_ch,
            target_channel_indices=tgt_ch,
        )
    except Exception as e:
        raise HTTPException(400, detail=f"AudioSession construction failed: {e}")

    # 7. Run adapter → materialize predictions
    try:
        adapter_instance = adapter_cls(**adapter_kwargs)
    except Exception as e:
        raise HTTPException(400, detail=f"Adapter instantiation failed: {e}")
    try:
        predictions = list(adapter_instance.process(session))
    except Exception as e:
        raise HTTPException(500, detail=f"Adapter.process failed: {e}")

    # 8. Run task
    try:
        task_instance = task_cls(**task_kwargs)
    except Exception as e:
        raise HTTPException(400, detail=f"Task instantiation failed: {e}")
    try:
        references = task_instance.parse_references(raw_labels)
        result = task_instance.evaluate(session, predictions, references)
    except Exception as e:
        raise HTTPException(500, detail=f"Task.evaluate failed: {e}")

    # 9. Shape predictions/references for the UI (best-effort serialization)
    def _serializable(x: Any) -> Any:
        if hasattr(x, "__dict__"):
            return {k: _serializable(v) for k, v in vars(x).items()}
        if isinstance(x, (list, tuple)):
            return [_serializable(v) for v in x]
        if isinstance(x, dict):
            return {k: _serializable(v) for k, v in x.items()}
        if hasattr(x, "item"):  # numpy scalar
            try:
                return x.item()
            except Exception:
                return repr(x)
        return x

    return {
        "task": task,
        "task_version": getattr(task_cls, "version", "unknown"),
        "scoring_method": getattr(task_cls, "scoring_method", "unknown"),
        "mode": getattr(task_cls, "mode", "unknown"),
        "adapter": adapter,
        "audio": {
            "frames": int(audio_arr.shape[0]),
            "channels": int(audio_arr.shape[1]),
            "sample_rate": int(sr),
            "duration_s": float(audio_arr.shape[0]) / float(sr),
        },
        "in_channels": in_ch,
        "tgt_channels": tgt_ch,
        "predictions_count": len(predictions),
        "references_count": len(references),
        "predictions_preview": _serializable(predictions[:20]),
        "references_preview": _serializable(list(references)[:20]),
        "score": float(result.score),
        "details": dict(result.details),
    }


app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
