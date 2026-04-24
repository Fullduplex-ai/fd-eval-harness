# fd-eval-harness — UI prototype

Exploratory web console for trying `fd-eval-harness` without the CLI.
Authored by Claude. Experimental, not part of the public API.
Separate from the Antigravity-owned `src/fd_eval/` code path.

## What it does

- Lists registered tasks and adapters from `importlib.metadata.entry_points`.
- Lets you pick a task + adapter, set kwargs, upload an audio + labels pair.
- Runs the exact same pipeline `fd_eval.cli` runs, and shows the score, details, predictions and raw `run.json` output.
- Ships a tiny 2-channel sample pair so you can hit RUN within seconds of starting the server.

## Run it

From the project root (`otoEvalHarness/`):

```bash
# 1. Make the harness importable (registers entry points)
pip install -e '.[dev]'

# 2. Install the UI's own deps
pip install -r _ui_prototype/requirements.txt

# 3. Start the server
uvicorn _ui_prototype.app:app --reload --port 8000

# 4. Open http://localhost:8000
```

## Flow

1. Wait for the status pill in the top right to go teal (`1 task · 1 adapter` or similar).
2. Click **Load sample pair** in the Inputs card. It fills the file pickers with a pre-generated 2-second 2-channel WAV plus a matching VAD label JSON, and auto-selects `voice_activity_detection` + `energy_vad`.
3. Hit **RUN EVALUATION**. Results render on the right with a circular score dial, details breakdown, predictions / references previews, and the raw `run.json`.

## Files

```
_ui_prototype/
├── app.py                        ← FastAPI backend
├── requirements.txt              ← fastapi, uvicorn, python-multipart
├── README.md                     ← this file
├── static/
│   └── index.html                ← single-file UI (Tailwind CDN + vanilla JS)
└── samples/                      ← generated on first run
    ├── sample_audio.wav
    └── sample_labels.json
```

## Endpoints

| Method | Path                         | Purpose                                  |
|--------|------------------------------|------------------------------------------|
| GET    | `/`                          | Serves the UI.                           |
| GET    | `/api/meta`                  | Registry dump + `harness_installed` flag.|
| GET    | `/api/samples/{name}`        | Download the generated sample files.     |
| POST   | `/api/evaluate`              | Run a task × adapter on uploaded files.  |

`/api/evaluate` accepts `multipart/form-data` with: `task`, `adapter`, `adapter_args` (JSON str), `task_args` (JSON str), `in_channels` (comma-separated), `tgt_channels` (comma-separated), `audio` (file), `labels` (file).

## Notes

- The harness is imported lazily, so the UI still boots when `pip install -e` hasn't been run. In that case the top-right pill turns red and the right pane shows a hint pointing to the install step.
- Sample generation is also lazy; if it fails the UI still works with user uploads.
- `_ui_prototype/` is under `_internal/`-style gitignored territory via the existing `.gitignore` pattern (the folder is new; if it starts showing in `git status`, add `_ui_prototype/` to `.gitignore`).
