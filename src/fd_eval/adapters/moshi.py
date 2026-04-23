"""Kyutai Moshi reference adapter.

Participant-Mode-only per D012: the underlying `moshi` Python API
(`mimi.encode`) accepts a single user-audio stream and autoregressively
generates the model's own response. It cannot passively observe a
pre-recorded two-channel session.

Per D011 and the 21:00 JST GO from the project owner, this adapter
supports an `emit_as` parameter so a single generation pass can feed
either raw Moshi output (for bespoke task plugins), the turn-taking
latency task, or a VAD task. Observer-mode reference scoring is served
by the separate `EnergyVADAdapter`; `emit_as="vad"` here is a
convenience for comparing Moshi's own speaking activity against VAD
references, not a substitute for the observer pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from fd_eval.core import AudioSession, FDModelAdapter, PredictionEvent, PredictionStream
from fd_eval.tasks._types import TurnTakingPredictionEvent, VADPredictionEvent

MoshiEmitFormat = Literal["raw", "vad", "turn_taking"]


@dataclass
class MoshiPredictionEvent(PredictionEvent):
    """Raw Moshi per-frame output: predicted text token plus audio codes.

    Emitted when ``MoshiAdapter(emit_as="raw")``. Task plugins that want
    to interpret raw Moshi tokens consume this event type directly.
    """

    text_token: int | None = None
    audio_codes: np.ndarray | None = None


class MoshiAdapter(FDModelAdapter):
    """Reference adapter for Kyutai Moshi.

    Parameters
    ----------
    voice:
        Which Kyutai checkpoint to load. Required per D010; no default
        meaning is assigned to either variant. ``moshiko`` is
        male-voiced, ``moshika`` is female-voiced.
    emit_as:
        What prediction-event shape to yield. ``"raw"`` (default) yields
        :class:`MoshiPredictionEvent` carrying the per-frame text token
        and audio codes. ``"turn_taking"`` decodes the generated audio
        and yields :class:`TurnTakingPredictionEvent` onsets/offsets
        whenever Moshi's own output crosses the energy threshold.
        ``"vad"`` yields :class:`VADPredictionEvent` with the same
        underlying state-change logic.
    vad_energy_threshold:
        RMS threshold on the decoded audio chunk that separates Moshi
        "speaking" from "silent". Only used for
        ``emit_as in {"vad", "turn_taking"}``.
    """

    def __init__(
        self,
        voice: Literal["moshiko", "moshika"] = "moshika",
        emit_as: MoshiEmitFormat = "raw",
        vad_energy_threshold: float = 0.01,
    ):
        self.voice = voice
        self.emit_as = emit_as
        self.vad_energy_threshold = vad_energy_threshold
        self.mimi = None
        self.moshi = None
        self.lm_gen = None

    def _lazy_load(self):
        if self.mimi is not None:
            return

        import torch
        from huggingface_hub import hf_hub_download
        from moshi.models import LMGen, loaders

        repo = f"kyutai/{self.voice}-pytorch-bf16"

        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)

        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        self.mimi = loaders.get_mimi(mimi_weight, device=device)
        self.mimi.set_num_codebooks(8)

        moshi_weight = hf_hub_download(repo, loaders.MOSHI_NAME)
        self.moshi = loaders.get_moshi_lm(moshi_weight, device=device)
        self.lm_gen = LMGen(self.moshi, temp=0.8, temp_text=0.7)
        self.device = device

    def process(self, session: AudioSession) -> PredictionStream:
        """
        Process the session in Participant Mode (Moshi cannot do pure Observer).
        The first specified input channel is treated as the user's voice.
        Moshi autoregressively generates its own voice (text tokens and audio codes).
        """
        self._lazy_load()
        import torch

        if not session.input_channel_indices:
            raise ValueError("MoshiAdapter requires at least one input channel.")

        in_ch = session.input_channel_indices[0]

        if self.emit_as in ("vad", "turn_taking") and not session.target_channel_indices:
            raise ValueError(
                f"MoshiAdapter emit_as={self.emit_as!r} requires a target channel "
                "on the AudioSession so onset/offset events can be attributed."
            )
        out_ch = session.target_channel_indices[0] if session.target_channel_indices else in_ch

        # 80ms chunk = 1920 samples at 24kHz
        frame_size = self.mimi.frame_size
        chunk_ms = int((frame_size / session.sample_rate) * 1000)

        prev_is_speech: bool | None = None

        with torch.no_grad(), self.lm_gen.streaming(1), self.mimi.streaming(1):
            for i, chunk in enumerate(session.stream(chunk_ms=chunk_ms)):
                if chunk.shape[0] < frame_size:
                    pad_len = frame_size - chunk.shape[0]
                    chunk = np.pad(chunk, ((0, pad_len), (0, 0)), mode="constant")

                user_audio = chunk[:, in_ch]  # shape: (T,)
                user_audio_tensor = (
                    torch.from_numpy(user_audio).float().unsqueeze(0).unsqueeze(0).to(self.device)
                )

                codes = self.mimi.encode(user_audio_tensor)
                tokens_out = self.lm_gen.step(codes)

                timestamp_s = (i * frame_size) / session.sample_rate

                text_token: int | None = None
                audio_codes: np.ndarray | None = None
                audio_codes_tensor = None

                if tokens_out is not None:
                    # tokens_out is [B, 1 + 8, 1]
                    text_token = tokens_out[0, 1, 0].item()
                    audio_codes_tensor = tokens_out[:, 1:, :]
                    audio_codes = tokens_out[0, 1:, 0].cpu().numpy()

                if self.emit_as == "raw":
                    yield MoshiPredictionEvent(
                        timestamp_s=timestamp_s,
                        text_token=text_token,
                        audio_codes=audio_codes,
                    )
                    continue

                # emit_as in {"vad", "turn_taking"} — decode, threshold, emit on state change.
                if audio_codes_tensor is None:
                    continue

                decoded = self.mimi.decode(audio_codes_tensor)
                decoded_np = np.asarray(decoded.detach().cpu().numpy()).reshape(-1)
                rms = float(np.sqrt(np.mean(decoded_np**2))) if decoded_np.size else 0.0
                is_speech = rms > self.vad_energy_threshold

                if prev_is_speech is None or is_speech != prev_is_speech:
                    if self.emit_as == "vad":
                        yield VADPredictionEvent(
                            timestamp_s=timestamp_s,
                            channel=out_ch,
                            is_speech=is_speech,
                        )
                    else:  # turn_taking
                        yield TurnTakingPredictionEvent(
                            timestamp_s=timestamp_s,
                            channel=out_ch,
                            event_kind="onset" if is_speech else "offset",
                        )
                    prev_is_speech = is_speech
