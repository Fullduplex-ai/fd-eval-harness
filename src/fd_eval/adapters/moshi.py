from dataclasses import dataclass
from typing import Literal

import numpy as np

from fd_eval.core import AudioSession, FDModelAdapter, PredictionEvent, PredictionStream
from fd_eval.tasks._types import TurnTakingPredictionEvent, VADPredictionEvent


@dataclass
class MoshiPredictionEvent(PredictionEvent):
    text_token: int | None = None
    audio_codes: np.ndarray | None = None


class MoshiAdapter(FDModelAdapter):
    def __init__(
        self,
        voice: Literal["moshiko", "moshika"] = "moshika",
        emit_as: Literal["raw", "turn_taking", "vad"] = "raw",
    ):
        self.voice = voice
        self.emit_as = emit_as
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
        target_ch = session.target_channel_indices[0] if session.target_channel_indices else 0
        _was_speech = False

        # 80ms chunk = 1920 samples at 24kHz
        frame_size = self.mimi.frame_size
        chunk_ms = int((frame_size / session.sample_rate) * 1000)

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

                text_token = None
                audio_codes = None
                is_speech = False

                if tokens_out is not None:
                    # tokens_out is [B, 1 + 8, 1]
                    text_token = tokens_out[0, 1, 0].item()
                    audio_codes = tokens_out[0, 1:, 0].cpu().numpy()

                    if self.emit_as != "raw":
                        codes_tensor = (
                            torch.from_numpy(audio_codes.astype(np.int64))
                            .unsqueeze(0)
                            .unsqueeze(-1)
                            .to(self.device)
                        )
                        wav_chunk = self.mimi.decode(codes_tensor)
                        rms = torch.sqrt(torch.mean(wav_chunk**2)).item()
                        is_speech = rms > 0.01

                if self.emit_as != "raw":
                    if is_speech and not _was_speech:
                        _was_speech = True
                        if self.emit_as == "turn_taking":
                            yield TurnTakingPredictionEvent(
                                timestamp_s=timestamp_s, channel=target_ch, event_kind="onset"
                            )
                        elif self.emit_as == "vad":
                            yield VADPredictionEvent(
                                timestamp_s=timestamp_s, channel=target_ch, is_speech=True
                            )
                    elif not is_speech and _was_speech:
                        _was_speech = False
                        if self.emit_as == "turn_taking":
                            yield TurnTakingPredictionEvent(
                                timestamp_s=timestamp_s, channel=target_ch, event_kind="offset"
                            )
                        elif self.emit_as == "vad":
                            yield VADPredictionEvent(
                                timestamp_s=timestamp_s, channel=target_ch, is_speech=False
                            )
                else:
                    yield MoshiPredictionEvent(
                        timestamp_s=timestamp_s, text_token=text_token, audio_codes=audio_codes
                    )
