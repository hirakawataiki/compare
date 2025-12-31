from __future__ import annotations

from typing import Optional
import os

import numpy as np

from src.audio.diarization import _patch_speechbrain_fetch


class SpeakerEmbeddingError(RuntimeError):
    pass


class SpeakerEmbeddingEngine:
    """音声から話者埋め込み（ベクトル）を抽出する薄いラッパー。"""

    def __init__(self, model: Optional[str] = None, device: Optional[str] = None) -> None:
        self.model = (model or os.getenv("SPEAKER_EMBEDDING_MODEL") or "speechbrain/spkrec-ecapa-voxceleb").strip()
        self.device = (device or os.getenv("SPEAKER_EMBEDDING_DEVICE") or "cpu").strip()
        self._model = None

    def _load(self) -> None:
        try:
            from speechbrain.inference.speaker import EncoderClassifier  # type: ignore
        except Exception:
            try:
                from speechbrain.pretrained import EncoderClassifier  # type: ignore
            except Exception as e:
                raise SpeakerEmbeddingError(
                    "speechbrain が見つかりません。`pip install speechbrain` を実行してください。"
                ) from e

        _patch_speechbrain_fetch()

        try:
            self._model = EncoderClassifier.from_hparams(
                source=self.model,
                run_opts={"device": self.device},
            )
        except Exception as e:
            raise SpeakerEmbeddingError(f"Embedding model の読み込みに失敗しました: {e}") from e

    def embed(self, samples: np.ndarray, sr: int) -> np.ndarray:
        if self._model is None:
            self._load()

        if self._model is None:
            raise SpeakerEmbeddingError("Embedding model の初期化に失敗しました。")

        import torch

        wav = np.asarray(samples, dtype=np.float32)
        if wav.ndim != 1:
            wav = wav.reshape(-1)
        if wav.size == 0:
            raise SpeakerEmbeddingError("empty audio for embedding")

        wav_tensor = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)

        # speechbrain の EncoderClassifier は sample rate を内部で扱う前提
        emb = self._model.encode_batch(wav_tensor)
        emb = emb.squeeze().detach().cpu().numpy()
        if emb.ndim != 1:
            emb = emb.reshape(-1)
        norm = float(np.linalg.norm(emb))
        if norm > 0:
            emb = emb / norm
        return emb
