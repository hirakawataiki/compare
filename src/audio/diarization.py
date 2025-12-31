# src/audio/diarization.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import os
import wave
import platform

class DiarizationError(RuntimeError):
    pass


@dataclass
class DiarizationSegment:
    start: float
    end: float
    speaker: str


def _wav_duration_sec(path: str) -> Optional[float]:
    try:
        with wave.open(path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate() or 0
            if rate <= 0:
                return None
            return frames / float(rate)
    except Exception:
        return None


_PYANNOTE_PATCHED = False
_HF_PATCHED = False
_SPEECHBRAIN_PATCHED = False
_AUDIO_DECODER_PATCHED = False


def _patch_pyannote_get_model() -> None:
    """pyannote が 'model@revision' 形式を受け取った時の互換パッチ。"""
    global _PYANNOTE_PATCHED
    if _PYANNOTE_PATCHED:
        return
    try:
        from pyannote.audio.pipelines.utils import getter as getter_mod  # type: ignore
        from pyannote.audio.core.model import Model  # type: ignore
        import sys
    except Exception:
        return

    orig_get_model = getter_mod.get_model

    def get_model_patched(model, token=None, cache_dir=None):
        if isinstance(model, str) and "@" in model:
            model_id, revision = model.split("@", 1)
            _model = Model.from_pretrained(
                model_id,
                revision=revision,
                token=token,
                cache_dir=cache_dir,
                strict=False,
            )
            if _model:
                _model.eval()
                return _model
        return orig_get_model(model, token=token, cache_dir=cache_dir)

    getter_mod.get_model = get_model_patched

    # 既に speaker_diarization が import 済みなら、そちらの参照も差し替える
    mod = sys.modules.get("pyannote.audio.pipelines.speaker_diarization")
    if mod is not None and hasattr(mod, "get_model"):
        try:
            setattr(mod, "get_model", get_model_patched)
        except Exception:
            pass

    _PYANNOTE_PATCHED = True


def _patch_hf_hub_download() -> None:
    """huggingface_hub の use_auth_token 互換パッチ。"""
    global _HF_PATCHED
    if _HF_PATCHED:
        return
    try:
        import huggingface_hub
    except Exception:
        return

    orig = getattr(huggingface_hub, "hf_hub_download", None)
    if orig is None:
        return

    def hf_hub_download_patched(*args, **kwargs):
        if "use_auth_token" in kwargs and "token" not in kwargs:
            kwargs["token"] = kwargs.pop("use_auth_token")
        return orig(*args, **kwargs)

    huggingface_hub.hf_hub_download = hf_hub_download_patched
    _HF_PATCHED = True


def _patch_audio_decoder() -> None:
    """torchcodec が無い環境向けに AudioDecoder を torchaudio ベースにフォールバックさせる。"""
    global _AUDIO_DECODER_PATCHED
    if _AUDIO_DECODER_PATCHED:
        return
    try:
        import torchaudio  # type: ignore
        import pyannote.audio.core.io as pa_io  # type: ignore
    except Exception:
        return

    # 既に AudioDecoder が定義されていれば何もしない
    if hasattr(pa_io, "AudioDecoder") and getattr(pa_io, "AudioDecoder") is not None:
        _AUDIO_DECODER_PATCHED = True
        return

    class _FallbackAudioStreamMetadata:
        def __init__(self, info):
            self.sample_rate = info.sample_rate
            self.num_channels = info.num_channels
            self.num_frames = info.num_frames
            self.duration_seconds_from_header = (
                float(info.num_frames) / float(info.sample_rate) if info.sample_rate else 0.0
            )

    class _FallbackAudioSamples:
        def __init__(self, data, sample_rate):
            self.data = data
            self.sample_rate = sample_rate

    class _FallbackAudioDecoder:
        def __init__(self, path):
            self.path = path
            info = torchaudio.info(path)
            self.metadata = _FallbackAudioStreamMetadata(info)

        def get_all_samples(self):
            data, sr = torchaudio.load(self.path)
            return _FallbackAudioSamples(data, sr)

        def get_samples_played_in_range(self, start: float, end: float):
            data, sr = torchaudio.load(self.path)
            start_sample = max(0, int(start * sr))
            end_sample = max(start_sample, int(end * sr))
            sliced = data[:, start_sample:end_sample]
            return _FallbackAudioSamples(sliced, sr)

    # モジュールにフォールバックを差し込む
    pa_io.AudioDecoder = _FallbackAudioDecoder  # type: ignore
    pa_io.AudioStreamMetadata = _FallbackAudioStreamMetadata  # type: ignore
    pa_io.AudioSamples = _FallbackAudioSamples  # type: ignore
    _AUDIO_DECODER_PATCHED = True


def _patch_speechbrain_fetch() -> None:
    """speechbrain の fetch で 404(custom.py) を ValueError に変換し、Windows では symlink を避ける互換パッチ。"""
    global _SPEECHBRAIN_PATCHED
    if _SPEECHBRAIN_PATCHED:
        return
    try:
        from speechbrain.utils import fetching as sb_fetching  # type: ignore
        import speechbrain.inference.interfaces as sb_interfaces  # type: ignore
    except Exception:
        return

    orig = getattr(sb_fetching, "fetch", None)
    if orig is None:
        return

    def fetch_patched(*args, **kwargs):
        # Windows では symlink が権限不足になりやすいので COPY に切り替える
        if platform.system() == "Windows":
            try:
                from speechbrain.utils.fetching import LocalStrategy  # type: ignore
                if kwargs.get("local_strategy", LocalStrategy.SYMLINK) == LocalStrategy.SYMLINK:
                    kwargs["local_strategy"] = LocalStrategy.COPY
            except Exception:
                pass

        try:
            return orig(*args, **kwargs)
        except Exception as e:
            filename = kwargs.get("filename")
            if len(args) >= 1 and filename is None:
                filename = args[0]
            if filename == "custom.py":
                msg = str(e)
                if "Entry Not Found" in msg or "404 Client Error" in msg:
                    # from_hparams 側で無視されるよう ValueError にする
                    raise ValueError("File not found on HF hub") from e
            raise

    sb_fetching.fetch = fetch_patched
    # interfaces.py は `from speechbrain.utils.fetching import fetch` を使うため、
    # 参照を差し替える
    try:
        sb_interfaces.fetch = fetch_patched
    except Exception:
        pass
    _SPEECHBRAIN_PATCHED = True


class DiarizationEngine:
    """
    pyannote.audio を利用した話者分離エンジン。
    - HUGGINGFACE_TOKEN / HF_TOKEN があれば使用
    - モデル名は DIARIZATION_MODEL で上書き可能
    """

    def __init__(self, model: Optional[str] = None, token: Optional[str] = None) -> None:
        self.model = (model or os.getenv("DIARIZATION_MODEL") or "pyannote/speaker-diarization-3.1").strip()
        self.token = token or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        self._pipeline = None

    def _load_pipeline(self) -> None:
        try:
            from pyannote.audio import Pipeline  # type: ignore
        except Exception as e:
            raise DiarizationError(
                "pyannote.audio が見つかりません。`pip install pyannote.audio` を実行してください。"
            ) from e

        _patch_hf_hub_download()
        _patch_pyannote_get_model()
        _patch_speechbrain_fetch()
        _patch_audio_decoder()

        if self.token:
            # pyannote のバージョン差異に対応
            try:
                self._pipeline = Pipeline.from_pretrained(self.model, token=self.token)
            except TypeError:
                # auth_token を受け付けない版もあるため、最後はトークン無しで試す
                self._pipeline = Pipeline.from_pretrained(self.model)
        else:
            self._pipeline = Pipeline.from_pretrained(self.model)

    def diarize_file(
        self,
        wav_path: str,
        num_speakers: Optional[int] = 2,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> List[DiarizationSegment]:
        if self._pipeline is None:
            self._load_pipeline()

        if self._pipeline is None:
            raise DiarizationError("Diarization pipeline の初期化に失敗しました。")

        kwargs: Dict[str, Any] = {}
        if min_speakers is not None or max_speakers is not None:
            kwargs["min_speakers"] = min_speakers
            kwargs["max_speakers"] = max_speakers
        elif num_speakers is not None:
            kwargs["num_speakers"] = num_speakers

        diarization = self._pipeline(wav_path, **kwargs)
        annotation = getattr(diarization, "speaker_diarization", diarization)

        segments: List[DiarizationSegment] = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            segments.append(
                DiarizationSegment(
                    start=float(turn.start),
                    end=float(turn.end),
                    speaker=str(speaker),
                )
            )

        if not segments:
            duration = _wav_duration_sec(wav_path)
            if duration is not None:
                segments.append(DiarizationSegment(start=0.0, end=float(duration), speaker="SPEAKER_00"))

        return segments
