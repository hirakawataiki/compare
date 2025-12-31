from __future__ import annotations

"""
features_audio.py

会話の「1発話ぶんの音声」から、盛り上がり度に使う特徴量を計算するモジュール。

現時点では、

- librosa で WAV（モノラル）の波形を読み込む
- ピッチ（基本周波数）の変動量
- エネルギ（音量）の変動量
- 発話長さからのざっくりした話速

を数値として返す。

このモジュールは「Python 側で音声ファイルを持っている」ことを前提としており、
ブラウザからのリアルタイムストリーミングはまだ行っていない。
将来、diarization / VAD で切り出された 1 発話ごとの WAV をここに渡す想定。

返り値は engagement.VoiceEngagementCalibrator でそのまま使える shape に揃えている::

    feats = {
        "pitch_var": 0.23,
        "energy_var": 0.41,
        "speech_rate": 4.2,
        "turn_rate": 0.0,   # ここは後で会話レベルで上書きする想定
    }

"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import librosa


@dataclass
class AudioSegment:
    """1発話ぶんの音声を表す軽いコンテナ。

    - samples: モノラルの波形（numpy 1次元配列、振幅 -1〜+1 程度）
    - sr: サンプリングレート（例: 16000）
    """

    samples: np.ndarray
    sr: int

    @property
    def duration(self) -> float:
        return float(len(self.samples)) / float(self.sr) if self.sr > 0 else 0.0


# ============================================================
# 1. 音声の読み込みユーティリティ
# ============================================================

def load_mono_wav(path: str, target_sr: int = 16000) -> AudioSegment:
    """WAV (または librosa が読める音声) をモノラルで読み込む。

    - target_sr でリサンプリング（デフォルト 16kHz）
    - 振幅は -1〜+1 程度になるよう librosa 側で正規化される
    """
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    if y.ndim != 1:
        y = np.ascontiguousarray(y.reshape(-1))
    return AudioSegment(samples=y, sr=sr)


# ============================================================
# 2. 基本的な特徴量の計算
# ============================================================

def _compute_energy_variation(seg: AudioSegment) -> float:
    """フレームごとの RMS エネルギから「どれくらい抑揚があるか」を見る。

    ここでは単純に、フレーム RMS の標準偏差を返す。
    後で SpeakerBaseline で相対値を見るので、スケールは厳密でなくてよい。
    """
    if seg.duration <= 0:
        return 0.0

    frame_length = int(0.025 * seg.sr)  # 25ms
    hop_length = int(0.010 * seg.sr)    # 10ms

    rms = librosa.feature.rms(
        y=seg.samples, frame_length=frame_length, hop_length=hop_length
    )[0]
    if rms.size == 0:
        return 0.0

    return float(np.std(rms))


def _compute_pitch_variation(seg: AudioSegment) -> float:
    """YIN 法で推定したピッチ系列の「ばらつき」を返す。

    - 無声音区間は除外
    - ピッチ系列の log をとった標準偏差を返す（音程差に近い指標）
    """
    if seg.duration <= 0:
        return 0.0

    frame_length = int(0.050 * seg.sr)  # 50ms
    hop_length = int(0.010 * seg.sr)    # 10ms

    # librosa.yin で F0 を推定（Hz）
    f0 = librosa.yin(
        seg.samples,
        fmin=80.0,
        fmax=400.0,
        sr=seg.sr,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    # 無声音 (0 or nan) を除外
    f0 = np.asarray(f0, dtype=float)
    f0 = f0[np.isfinite(f0) & (f0 > 0.0)]
    if f0.size == 0:
        return 0.0

    log_f0 = np.log2(f0)  # 半音差に対応
    return float(np.std(log_f0))


def _compute_speech_rate(seg: AudioSegment, num_chars: Optional[int] = None) -> float:
    """ざっくりした話速を返す。

    ここでは一旦、
        - num_chars が与えられていれば: num_chars / duration
        - 無ければ: 簡易な「有声区間の長さ / duration」から proxy を作る
    という方針にしておく。

    後で Web Speech API の文字数や morpheme 数などを渡すことで精度を上げられる。
    """
    dur = max(seg.duration, 1e-3)

    if num_chars is not None and num_chars > 0:
        return float(num_chars) / dur

    # 音声だけからの簡易版: エネルギが一定以上のフレームを「有声」とみなす
    frame_length = int(0.025 * seg.sr)
    hop_length = int(0.010 * seg.sr)
    rms = librosa.feature.rms(
        y=seg.samples, frame_length=frame_length, hop_length=hop_length
    )[0]
    if rms.size == 0:
        return 0.0

    thr = float(np.median(rms) * 0.8)  # 適当な閾値
    voiced = rms > thr
    voiced_ratio = float(np.mean(voiced))
    # 「1秒あたり何文字相当か」のような proxy にするため、適当な係数を掛ける
    return voiced_ratio * 5.0


# ============================================================
# 3. 公開 API：1発話ぶんの特徴量をまとめて計算
# ============================================================

def extract_segment_features(
    seg: AudioSegment,
    *,
    num_chars: Optional[int] = None,
    turn_rate: float = 0.0,
) -> Dict[str, float]:
    """1発話ぶんの音声から、盛り上がり度に使う基本特徴量を抽出する。

    パラメータ:
        seg: AudioSegment (1発話ぶんの波形)
        num_chars: その発話に対応するテキストの文字数（あれば話速推定に使う）
        turn_rate: 単位時間あたりの話者交代数（会話全体の情報から後で上書きする想定）

    戻り値:
        {
            "pitch_var":  ...,
            "energy_var": ...,
            "speech_rate": ...,
            "turn_rate":   ...,
        }
    """
    pitch_var = _compute_pitch_variation(seg)
    energy_var = _compute_energy_variation(seg)
    speech_rate = _compute_speech_rate(seg, num_chars=num_chars)

    return {
        "pitch_var": float(pitch_var),
        "energy_var": float(energy_var),
        "speech_rate": float(speech_rate),
        "turn_rate": float(turn_rate),
    }


def extract_features_for_segments(
    wav_path: str,
    segments: list,
    *,
    target_sr: int = 16000,
    min_duration_sec: float = 0.15,
) -> list[dict]:
    """話者分離セグメントごとに特徴量を計算する。"""
    audio = load_mono_wav(wav_path, target_sr=target_sr)
    results: list[dict] = []

    for seg in segments:
        start = float(getattr(seg, "start", 0.0))
        end = float(getattr(seg, "end", 0.0))
        speaker = getattr(seg, "speaker", "SPEAKER_00")
        if end <= start or (end - start) < min_duration_sec:
            continue
        s = int(max(0.0, start) * audio.sr)
        e = int(max(0.0, end) * audio.sr)
        if e <= s:
            continue
        slice_samples = audio.samples[s:e]
        if slice_samples.size == 0:
            continue
        seg_audio = AudioSegment(samples=slice_samples, sr=audio.sr)
        feats = extract_segment_features(seg_audio)
        results.append({
            "start": start,
            "end": end,
            "speaker": speaker,
            "features": feats,
        })

    return results


# ============================================================
# 4. 使い方の例（将来のための参考）
# ============================================================

def example_usage(path: str, num_chars: Optional[int] = None) -> Dict[str, float]:
    """簡単なデモ用関数。

    将来、VSCode のターミナルから

        python -m src.features_audio path/to/utt.wav

    のように実行して挙動を確認できるようにしている。
    """
    seg = load_mono_wav(path)
    feats = extract_segment_features(seg, num_chars=num_chars)
    return feats


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Compute basic audio features for one segment.")
    parser.add_argument("path", help="path to audio file (wav, etc.)")
    parser.add_argument("--chars", type=int, default=None, help="number of characters in transcript (optional)")
    args = parser.parse_args()

    out = example_usage(args.path, num_chars=args.chars)
    print(json.dumps(out, ensure_ascii=False, indent=2))
