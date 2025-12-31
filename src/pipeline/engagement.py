# src/pipeline/engagement.py
# 盛り上がり度（エンゲージメント）を扱うモジュール
#
# 役割は大きく 3 つ：
#  1. 音声特徴量から「話者ごとの盛り上がり度（音声側スコア）」を出す
#  2. UUDB からのスコアと音声側スコアを統合して E_total を出す
#  3. 直近の時系列からベースラインや low/mid/high ラベルを計算する
#
# まだ音声特徴量パイプラインとは完全には接続していないが、
# 将来 features_audio / diarization からこのモジュールを呼べるように
# クラス設計だけ先に固めておく。

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Dict, List, Optional

import math
import numpy as np


# ============================================================
# 1. 「どのくらいのスコアが low / mid / high か」のラベル付け
# ============================================================

def classify_engagement(score: float) -> str:
    """0〜1 のスコアを 'low' / 'mid' / 'high' にざっくり分類する。

    しきい値はとりあえず:
      - < 0.33  → 'low'
      - < 0.66  → 'mid'
      - それ以上 → 'high'

    将来、実験結果に応じて調整して構わない。
    """
    if score is None or not math.isfinite(score):
        return "unknown"

    if score < 0.33:
        return "low"
    if score < 0.66:
        return "mid"
    return "high"


# ============================================================
# 2. 話者ごとのキャリブレーション + 音声側スコア E_voice
# ============================================================

FEATURE_KEYS_DEFAULT: List[str] = [
    "pitch_var",      # ピッチ変動量
    "energy_var",     # 声量の変動
    "speech_rate",    # 話速（文字/秒など）
    "turn_rate",      # 単位時間あたりの話者交代
]


@dataclass
class SpeakerBaseline:
    """各話者ごとのキャリブレーション結果を保持する。

    - 初期状態では n_calib < calib_utts のあいだ baseline_* が徐々に更新される
    - n_calib >= calib_utts になったら「その人の基準値」として固定扱い
    """
    calib_utts: int = 5                     # 何発話ぶんをキャリブレーションに使うか
    n_calib: int = 0                        # いま何発話ぶんたまったか
    sums: Dict[str, float] = field(default_factory=dict)

    def is_calibrated(self) -> bool:
        return self.n_calib >= self.calib_utts

    def update_with_features(self, feats: Dict[str, float], keys: List[str]) -> None:
        """キャリブレーション用に発話ごとの特徴量を足し込む。"""
        for k in keys:
            v = float(feats.get(k, 0.0))
            self.sums[k] = self.sums.get(k, 0.0) + v
        self.n_calib += 1

    def get_baseline(self, key: str) -> float:
        """指定キーの基準値（平均）を返す。未定義なら 0.0。"""
        if self.n_calib <= 0:
            return 0.0
        return self.sums.get(key, 0.0) / float(self.n_calib)


class VoiceEngagementCalibrator:
    """話者ごとの音声特徴量から E_voice を出すためのクラス。

    想定する入力（1 発話あたり）:
        speaker_id: "spk1" など
        feats: {
            "pitch_var": ...,
            "energy_var": ...,
            "speech_rate": ...,
            "turn_rate": ...,
            （必要なら将来キーを増やしていく）
        }

    ロジック:
        - 各話者ごとに最初の calib_utts 発話をキャリブレーションに使う
        - キャリブ完了後は、現在値と baseline の「差分」を 0〜1 に正規化し、
          各特徴量の平均をその話者の E_voice_speaker にする
        - 複数話者のときは単純平均を取って全体の E_voice とする
    """

    def __init__(
        self,
        calib_utts: int = 5,
        feature_keys: Optional[List[str]] = None,
        # 変化量をどのくらい強くスコアに反映するかの係数
        delta_scale: float = 1.5,
    ) -> None:
        self.calib_utts = calib_utts
        self.feature_keys = feature_keys or FEATURE_KEYS_DEFAULT
        self.delta_scale = float(delta_scale)

        # speaker_id → SpeakerBaseline
        self._speakers: Dict[str, SpeakerBaseline] = {}

    # ---- 内部ユーティリティ ----

    def _get_or_create_speaker(self, speaker_id: str) -> SpeakerBaseline:
        if speaker_id not in self._speakers:
            self._speakers[speaker_id] = SpeakerBaseline(calib_utts=self.calib_utts)
        return self._speakers[speaker_id]

    def _feature_delta_score(self, value: float, baseline: float) -> float:
        """1つの特徴量に対して「どれだけ盛り上がっているか」を 0〜1 にマップする。

        - baseline からの相対差分 (value - baseline) / (|baseline| + eps) を計算
        - それを tanh で -1〜+1 に押し込み
        - 最後に 0〜1 に線形変換する
        """
        eps = 1e-6
        if baseline == 0.0:
            # まだ基準が無い or 非常に小さい場合は絶対値で見る
            diff = value
        else:
            diff = (value - baseline) / (abs(baseline) + eps)

        z = math.tanh(diff * self.delta_scale)  # -1〜+1 くらい
        return 0.5 * (z + 1.0)  # 0〜1 に変換

    # ---- 公開 API ----

    def update_speaker(
        self,
        speaker_id: str,
        feats: Dict[str, float],
    ) -> Dict[str, float]:
        """1 発話ぶんの特徴量を受け取り、現在の状態を返す。

        戻り値の例:
        {
            "phase": "calib" / "run",
            "speaker_score": 0.73  (calib 中は None),
        }
        """
        sp = self._get_or_create_speaker(speaker_id)

        if not sp.is_calibrated():
            sp.update_with_features(feats, self.feature_keys)
            return {"phase": "calib", "speaker_score": None}

        # ここから本番フェーズ
        per_feat_scores: List[float] = []
        for k in self.feature_keys:
            v = float(feats.get(k, 0.0))
            base = sp.get_baseline(k)
            per_feat_scores.append(self._feature_delta_score(v, base))

        if not per_feat_scores:
            speaker_score = 0.5  # 何も特徴量が無いときは中立
        else:
            speaker_score = float(sum(per_feat_scores) / len(per_feat_scores))

        return {"phase": "run", "speaker_score": speaker_score}

    def aggregate_over_speakers(self, current_scores: Dict[str, float]) -> float:
        """複数話者のスコアを単純平均して E_voice を出す。"""
        vals = [float(v) for v in current_scores.values() if v is not None]
        if not vals:
            return 0.5  # 中立
        return float(sum(vals) / len(vals))

    def is_fully_calibrated(self) -> bool:
        """すべての話者が calib_utts 発話ぶんたまっているかどうか。"""
        if not self._speakers:
            return False
        return all(sp.is_calibrated() for sp in self._speakers.values())


# ============================================================
# 3. UUDB スコアとの統合
# ============================================================

def combine_engagement_scores(
    uudb_score: Optional[float],
    voice_score: Optional[float],
    w_uudb: float = 0.6,
    w_voice: float = 0.4,
) -> float:
    """UUDB 側と音声側のスコアを統合して E_total を返す。

    - 入力はどちらも 0〜1 を想定
    - どちらかが None のときは、もう一方だけで計算する
    - 両方 None の場合は 0.5（中立）を返す
    """
    has_u = uudb_score is not None and math.isfinite(uudb_score)
    has_v = voice_score is not None and math.isfinite(voice_score)

    if has_u and has_v:
        # 重みは正規化しておく
        w_sum = w_uudb + w_voice
        if w_sum <= 0:
            w_u, w_v = 0.5, 0.5
        else:
            w_u, w_v = w_uudb / w_sum, w_voice / w_sum
        return float(w_u * float(uudb_score) + w_v * float(voice_score))

    if has_u:
        return float(uudb_score)
    if has_v:
        return float(voice_score)
    return 0.5


# ============================================================
# 4. 直近ウィンドウの平均などを扱うシンプルなメーター
# ============================================================

class EngagementMeter:
    """直近 window 個のスコアからベースラインを出すための簡易メーター。

    ここでは「すでに 0〜1 で計算済みの E_total」を受け取り、
    - そのまま score として返しつつ
    - rolling_mean() で「最近の平均」を出せるようにしている。

    既存コードとの互換性のために、update() の引数は dict も受けるようにし、
    以下の優先順位でスコアを取り出す:

        features["e_total"] → features["uudb_score"] → features["score"] → 0.5
    """

    def __init__(self, window: int = 30) -> None:
        self.window = int(window)
        self.buf: deque[float] = deque(maxlen=self.window)

    def _extract_score_from_features(self, features: Dict) -> float:
        if not isinstance(features, dict):
            # 数値が直接渡された場合（後方互換）
            try:
                return float(features)
            except Exception:
                return 0.5

        for key in ("e_total", "uudb_score", "score"):
            if key in features:
                try:
                    return float(features[key])
                except Exception:
                    continue
        return 0.5

    def update(self, features) -> float:
        """スコアを 1 点受け取り、バッファに追加して返す。"""
        s = self._extract_score_from_features(features)
        s = max(0.0, min(1.0, s))  # 念のためクリップ
        self.buf.append(s)
        return s

    def rolling_mean(self) -> float:
        if not self.buf:
            return 0.0
        return float(np.mean(self.buf))
