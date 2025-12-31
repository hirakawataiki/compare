# src/audio/voice_engagement.py
"""
音声特徴量ベースの「盛り上がり度」を計算するランタイム。

目的（安全側）:
- ブラウザ側の特徴量が多少荒くても（例: pitch_var が 0 に張り付く等）、
  score が 0.5 付近に張り付かず、変化が出るようにする。
- 「その人の最初の数回」をベースライン（キャリブ）にして、以降は差分で評価する。
- 0.5 を「いつも通り」として、上振れ→>0.5、下振れ→<0.5 を許可する。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import math


@dataclass
class _VoiceBaseline:
    # 必須3特徴量（count はこれの基準）
    count: int = 0
    pitch_var_sum: float = 0.0
    energy_var_sum: float = 0.0
    rate_sum: float = 0.0

    # 追加特徴量（来たときだけ集計）
    pitch_mean_sum: float = 0.0
    pitch_mean_count: int = 0

    energy_mean_sum: float = 0.0
    energy_mean_count: int = 0

    voiced_sum: float = 0.0
    voiced_count: int = 0


class VoiceEngagementRuntime:
    def __init__(self, calib_utts: int = 5) -> None:
        self.calib_utts = calib_utts
        self._base = _VoiceBaseline()

        # ベースライン（必須3）
        self.pitch_mu: Optional[float] = None        # pitch_var の平均（0 でもOK）
        self.energy_mu: Optional[float] = None       # energy_var の平均
        self.rate_mu: Optional[float] = None         # speech_rate の平均

        # ベースライン（追加）
        self.pitch_mean_mu: Optional[float] = None   # pitch_mean の平均
        self.energy_mean_mu: Optional[float] = None  # energy_mean の平均
        self.voiced_mu: Optional[float] = None       # voiced_ratio の平均

        self.calibrated: bool = False

    # -----------------------------
    # util
    # -----------------------------
    @staticmethod
    def _is_finite(x: object) -> bool:
        return isinstance(x, (int, float)) and math.isfinite(float(x))

    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x

    # -----------------------------
    # baseline
    # -----------------------------
    def _update_baseline(self, feats: Dict[str, float]) -> None:
        b = self._base
        b.count += 1

        # 必須3（ない場合は 0 扱いにして落とさない）
        pv = float(feats.get("pitch_var", 0.0))
        ev = float(feats.get("energy_var", 0.0))
        sr = float(feats.get("speech_rate", 0.0))

        b.pitch_var_sum += pv
        b.energy_var_sum += ev
        b.rate_sum += sr

        # 追加（来ているときだけ）
        pm = feats.get("pitch_mean", None)
        if self._is_finite(pm) and float(pm) > 0.0:
            b.pitch_mean_sum += float(pm)
            b.pitch_mean_count += 1

        em = feats.get("energy_mean", None)
        if self._is_finite(em) and float(em) >= 0.0:
            b.energy_mean_sum += float(em)
            b.energy_mean_count += 1

        vr = feats.get("voiced_ratio", None)
        if self._is_finite(vr):
            v = float(vr)
            # voiced_ratio は [0,1] 想定（多少外れても丸める）
            v = self._clamp(v, 0.0, 1.0)
            b.voiced_sum += v
            b.voiced_count += 1

        if b.count >= self.calib_utts:
            # ベースライン確定
            self.pitch_mu = b.pitch_var_sum / b.count
            self.energy_mu = b.energy_var_sum / b.count
            self.rate_mu = b.rate_sum / b.count

            self.pitch_mean_mu = (b.pitch_mean_sum / b.pitch_mean_count) if b.pitch_mean_count > 0 else None
            self.energy_mean_mu = (b.energy_mean_sum / b.energy_mean_count) if b.energy_mean_count > 0 else None
            self.voiced_mu = (b.voiced_sum / b.voiced_count) if b.voiced_count > 0 else None

            self.calibrated = True

            # ログ（最小限）
            msg = (
                "[VoiceEngagementRuntime] baseline fixed: "
                f"pitch_var_mu={self.pitch_mu:.6f}, "
                f"energy_var_mu={self.energy_mu:.6f}, "
                f"rate_mu={self.rate_mu:.3f}"
            )
            if self.pitch_mean_mu is not None:
                msg += f", pitch_mean_mu={self.pitch_mean_mu:.2f}"
            if self.energy_mean_mu is not None:
                msg += f", energy_mean_mu={self.energy_mean_mu:.6f}"
            if self.voiced_mu is not None:
                msg += f", voiced_mu={self.voiced_mu:.3f}"
            print(msg)

    # -----------------------------
    # scoring
    # -----------------------------
    def _score_from_features(self, feats: Dict[str, float]) -> float:
        """
        0.5 を中心に上下するスコア。
        - 変化が小さい: 0.5 付近
        - 活発寄りの変化: >0.5
        - 静か寄りの変化: <0.5
        """
        if not self.calibrated:
            return 0.5

        eps = 1e-9

        pv = float(feats.get("pitch_var", 0.0))
        ev = float(feats.get("energy_var", 0.0))
        sr = float(feats.get("speech_rate", 0.0))

        pm = feats.get("pitch_mean", None)
        em = feats.get("energy_mean", None)
        vr = feats.get("voiced_ratio", None)

        # --- delta 設計（安全側：極端値を抑える） ---
        # energy系はスケールが揺れやすいので log1p を使って「大きすぎる値」の暴れを抑える
        def delta_log1p(v: float, mu: Optional[float]) -> float:
            if mu is None:
                return 0.0
            lv = math.log1p(max(v, 0.0))
            lmu = math.log1p(max(mu, 0.0))
            return (lv - lmu) / (abs(lmu) + 0.2 + eps)  # 0付近の発散を避けるため +0.2

        # speech_rate は相対差（muが 0 でも落とさない）
        def delta_rel(v: float, mu: Optional[float]) -> float:
            if mu is None:
                return 0.0
            return (v - mu) / (abs(mu) + 1.0 + eps)  # +1.0 で過剰増幅を抑える

        # pitch_mean は log2 比（音高は比で見る方が自然）
        def delta_pitch_mean(v: Optional[float], mu: Optional[float]) -> float:
            if v is None or mu is None:
                return 0.0
            if (not self._is_finite(v)) or (not self._is_finite(mu)):
                return 0.0
            v = float(v)
            mu = float(mu)
            if v <= 0.0 or mu <= 0.0:
                return 0.0
            return math.log(v / mu, 2.0)  # 上がると +, 下がると -

        # voiced_ratio は差分（0..1）
        def delta_voiced(v: Optional[float], mu: Optional[float]) -> float:
            if v is None or mu is None:
                return 0.0
            if (not self._is_finite(v)) or (not self._is_finite(mu)):
                return 0.0
            v = self._clamp(float(v), 0.0, 1.0)
            mu = self._clamp(float(mu), 0.0, 1.0)
            return (v - mu) / (0.3 + eps)  # 0.3 でスケール固定（荒れにくい）

        d_pitch_var = delta_log1p(pv, self.pitch_mu)      # pitch_var が 0 でもOK
        d_energy_var = delta_log1p(ev, self.energy_mu)
        d_rate = delta_rel(sr, self.rate_mu)

        d_pitch_mean = delta_pitch_mean(pm, self.pitch_mean_mu)
        d_energy_mean = delta_log1p(float(em), self.energy_mean_mu) if self._is_finite(em) else 0.0
        d_voiced = delta_voiced(vr, self.voiced_mu)

        # --- 安全クリップ（暴れ防止） ---
        d_pitch_var = self._clamp(d_pitch_var, -3.0, 3.0)
        d_energy_var = self._clamp(d_energy_var, -3.0, 3.0)
        d_rate = self._clamp(d_rate, -3.0, 3.0)
        d_pitch_mean = self._clamp(d_pitch_mean, -2.0, 2.0)
        d_energy_mean = self._clamp(d_energy_mean, -3.0, 3.0)
        d_voiced = self._clamp(d_voiced, -3.0, 3.0)

        # --- 重み（最初は控えめ・安全） ---
        # pitch_var が死んでも pitch_mean が効くように分散
        w_pitch_mean = 0.25
        w_pitch_var = 0.10
        w_energy_mean = 0.25
        w_energy_var = 0.20
        w_rate = 0.15
        w_voiced = 0.05

        x = (
            w_pitch_mean * d_pitch_mean +
            w_pitch_var * d_pitch_var +
            w_energy_mean * d_energy_mean +
            w_energy_var * d_energy_var +
            w_rate * d_rate +
            w_voiced * d_voiced
        )

        # 0.5 を中心に、上下を見やすく（k を上げると差が出やすい）
        k = 2.5
        x = self._clamp(k * x, -6.0, 6.0)
        score = 1.0 / (1.0 + math.exp(-x))

        # 念のため
        if score < 0.0:
            score = 0.0
        if score > 1.0:
            score = 1.0
        return float(score)

    # -----------------------------
    # public
    # -----------------------------
    def update(self, feats: Dict[str, float]) -> Tuple[Optional[float], str]:
        if not self.calibrated:
            self._update_baseline(feats)
            if not self.calibrated:
                return None, "calib"

        score = self._score_from_features(feats)
        return score, "run"
