# src/pipeline/pipeline.py
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from collections import deque
import time
from typing import Any, Dict, List, Deque, Tuple, Optional

from .engagement import EngagementMeter, classify_engagement
from .intervention import should_intervene
from src.audio.voice_engagement import VoiceEngagementRuntime
from src.core.utils import clamp01  # あなたの utils.py にある前提。無ければ下の clamp01 を使う
from src.core.config import (
    UUDB_SESSIONS_DIR,
    SILENCE_SEC_THRESHOLD,
    ENGAGEMENT_DROP_THRESHOLD,
)
from .segment_boundary import SegmentAverager, SegmentResult

# utils.py に clamp01 が無い場合の保険（あるなら不要だが、あっても害はない）
def _clamp01(x: float) -> float:
    try:
        if x < 0.0:
            return 0.0
        if x > 1.0:
            return 1.0
        return float(x)
    except Exception:
        return 0.5


@dataclass
class UUDItem:
    label_score: float
    wav_path: Optional[Path] = None
    utt_id: str = ""


class ConversationPipeline:
    """
    Tick で stats を返すパイプライン。

    優先順位（安全・成功率優先）:
      1) /voice_features 等の live score（新しければ）
      2) 無ければ UUDB の label_score（あれば）
      3) どちらも無ければ 0.5
    """

    def __init__(self):
        # UUDB リプレイ（無くても起動できるように）
        self.uudb_items: List[UUDItem] = self._try_load_uudb_items("C001")
        self.uudb_index = 0

        # スコアの移動平均（baseline）
        self.meter = EngagementMeter()

        # /audio_chunk 用（残しておく：使わなくてもOK）
        self.voice_runtime_api = VoiceEngagementRuntime(calib_utts=5)

        # Live（/voice_features）入力を保持
        self.live_voice_score: Optional[float] = None
        self.live_voice_phase: str = "none"  # "calib" / "run" / "error" ...
        self.live_voice_window_index: int = -1
        self.live_voice_timestamp: float = 0.0
        self.live_voice_ttl_sec: float = 12.0  # 3秒窓ならこのくらいが安全

        # 沈黙計測
        self.last_speech_time = time.time()

        self._live_score_history: Deque[Tuple[float, float]] = deque()
        self._live_score_keep_sec: float = 15 * 60  # 15分ぶん保持（適当でOK）
        self._interval_cursor_ts: Optional[float] = None
        self._last_live_score: Optional[float] = None

        self.segmenter = SegmentAverager()

    # ---------------------------
    # UUDB を「読めたら読む」(読めなくても起動OK)
    # ---------------------------
    def _try_load_uudb_items(self, session_id: str) -> List[UUDItem]:
        try:
            # あなたの uudb_loader.py に合わせる（存在すれば使う）
            from src.data.uudb_loader import load_para_file, compute_engagement_from_labels  # type: ignore
        except Exception:
            # 関数名が違う/未実装でも起動を止めない
            print("[ConversationPipeline] UUDB loader not available. (OK: live voice only)")
            return []

        try:
            utterances = load_para_file(session_id)  # ここがあなたの実装に存在する前提
        except Exception as e:
            print(f"[ConversationPipeline] UUDB load_para_file failed: {e} (OK: live voice only)")
            return []

        items: List[UUDItem] = []
        for u in utterances:
            try:
                # u に session_id / channel / index / labels がある想定
                sid = getattr(u, "session_id", session_id)
                ch = getattr(u, "channel", "L")
                idx = int(getattr(u, "index", 1))
                labels = getattr(u, "labels", {})
                label_score = float(compute_engagement_from_labels(labels))
                utt_id = f"{sid}{ch}_{idx:03d}"

                # wav を探す（見つからなければ None）
                cand1 = UUDB_SESSIONS_DIR / sid / f"{utt_id}.wav"
                cand2 = UUDB_SESSIONS_DIR / f"{utt_id}.wav"
                wav_path = cand1 if cand1.exists() else (cand2 if cand2.exists() else None)

                items.append(UUDItem(label_score=label_score, wav_path=wav_path, utt_id=utt_id))
            except Exception:
                continue

        print(f"[ConversationPipeline] loaded UUDB items: {len(items)} (session={session_id})")
        return items

    # ---------------------------
    # テキスト（沈黙リセット用）
    # ---------------------------
    def on_speech(self, text: str) -> None:
        self.last_speech_time = time.time()

    # ---------------------------
    # Live voice を受け取る（/voice_features から呼ぶ）
    # ---------------------------
    def on_live_voice(
        self,
        *,
        score: Optional[float],
        phase: str,
        window_index: int,
        timestamp: Optional[float] = None,
    ) -> None:
        ts = float(timestamp) if timestamp is not None else time.time()
        self.live_voice_score = None if score is None else float(score)
        self.live_voice_phase = str(phase or "none")
        self.live_voice_window_index = int(window_index)
        self.live_voice_timestamp = ts

        # run でスコアが出ているなら「発話があった扱い」にする
        if self.live_voice_phase == "run" and self.live_voice_score is not None:
            self.last_speech_time = ts

    # ---------------------------
    # Tick：stats を作って返す
    # ---------------------------
    def tick(self) -> Dict[str, Any]:
        now = time.time()
        silence = now - self.last_speech_time

        label_score: Optional[float] = None
        voice_score: Optional[float] = None

        # 1) live voice が新しければ最優先
        fresh_live = (
            self.live_voice_timestamp > 0.0
            and (now - self.live_voice_timestamp) <= self.live_voice_ttl_sec
        )
        if fresh_live and self.live_voice_phase == "run":
            voice_score = self.live_voice_score

        # 2) live が無ければ UUDB を1つ進める（あれば）
        if voice_score is None and self.uudb_index < len(self.uudb_items):
            item = self.uudb_items[self.uudb_index]
            self.uudb_index += 1
            label_score = float(item.label_score)

        # 3) 統合（今は安全最優先：liveがあるならlive、無ければlabel、どちらも無ければ0.5）
        if voice_score is not None:
            score = float(voice_score)
        elif label_score is not None:
            score = float(label_score)
        else:
            score = 0.5

        # clamp（utilsの clamp01 が壊れてても動くように）
        try:
            score = clamp01(score)
        except Exception:
            score = _clamp01(score)

        # baseline 更新＆算出
        self.meter.update(score)
        baseline = float(self.meter.rolling_mean())

        # level
        level = classify_engagement(score)

        # intervention
        sig = should_intervene(
            silence_sec=float(silence),
            current_score=float(score),
            baseline=float(baseline),
            silence_th=float(SILENCE_SEC_THRESHOLD),
            drop_ratio_th=float(ENGAGEMENT_DROP_THRESHOLD),
        )

        return {
            "score": score,
            "label_score": label_score,
            "voice_score": voice_score,
            "baseline": baseline,
            "silence": float(silence),
            "level": level,
            "intervene": {"trigger": bool(sig.trigger), "reason": sig.reason},
        }
    
    def reset_interval_cursor(self, now_ts: Optional[float] = None) -> None:
        """新しい音声ストリーム開始時などに、区間平均の起点をリセットする。"""
        if now_ts is None:
            now_ts = time.time()
        self._interval_cursor_ts = now_ts
        # 履歴を全消しするかは好み。安全側で “直近だけ” 残すなら消してOK
        self._live_score_history.clear()
        self._last_live_score = None

    def push_live_score(self, ts: float, score: float) -> None:
        """/voice_features で計算した score を時系列で保存する。"""
        if score is None:
            return
        self._last_live_score = float(score)
        self._live_score_history.append((float(ts), float(score)))

        # 古い履歴を間引く
        cutoff = float(ts) - self._live_score_keep_sec
        while self._live_score_history and self._live_score_history[0][0] < cutoff:
            self._live_score_history.popleft()

    def consume_interval_avg(self, end_ts: Optional[float] = None, default: float = 0.5) -> float:
        """
        前回の話題抽出（cursor）から今回（end_ts）までの平均 score を返し、
        cursor を end_ts に進める。= 話題ごとに「1回だけ」割り当てるための関数。
        """
        if end_ts is None:
            end_ts = time.time()
        end_ts = float(end_ts)

        # 初回は「履歴の最初」から（なければ end_ts-3秒）
        if self._interval_cursor_ts is None:
            if self._live_score_history:
                start_ts = self._live_score_history[0][0]
            else:
                start_ts = end_ts - 3.0
        else:
            start_ts = float(self._interval_cursor_ts)

        vals = [s for (t, s) in self._live_score_history if (start_ts < t <= end_ts)]

        if vals:
            avg = sum(vals) / len(vals)
        else:
            # 区間にサンプルが無いときは直近スコア or 0.5
            avg = self._last_live_score if self._last_live_score is not None else float(default)

        self._interval_cursor_ts = end_ts
        return float(avg)
    
    def record_voice_score(self, window_index: int, score: float) -> None:
        # /voice_features で得た score を蓄積
        self.segmenter.add_sample(window_index=window_index, score=score)

    def cut_segment_for_new_topic(self) -> SegmentResult:
        # 「前回話題確定〜今回話題確定」区間の平均を1回だけ確定
        return self.segmenter.cut()
