# src/pipeline/segment_boundary.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import time


@dataclass
class SegmentResult:
    start_window: Optional[int]
    end_window: Optional[int]
    avg_score: Optional[float]
    n: int


@dataclass
class SegmentAverager:
    """
    /voice_features の window_index に紐づく score を蓄積し、
    「前回 cut 〜 今回 cut」区間の平均スコアを返す。
    """
    # (window_index, score, ts)
    samples: List[Tuple[int, float, float]] = field(default_factory=list)

    # 直前の cut の end（次の区間は last_cut_end より大きい window から）
    last_cut_end: Optional[int] = None

    # メモリ肥大防止（だいたい 3秒窓×600=30分）
    max_samples: int = 600

    def add_sample(self, window_index: int, score: float, ts: Optional[float] = None) -> None:
        if ts is None:
            ts = time.time()
        # window_index が None/不正のときは捨てる
        if window_index is None:
            return
        try:
            wi = int(window_index)
        except Exception:
            return

        # score が NaN っぽい/None のときは捨てる
        if score is None:
            return
        try:
            sc = float(score)
        except Exception:
            return

        self.samples.append((wi, sc, ts))

        # 古いものを落とす
        if len(self.samples) > self.max_samples:
            self.samples = self.samples[-self.max_samples :]

    def latest_window(self) -> Optional[int]:
        if not self.samples:
            return None
        return self.samples[-1][0]

    def cut(self, end_window: Optional[int] = None) -> SegmentResult:
        """
        last_cut_end(含まない) 〜 end_window(含む) の score 平均を返し、
        last_cut_end を end_window に更新する。
        """
        if not self.samples:
            return SegmentResult(self.last_cut_end, end_window, None, 0)

        if end_window is None:
            end_window = self.latest_window()

        if end_window is None:
            return SegmentResult(self.last_cut_end, None, None, 0)

        start_exclusive = self.last_cut_end

        vals: List[float] = []
        for wi, sc, _ts in self.samples:
            if start_exclusive is not None and wi <= start_exclusive:
                continue
            if wi <= end_window:
                vals.append(sc)

        if vals:
            avg = sum(vals) / len(vals)
            res = SegmentResult(
                start_window=(start_exclusive + 1) if start_exclusive is not None else None,
                end_window=end_window,
                avg_score=avg,
                n=len(vals),
            )
        else:
            res = SegmentResult(
                start_window=(start_exclusive + 1) if start_exclusive is not None else None,
                end_window=end_window,
                avg_score=None,
                n=0,
            )

        # 次の区間の開始点になる
        self.last_cut_end = end_window
        return res
