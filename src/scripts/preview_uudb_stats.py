# src/scripts/preview_uudb_stats.py

from statistics import mean
from typing import List

from src.data.uudb_loader import load_para_file, compute_engagement_from_labels


def summarize_session(session_id: str) -> None:
    """指定したセッションの盛り上がり度スコアの統計量を表示する。"""
    uttrs = load_para_file(session_id)
    scores: List[float] = [compute_engagement_from_labels(u.labels) for u in uttrs]

    if not scores:
        print(f"{session_id}: 発話が 0 件です")
        return

    print(f"=== UUDB session {session_id} ===")
    print(f"  発話数: {len(scores)}")
    print(f"  最小値: {min(scores):.3f}")
    print(f"  最大値: {max(scores):.3f}")
    print(f"  平均値: {mean(scores):.3f}")

    # とりあえず 3 分割して件数を見る（0〜1 を 0.33 / 0.66 で区切る）
    low = [s for s in scores if s < 0.33]
    mid = [s for s in scores if 0.33 <= s < 0.66]
    high = [s for s in scores if s >= 0.66]

    print(f"  low (<0.33):  {len(low)} 件")
    print(f"  mid (0.33-):  {len(mid)} 件")
    print(f"  high (>=0.66): {len(high)} 件")


if __name__ == "__main__":
    # とりあえず C001 だけ見る。必要なら C031 とか増やしてもOK。
    summarize_session("C001")
