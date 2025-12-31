#共通で使う小さな関数（正規化、時刻フォーマットなど）を集約# src/utils.py
# 共通で使う小さな関数（正規化、クリップなど）を集約

from __future__ import annotations

def clamp01(x: float) -> float:
    """数値を 0.0〜1.0 に丸める（安全版）。"""
    try:
        x = float(x)
    except Exception:
        return 0.5
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x
