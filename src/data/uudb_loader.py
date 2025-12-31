# src/data/uudb_loader.py
"""
UUDB の var/ ディレクトリにあるファイルを読み込むためのユーティリティ。

主に以下を扱う：
- Cxxx.para  : 発話ごとの 6軸パラ言語ラベル（快・覚醒など）
- Cxxx.list  : 発話ごとの ID リスト（C001L_001 など）
- CxxxL_001.wav / CxxxR_001.wav : 発話ごとの切り出し音声

このモジュールの目的は、
「UUDB から 1 発話ごとのラベルを取り出して、
　盛り上がり度スコアを計算できるようにすること」
です。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Union

from src.core.config import UUDB_VAR_DIR


# 左右どちらの話者か
Channel = Literal["L", "R"]


@dataclass
class UUDCUtterance:
    """
    UUDB の 1 発話分の情報を表すデータクラス。
    """
    session_id: str
    channel: Channel
    index: int
    wav_path: Path
    labels: Dict[str, float]


def load_para_file(session_id: str) -> List[UUDCUtterance]:
    """
    指定したセッション ID（例: "C001"）に対応する .para / .list を読み、
    UUDCUtterance のリストを返す。

    - .para : 各行が「6軸ラベルだけ」の行（タブ区切り）
    - .list : 同じ順番で、発話ID (C001L_001 など) が1行ずつ
    """
    session_dir = UUDB_VAR_DIR / session_id
    para_path = session_dir / f"{session_id}.para"
    list_path = session_dir / f"{session_id}.list"

    if not para_path.exists():
        raise FileNotFoundError(f".para file not found: {para_path}")

    # --- 1) .para を読み込んで、6軸ラベルのリストにする ---
    label_rows: List[List[float]] = []
    with para_path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith(";"):
                continue
            cols = line.split()
            if len(cols) != 6:
                # 想定外の列数はスキップ
                continue
            scores = list(map(float, cols))
            label_rows.append(scores)

    # --- 2) 発話IDリストを読む（あれば .list を使う） ---
    utt_names: List[str] = []
    if list_path.exists():
        with list_path.open(encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or s.startswith(";"):
                    continue
                # 1列目だけ使う（後ろに何かあっても無視）
                utt_names.append(s.split()[0])
    else:
        # .list がない場合は仮の名前を振る（C001_001, C001_002, ...）
        utt_names = [f"{session_id}_{i+1:03d}" for i in range(len(label_rows))]

    # label_rows と utt_names をペアにする
    utterances: List[UUDCUtterance] = []

    for i, (utt_name, scores) in enumerate(zip(utt_names, label_rows)):
        # チャンネル推定（名前に L / R が含まれていればそれを使う）
        if "L" in utt_name and "R" in utt_name:
            channel: Channel = "L"  # どちらも含んでいたらとりあえず L
        elif "L" in utt_name:
            channel = "L"
        elif "R" in utt_name:
            channel = "R"
        else:
            channel = "L"  # 不明な場合は仮で L

        # 発話番号（名前の最後の "_XXX" から取る。失敗したら行番号）
        try:
            idx_part = utt_name.split("_")[-1]
            index = int(idx_part)
        except Exception:
            index = i + 1

        # --- ここが今回のポイント ---
        # utt_name が "C001R_001.wav" でも "C001R_001" でも .wav が二重にならないようにする
        base = utt_name
        lower = base.lower()
        if lower.endswith(".wav"):
            base = base[:-4]  # ".wav" を削る

        wav_path = session_dir / f"{base}.wav"

        labels = {
            "pleasantness": scores[0],
            "arousal": scores[1],
            "dominance": scores[2],
            "credibility": scores[3],
            "confidency": scores[4],
            "friendliness": scores[5],
        }

        utterances.append(
            UUDCUtterance(
                session_id=session_id,
                channel=channel,
                index=index,
                wav_path=wav_path,
                labels=labels,
            )
        )

    return utterances

# ==========
# 盛り上がり度スコア（暫定版）
# ==========

def compute_engagement_from_labels(labels: Dict[str, float]) -> float:
    """
    UUDB の 6軸ラベル（1〜7）から、
    0〜1 の「盛り上がり度スコア」を計算する暫定関数。

    ここでは例として、
        E = 0.4*Arousal + 0.3*Interest + 0.2*Pleasantness + 0.1*Positivity
      （各軸は 1〜7 → 0〜1 に線形正規化）
    という単純な重み付き平均を使う。
    """

    def norm(x: float) -> float:
        # 1〜7 → 0〜1 に変換
        return (x - 1.0) / 6.0

    arousal = norm(labels.get("arousal", 4.0))
    interest = norm(labels.get("interest", 4.0))
    pleasant = norm(labels.get("pleasantness", 4.0))
    positivity = norm(labels.get("positivity", 4.0))

    score = (
        0.4 * arousal +
        0.3 * interest +
        0.2 * pleasant +
        0.1 * positivity
    )

    # 念のため [0,1] にクリップ
    if score < 0.0:
        score = 0.0
    elif score > 1.0:
        score = 1.0

    return score


# ==========
# 簡単な動作確認用
# ==========

if __name__ == "__main__":
    # 手元にあるセッション ID に合わせて書き換えてOK（C001 や C031 など）
    test_session = "C001"

    print(f"Loading UUDB session: {test_session}")
    try:
        uttrs = load_para_file(test_session)
    except FileNotFoundError as e:
        print("ERROR:", e)
    else:
        print(f"  utterances: {len(uttrs)}")
        for u in uttrs[:10]:
            e = compute_engagement_from_labels(u.labels)
            print(
                f"{u.session_id}{u.channel}_{u.index:03d} "
                f"E={e:.3f} labels={u.labels}"
            )
