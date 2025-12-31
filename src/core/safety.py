#LLMが出した質問にNG表現が紛れたら落とす。出力後の安全フィルタ

# src/core/safety.py
"""
質問文の安全フィルタと、
設計者が指定する NG ワード／推奨方向の定義をまとめたモジュール。

ポイント:
- HARD_NG_KEYWORDS: これに引っかかった質問は「絶対に採用しない」
- SOFT_NG_KEYWORDS: できるだけ避けたい方向（現段階では主にプロンプト側で使用）
- RECOMMENDED_DIRECTIONS: 「こういう方向の質問だと好ましい」という設計者の意図
- filter_questions(raw): LLM からの JSON 出力に対して NG チェックをかける
"""

from __future__ import annotations
from typing import Dict, Any, List

# ======== 1. NG / 推奨リスト ========

#: 絶対に避けたいキーワード（含まれていたら質問ごと削除）
HARD_NG_KEYWORDS: List[str] = [
    # プライバシー・個人特定系
    "住所", "電話番号", "LINE", "LINE ID", "連絡先",
    "本名", "フルネーム",

    # 政治・宗教・お金
    "政治", "政党", "選挙",
    "宗教", "信仰",
    "年収", "給料", "収入", "資産",

    # 健康・センシティブ
    "病気", "持病", "障害", "メンタル", "うつ",

    # 恋愛・セクシャル（今回の研究では踏み込まない想定）
    "恋愛", "元カレ", "元彼女", "浮気",

    # 露骨な攻撃・暴力
    "死ね", "殺す", "バカ", "ブス", "キモい",
]

#: できれば避けたいキーワード（現状は主にプロンプトで「触れないで」と伝える）
SOFT_NG_KEYWORDS: List[str] = [
    "コンプレックス",
    "トラウマ",
    "つらい過去",
    "失敗談をえぐるような話",
]

#: 「こういう方向の質問だと好ましい」という設計者の意図
RECOMMENDED_DIRECTIONS: List[str] = [
    "趣味や好きなこと",
    "休日や休みの日の過ごし方",
    "最近のうれしかった出来事",
    "これからやってみたいこと・興味",
    "印象的だったポジティブな思い出",
    "その人らしさが自然に見えるエピソード",
]


# ======== 2. 内部ユーティリティ ========

def _contains_any(text: str, keywords: List[str]) -> bool:
    """
    text に keywords のいずれかが「部分一致」するかを判定する簡易関数。

    - 日本語なので単語分割はしていない（まずはシンプルに substring マッチ）
    - 後で精度を上げたくなったら、ここに形態素解析等を導入すれば良い
    """
    if not text:
        return False
    for kw in keywords:
        if kw and kw in text:
            return True
    return False


# ======== 3. 質問フィルタ本体 ========

def filter_questions(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    LLM からの JSON 出力 (raw) に対して NG チェックを行い、
    危険な質問を取り除いた結果を返す。

    期待する raw の形式:
    {
      "safe": true/false,
      "topics": [...],
      "questions_by_topic": {
         "<topic>": {
           "shallow": [string, ...],
           "medium":  [string, ...],
           "deep":    [string, ...]
         },
         ...
      },
      "notes": "..."
    }

    戻り値も同じスキーマを保つように整形する。
    """
    if not isinstance(raw, dict):
        # 想定外の形式なら、そのまま「安全ではない」として返す
        return {"safe": False, "error": "raw is not dict", "raw": raw}

    qbt = raw.get("questions_by_topic") or {}
    if not isinstance(qbt, dict):
        return {"safe": False, "error": "questions_by_topic is not dict", "raw": raw}

    topics = raw.get("topics") or list(qbt.keys())
    notes = raw.get("notes", "")

    cleaned_qbt: Dict[str, Dict[str, List[str]]] = {}
    any_ok = False  # 少なくとも 1 つでも安全な質問が残ったか

    for topic, levels in qbt.items():
        # levels は {"shallow": [...], "medium": [...], "deep": [...]} を想定
        if not isinstance(levels, dict):
            continue

        cleaned_levels: Dict[str, List[str]] = {}
        for depth in ("shallow", "medium", "deep"):
            qs = levels.get(depth) or []
            if not isinstance(qs, list):
                qs = []

            safe_list: List[str] = []
            for q in qs:
                if not isinstance(q, str):
                    continue
                # HARD_NG に引っかかるものは採用しない
                if _contains_any(q, HARD_NG_KEYWORDS):
                    continue
                # ここで SOFT_NG まで全部弾くと質問が枯れやすいので、
                # 現段階では「プロンプトで避けるように伝えるだけ」にしておく。
                safe_list.append(q)

            if safe_list:
                cleaned_levels[depth] = safe_list
                any_ok = True

        if cleaned_levels:
            cleaned_qbt[topic] = cleaned_levels

    # 質問が 1 つも残らなかった場合の fallback
    if not any_ok:
        return {
            "safe": False,
            "topics": topics,
            "questions_by_topic": {},
            "notes": notes or "安全と判断できる質問が生成されませんでした。",
        }

    return {
        "safe": True,
        "topics": topics,
        "questions_by_topic": cleaned_qbt,
        "notes": notes,
    }

