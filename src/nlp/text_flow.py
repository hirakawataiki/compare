# src/nlp/text_flow.py
from __future__ import annotations

import json
import logging
from typing import Dict, Any, Optional, List

import requests

from .nlp_ja import extract_keywords as extract_keywords_sudachi, analyze as analyze_ja
from src.core.config import LLM_API_BASE, LLM_API_KEY, LLM_MODEL
from src.core.safety import filter_questions
from .yahoo_keyphrase import extract_keyphrases_yahoo, YahooKeyphraseError

logger = logging.getLogger(__name__)

# ==========================================================
# 0. LLM 用のプロンプトとユーティリティ
# ==========================================================

SYSTEM = """
あなたは初対面に近い二者の会話を、長期的な関係構築の観点から支援するアシスタントです。

出力は必ず次の JSON スキーマで返してください（日本語、UTF-8）:

{
  "safe": true/false,
  "topics": [string, ...],
  "questions_by_topic": {
    "<topic>": {
      "shallow": [string, ...],
      "medium":  [string, ...],
      "deep":    [string, ...]
    },
    ...
  },
  "notes": string
}

制約:
- センシティブな話題（政治・宗教・収入・健康・恋愛・個人特定など）は避けるか、マイルドな聞き方に言い換える。
- 相手を詰問したり、評価・ジャッジするような聞き方は避ける。
- 相手のペースや安心感を尊重し、Yes/No で答えられるライトな質問から、様子を見て少しずつ深める。
- 学生同士や若い社会人が初対面〜数回目の雑談で使うことを想定した口調にする。
- 出力は JSON オブジェクトのみとし、前後に説明文や ``` などは絶対に付けない。
"""

def build_user_prompt(
    input_text: str,
    topics: List[str],
    prefs: Optional[Dict[str, Any]] = None,
) -> str:
    """会話テキスト + 話題 + ユーザ設定から、LLM 用の user プロンプトを組み立てる。"""
    prefs = prefs or {}
    goal = (prefs.get("goal") or "").strip() or "初対面から自然に雑談を広げ、次回も話しやすい関係を作る"
    style = (prefs.get("style") or "").strip() or "丁寧で配慮あるカジュアル"
    avoid = (prefs.get("avoid") or "").strip() or "政治, 宗教, 収入, 健康, 恋愛, 個人特定"

    topics_str = ", ".join(topics) if topics else "（まだ話題候補は抽出されていません）"

    return f"""
【会話テキスト】
{input_text}

【会話の話題候補】
{topics_str}

【関係目標】
{goal}

【会話のトーン】
{style}

【避けたい話題・配慮したい点】
{avoid}

上の情報を踏まえて、それぞれの話題ごとに shallow / medium / deep の3段階の質問案を作成してください。
出力は指定の JSON スキーマのみで返してください。
""".strip()


def _call_llm(system: str, user: str, timeout: int = 30) -> Dict[str, Any]:
    """
    OpenAI 互換の /chat/completions エンドポイントを叩いて、
    JSON 文字列として返ってきた content を dict にパースする。
    """
    if not LLM_API_BASE or not LLM_API_KEY:
        raise RuntimeError("LLM_API_BASE または LLM_API_KEY が設定されていません。")

    url = LLM_API_BASE.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        # JSON で返してほしいことを明示
        "response_format": {"type": "json_object"},
        "temperature": 0.7,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]

    try:
        parsed = json.loads(content)
    except Exception:
        logger.exception("LLM からの応答を JSON として解釈できませんでした: %s", content[:2000])
        # 最低限のスキーマだけ埋める
        parsed = {
            "safe": False,
            "topics": [],
            "questions_by_topic": {},
            "notes": "LLM 出力の JSON 解析に失敗しました。",
        }
    return parsed


def _build_mock_questions_for_topics(topics: List[str]) -> Dict[str, Any]:
    """
    LLM_API_KEY が設定されていない場合に使う「モック質問生成」。
    フロントエンドの動作確認用。
    """
    qbt: Dict[str, Dict[str, List[str]]] = {}
    for t in topics:
        qbt[t] = {
            "shallow": [f"{t} について、ざっくりどんなところが好きですか？"],
            "medium": [f"{t} を通じて、どんな経験が印象に残っていますか？"],
            "deep": [f"{t} に関して、これからやってみたいことや、少し踏み込んだ話題はありますか？"],
        }

    return {
        "safe": True,
        "topics": topics,
        "questions_by_topic": qbt,
        "notes": "モック生成（LLM_API_KEY 未設定のため）",
    }


# ==========================================================
# 1. Yahoo キーフレーズ → 「話題候補の名詞」に落とし込むヘルパー
# ==========================================================

# 「名詞だけど話題にしたくないもの」をここで列挙
_TOPIC_STOPWORDS = set("""
先 こと 場合 部分 所 ところ 以上 以下 時 今 今日 明日 今夜 以前 以降
前後 人たち 自分 こちら あちら どちら みんな 私 僕 俺
""".split())

def _extract_noun_candidates_from_phrase(phrase: str) -> List[str]:
    """
    Yahooのキーフレーズ（例: 「展示会の準備を進める」）から
    「話題にしたい名詞」の候補だけを抽出する。

    - Sudachiで形態素解析
    - 品詞が「名詞」のものだけ採用
    - 1文字だけ/ASCIIのみ/ストップワードは除外
    """
    candidates: List[str] = []

    for surface, lemma, pos in analyze_ja(phrase):
        # lemma が空のときは surface を使う
        base = lemma or surface

        # 品詞の先頭が「名詞」以外はスキップ
        if not pos or pos[0] != "名詞":
            continue

        # 1文字だけのもの・ASCIIだけのものは除外
        if len(base) < 2:
            continue
        if base.isascii():
            continue

        # トピック用のストップワードは除外
        if base in _TOPIC_STOPWORDS:
            continue

        candidates.append(base)

    return candidates


# ==========================================================
# 2. 話題抽出本体
# ==========================================================
def extract_topics(input_text: str, topk: int = 8) -> List[str]:
    """
    テキストから会話の「話題」候補を抽出する。

    優先順位:
      1. LINEヤフー キーフレーズ抽出API
         → Sudachi で形態素解析して「名詞」に絞り込む
         → 名詞ごとにスコアを集約して上位 topk 件を返す
      2. Sudachiベースのローカル抽出（フォールバック）
    """
    input_text = (input_text or "").strip()
    if not input_text:
        return []

    # ------------------------------
    # 1) まず Yahoo API を試す
    # ------------------------------
    try:
        # 少し多めに候補をもらってからフィルタする
        phrases = extract_keyphrases_yahoo(input_text, max_phrases=32)
        # phrases: List[Tuple[str, int]] = [(text, score), ...]

        topic_scores: Dict[str, float] = {}

        for text_, score_ in phrases:
            # そのフレーズの中から「話題にしたい名詞」だけ抜き出す
            nouns = _extract_noun_candidates_from_phrase(text_)
            if not nouns:
                continue

            # いまはシンプルに「一番長い名詞」をそのフレーズの代表にする
            head = max(nouns, key=len)

            # 同じ名詞が複数フレーズに出てきた場合は、スコアの最大値を採用
            prev = topic_scores.get(head, 0.0)
            topic_scores[head] = max(prev, float(score_))

        # Yahoo からまともな話題が取れなかった場合は Sudachi にフォールバック
        if not topic_scores:
            raise ValueError("no good noun topics from Yahoo keyphrases")

        # スコアの高い順に並べて topk 個だけ返す
        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        topics = [w for (w, _) in sorted_topics[:topk]]

        logger.info("extract_topics: Yahoo -> %s", topics)
        return topics

    except (YahooKeyphraseError, Exception) as e:
        # Yahooのエラー or 名詞が全く取れなかった場合
        logger.warning("Yahoo keyphrase failed or no good topics, fallback to Sudachi: %s", e)

    # ------------------------------
    # 2) Sudachi ベースのローカル抽出
    # ------------------------------
    topics = extract_keywords_sudachi(input_text, topk=topk)
    logger.info("extract_topics: Sudachi -> %s", topics)
    return topics


# ==========================================================
# 3. フロー本体：テキスト → 話題 + 質問文
# ==========================================================
def run_text_flow(
    input_text: str,
    prefs: Optional[Dict[str, Any]] = None,
    topk: int = 8,
    timeout_sec: int = 30,
) -> Dict[str, Any]:
    """
    フロントから呼ばれるメインの関数。

    - input_text: 会話テキスト全体
    - prefs: 関係目標 / トーンなど
    - topk: 話題の最大数
    """
    # 1) 話題抽出
    topics = extract_topics(input_text, topk=topk)

    # 2) LLM に渡す user プロンプト構築
    user = build_user_prompt(input_text, topics, prefs=prefs)

    # 3) 質問生成（本番: LLM / 開発中: モック）
    # ★ LLM_API_KEY が無いときは「モック質問生成」を使う
    if not LLM_API_KEY:
        raw = _build_mock_questions_for_topics(topics)
    else:
        raw = _call_llm(SYSTEM, user, timeout=timeout_sec)

    # 4) NG ワードフィルタを適用
    safe = filter_questions(raw)

    # 念のため topics を上書きしておく（LLM 側で微変更された場合でも UI と揃う）
    safe["topics"] = topics

    # フロントが使いやすいように、「生の結果」と「フィルタ後の結果」をセットで返す
    return {"topics": topics, "llm_raw": raw, "result": safe}
