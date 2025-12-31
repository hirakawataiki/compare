# src/nlp/yahoo_keyphrase.py

"""
LINEヤフー テキスト解析API（キーフレーズ抽出）のクライアント。

・環境変数 YAHOO_APPID に Client ID を入れておくこと
・エラー時は例外を投げるので、呼び出し側で try/except してください
"""

import logging
from typing import List, Tuple

import requests

from src.core.config import YAHOO_APPID

logger = logging.getLogger(__name__)

API_URL = "https://jlp.yahooapis.jp/KeyphraseService/V2/extract"


class YahooKeyphraseError(Exception):
    """Yahoo Keyphrase APIでエラーが起きたときの例外。"""
    pass


def extract_keyphrases_yahoo(text: str, max_phrases: int = 8) -> List[Tuple[str, int]]:
    """
    指定したテキストからキーフレーズを抽出する。

    Parameters
    ----------
    text : str
        解析したい日本語テキスト。
    max_phrases : int, optional
        重要度の高い順に最大いくつまで返すか。

    Returns
    -------
    List[Tuple[str, int]]
        (フレーズ文字列, スコア) のリスト。
        スコアは 0〜100 の整数で、100 に近いほど重要度が高い。
    """
    if not YAHOO_APPID:
        raise YahooKeyphraseError(
            "YAHOO_APPID が設定されていません。.env に YAHOO_APPID=... を追加してください。"
        )

    if not text.strip():
        return []

    # JSON-RPC 2.0形式のリクエストボディ
    payload = {
        "id": "conv-1",
        "jsonrpc": "2.0",
        "method": "jlp.keyphraseservice.extract",
        "params": {
            "q": text
        }
    }

    params = {"appid": YAHOO_APPID}

    try:
        resp = requests.post(API_URL, params=params, json=payload, timeout=5)
    except Exception as e:
        logger.exception("Yahoo Keyphrase API への接続に失敗しました")
        raise YahooKeyphraseError(f"request failed: {e}") from e

    if resp.status_code != 200:
        raise YahooKeyphraseError(
            f"HTTP error {resp.status_code}: {resp.text[:200]}"
        )

    data = resp.json()

    # API側の論理エラー
    if "error" in data:
        raise YahooKeyphraseError(f"API error: {data['error']}")

    phrases = data.get("result", {}).get("phrases", [])
    # score の高い順にソートして上位だけ返す
    phrases_sorted = sorted(
        phrases,
        key=lambda p: p.get("score", 0),
        reverse=True
    )

    result: List[Tuple[str, int]] = []
    for p in phrases_sorted[:max_phrases]:
        text_ = p.get("text")
        score_ = p.get("score")
        if text_ and isinstance(score_, int):
            result.append((text_, score_))

    return result
