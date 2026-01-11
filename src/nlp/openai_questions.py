# src/nlp/openai_questions.py
from __future__ import annotations

import json
import re
import os
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field
from openai import OpenAI


RAW_NOTE_MAX_CHARS = 2000  # 長すぎるとDevToolsが見づらいので上限

def _truncate(s: str, max_chars: int = RAW_NOTE_MAX_CHARS) -> str:
    s = s or ""
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + f"\n...[truncated {len(s) - max_chars} chars]"


class LLMQuestions(BaseModel):
    # 3つちょうどを強制（Pydantic v2）
    questions: list[str] = Field(min_length=3, max_length=3)

# ========= 1) 入出力スキーマ（API用） =========

class GenerateQuestionsRequest(BaseModel):
    topic: str = Field(..., description="ユーザーが選択した話題（名詞想定）")
    context_text: str = Field("", description="最近の会話テキスト（長すぎる場合はサーバ側で短縮）")


class QuestionsBlock(BaseModel):
    # UI互換のため残す（ただし今回は shallow に「3つだけ」入れる方針）
    shallow: List[str] = Field(default_factory=list)
    medium: List[str] = Field(default_factory=list)
    deep: List[str] = Field(default_factory=list)


class GenerateQuestionsResponse(BaseModel):
    topic: str
    questions: QuestionsBlock
    safe: bool = True
    notes: str = ""


# ========= 2) LLM呼び出し本体 =========

def _get_client() -> OpenAI:
    # openai>=2 系は環境変数 OPENAI_API_KEY を読む（.env は main.py 側で読み込む想定）
    return OpenAI()


def _build_prompt(topic: str, context_text: str) -> str:
    """
    目的：
      - 「レベル分け無し」で質問を3つだけ生成
      - 余計な出力をさせない（速度＆安定性）
    """

    # 遅延とコストを抑えるため、末尾中心に短めで渡す
    ctx = (context_text or "").strip()
    if len(ctx) > 1200:
        ctx = ctx[-1200:]  # 末尾1200文字

    # ★重要：出力形式は「UI互換のQuestionsBlock」を保ちつつ shallow に3つだけ入れる
    # medium/deep は空配列で固定する（UI側は空なら表示しないようにする）
    return f"""
あなたは「初対面の2人の関係構築」を支援する質問生成AIです。
ユーザーが選択した話題「{topic}」について、会話が自然に続き、相手に配慮した質問を3つ作ってください。

会話の直近文脈（参考。無視してもよい）:
{ctx if ctx else "（なし）"}

出力条件:
- 日本語
- 質問は短め（1文中心）
- “3つだけ”出す（番号を付けない）
- 余計な説明は書かない
- 次のJSONだけを返す（前後に文章を付けない）

出力JSONスキーマ（この形で固定）:
{{
  "questions_by_topic": {{
    "{topic}": {{
      "shallow": ["...","...","..."]
    }}
  }},
  "notes": ""
}}
""".strip()

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        # 先頭の ```json や ``` を落とす
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        # 末尾の ``` を落とす（末尾以外にあっても最悪OK）
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _parse_last_json_object(text: str) -> Any:
    """
    文字列の中に含まれる「最後のJSON（objectでもarrayでもOK）」を取り出す。
    - ```json ... ``` のコードフェンスも除去
    - 余計な文章が混ざっていても、最後に出てくる JSON を拾う
    """
    s = _strip_code_fences(text).strip()
    dec = json.JSONDecoder()

    last_obj: Any = None
    i = 0

    while i < len(s):
        # 次の { または [ を探す
        m = re.search(r"[\{\[]", s[i:])
        if not m:
            break
        j = i + m.start()

        try:
            obj, end = dec.raw_decode(s[j:])
            last_obj = obj
            i = j + end
        except Exception:
            i = j + 1

    if last_obj is None:
        raise ValueError("no JSON found in model output")

    return last_obj


def _normalize_questions_payload(raw_any: Any, topic: str) -> tuple[Dict[str, Any], str]:
    """
    LLM 出力を safety.py が期待するスキーマに正規化する。
    返り値: (payload, local_notes)
    """
    topic_key = (topic or "").strip()
    local_notes = ""

    def build_payload(shallow: List[str], medium: Optional[List[str]] = None, deep: Optional[List[str]] = None, notes: str = "") -> Dict[str, Any]:
        return {
            "topics": [topic_key] if topic_key else [],
            "questions_by_topic": {
                topic_key: {
                    "shallow": shallow or [],
                    "medium": medium or [],
                    "deep": deep or [],
                }
            },
            "notes": notes or "",
        }

    # list なら shallow 扱い
    if isinstance(raw_any, list):
        local_notes = "normalized: list -> questions_by_topic.shallow"
        return build_payload([x for x in raw_any if isinstance(x, str)]), local_notes

    if isinstance(raw_any, dict):
        notes = raw_any.get("notes", "") if isinstance(raw_any.get("notes", ""), str) else ""

        # すでに期待スキーマ
        qbt = raw_any.get("questions_by_topic")
        if isinstance(qbt, dict):
            payload = dict(raw_any)
            if not payload.get("topics"):
                payload["topics"] = list(qbt.keys())
            return payload, local_notes

        # {"questions":[...]} 形式
        questions = raw_any.get("questions")
        if isinstance(questions, list):
            local_notes = "normalized: {questions:[...]} -> questions_by_topic.shallow"
            return build_payload([x for x in questions if isinstance(x, str)], notes=notes), local_notes

        # {"questions": {"shallow":[...], ...}} 形式
        if isinstance(questions, dict):
            shallow = questions.get("shallow") if isinstance(questions.get("shallow"), list) else []
            medium = questions.get("medium") if isinstance(questions.get("medium"), list) else []
            deep = questions.get("deep") if isinstance(questions.get("deep"), list) else []
            local_notes = "normalized: {questions:{...}} -> questions_by_topic"
            return build_payload(shallow, medium, deep, notes=notes), local_notes

        # {"shallow":[...], "medium":[...], "deep":[...]} 形式
        shallow = raw_any.get("shallow") if isinstance(raw_any.get("shallow"), list) else []
        medium = raw_any.get("medium") if isinstance(raw_any.get("medium"), list) else []
        deep = raw_any.get("deep") if isinstance(raw_any.get("deep"), list) else []
        if shallow or medium or deep:
            local_notes = "normalized: top-level shallow/medium/deep -> questions_by_topic"
            return build_payload(shallow, medium, deep, notes=notes), local_notes

    raise ValueError(f"parsed JSON is not supported (got {type(raw_any).__name__})")


def generate_questions_for_topic(topic: str, context_text: str, model: Optional[str] = None) -> GenerateQuestionsResponse:
    """
    FastAPI から呼ばれる想定。
    - OpenAIに投げる → JSON抽出 → UI互換のQuestionsBlockへ整形
    """
    model = (model or os.getenv("LLM_MODEL") or "gpt-5-mini").strip()

    client = _get_client()
    prompt = _build_prompt(topic=topic, context_text=context_text)

    raw_obj: Dict[str, Any]

    try:
        # Responses API（最小パラメータで安定優先）
        # 速度を狙って、生成量を小さく＆温度低めにする
        resp = client.responses.create(
            model=model,
            input=prompt,
            max_output_tokens=800,
        )

        raw_text = ""
        try:
            raw_text = getattr(resp, "output_text", "") or ""
        except Exception:
            raw_text = ""

        # テキスト取得（SDK差分に備えて頑健に）
        text = ""
        try:
            text = resp.output_text
        except Exception:
            pass
        if not text:
            text = str(resp)
        
        raw_text_local = text

        raw_any = _parse_last_json_object(text)
        raw_obj, local_notes = _normalize_questions_payload(raw_any, topic)

    except Exception as e:
        # try側で text/raw_text_local が作れていない場合でも落ちないように保険
        raw_text_local = (
            locals().get("raw_text_local", "")
            or locals().get("text", "")
            or ""
        )

        # DevToolsで見やすいように長さ制限（お好みで調整）
        raw_preview = raw_text_local
        if len(raw_preview) > 2000:
            raw_preview = raw_preview[:2000] + "\n...(truncated)"

        return GenerateQuestionsResponse(
            topic=topic,
            questions=QuestionsBlock(),
            safe=False,
            notes=(
                f"OpenAI呼び出しまたはJSON解析に失敗: {e}\n\n"
                f"=== RAW_FROM_OPENAI(output_text) ===\n{raw_preview}"
            ),
        )

    cleaned = raw_obj if isinstance(raw_obj, dict) else {}

    # notes を合成（rawのnotes + 変換メモ）
    notes = cleaned.get("notes", "") if isinstance(cleaned, dict) else ""
    if local_notes:
        notes = (notes + " / " + local_notes).strip(" /")

    # --- ここから「質問取り出し」を堅牢化 ---
    qbt = cleaned.get("questions_by_topic") or {}

    topic_key = (topic or "").strip()

    # 1) まずは完全一致
    block = None
    if isinstance(qbt, dict):
        block = qbt.get(topic_key)

        # 2) strip一致で探す（LLMが " 良い " のように返すケース対策）
        if not block:
            for k, v in qbt.items():
                if str(k).strip() == topic_key:
                    block = v
                    break

        # 3) 1件しかないならそれを採用（topicキーがズレた/違うケース対策）
        if not block and len(qbt) == 1:
            block = next(iter(qbt.values()), None)

    # block が dict じゃなければ空扱い
    if not isinstance(block, dict):
        block = {}

    # shallow/medium/deep/（将来用に questions キーも）を全部候補に入れる
    candidates: List[str] = []
    for k in ("shallow", "medium", "deep", "questions"):
        v = block.get(k)
        if isinstance(v, list):
            candidates.extend([x for x in v if isinstance(x, str) and x.strip()])

    # それでも空なら、cleaned直下の keys を拾う（返却形式ブレ対策）
    if not candidates and isinstance(cleaned, dict):
        for k in ("questions", "shallow", "medium", "deep"):
            v = cleaned.get(k)
            if isinstance(v, list):
                candidates.extend([x for x in v if isinstance(x, str) and x.strip()])

    # 重複除去（順序維持）して3件に丸める
    merged: List[str] = []
    seen = set()
    for q in candidates:
        qq = q.strip()
        if qq and qq not in seen:
            seen.add(qq)
            merged.append(qq)
    merged = merged[:3]

    # もし全部消えた（抽出失敗）なら、定型を返す
    if not merged:
        merged = [
            f"「{topic_key}」について、最近いちばん印象に残ったことは？",
            f"それが「{topic_key}」に興味を持つきっかけになった出来事はある？",
            f"「{topic_key}」でおすすめ（やり方・場所・作品など）があれば教えて！",
        ]

    # 3つに絞る（念のため）
    merged = merged[:3]

    # FastAPI の response_model を満たすため、必ず GenerateQuestionsResponse を返す
    questions_block = QuestionsBlock(shallow=merged, medium=[], deep=[])

    safe_flag = bool(cleaned.get("safe", True)) if isinstance(cleaned, dict) else True
    notes = notes.strip()

    # 「配列が返ってきたので shallow にマップした」などの注釈があれば notes に追記
    try:
        if local_notes:
            notes = (notes + " | " if notes else "") + str(local_notes)
    except Exception:
        pass

    debug_raw = os.getenv("DEBUG_LLM_RAW", "").strip().lower() in {"1", "true", "yes"}
    if debug_raw:
        try:
            raw_preview = _truncate(locals().get("raw_text_local", ""))
        except Exception:
            raw_preview = (locals().get("raw_text_local", "") or "")[:2000]
        notes = (notes + "\n\n=== RAW_FROM_OPENAI(output_text) ===\n" + raw_preview).strip()

    return GenerateQuestionsResponse(
        topic=topic_key,
        questions=questions_block,
        safe=safe_flag,
        notes=notes,
    )
