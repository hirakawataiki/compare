# API & WebSocket Documentation

FastAPI の自動ドキュメントも利用できます。
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`


<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [API & WebSocket Documentation](#api--websocket-documentation)
  - [Base URL](#base-url)
  - [認証](#認証)
  - [HTTP Endpoints](#http-endpoints)
    - [GET `/`](#get-)
    - [POST `/generate_questions`](#post-generate_questions)
    - [POST `/text_flow`](#post-text_flow)
    - [POST `/topics_incremental`](#post-topics_incremental)
    - [POST `/voice_features`](#post-voice_features)
    - [POST `/audio_chunk`](#post-audio_chunk)
    - [POST `/diarize_chunk`](#post-diarize_chunk)
  - [WebSocket](#websocket)
    - [WS `/ws`](#ws-ws)
    - [WS `/ws_diarize`](#ws-ws_diarize)
  - [設定（環境変数）](#設定環境変数)

<!-- /code_chunk_output -->



## Base URL
- ローカル開発: `http://localhost:8000`

## 認証
- なし（現状は全エンドポイント無認証）

## HTTP Endpoints

### GET `/`
UI を返します（`ui/index.html`）。

### POST `/generate_questions`
指定トピックに対して質問を3つ生成します。

Request (JSON)
```json
{
  "topic": "旅行",
  "context_text": "最近は友人と国内旅行の話をしていた"
}
```

Response (JSON)
```json
{
  "topic": "旅行",
  "questions": {
    "shallow": ["...","...","..."],
    "medium": [],
    "deep": []
  },
  "safe": true,
  "notes": ""
}
```

### POST `/text_flow`
会話テキストから話題抽出と質問生成をまとめて実行します。

Request (JSON)
```json
{
  "text": "最近カフェ巡りにハマっていて…",
  "prefs": {
    "goal": "初対面でも自然に会話が続くこと",
    "style": "丁寧で配慮あるカジュアル",
    "avoid": "政治, 宗教, 収入, 健康"
  }
}
```

Response (JSON)
```json
{
  "safe": true,
  "topics": ["カフェ", "休日"],
  "questions_by_topic": {
    "カフェ": { "shallow": ["..."], "medium": ["..."], "deep": ["..."] }
  },
  "notes": ""
}
```

### POST `/topics_incremental`
テキストチャンクから話題候補だけを抽出します。

Request (JSON)
```json
{ "text": "昨日は音楽フェスの話をしていて…", "topk": 8 }
```

Response (JSON)
```json
{
  "ok": true,
  "topics": [
    {
      "text": "音楽フェス",
      "score_avg": 0.62,
      "segment": { "start_window": 3, "end_window": 7, "n": 5 }
    }
  ]
}
```

### POST `/voice_features`
ブラウザ側で計算した音声特徴量を受け取り、盛り上がり度を返します。

Request (JSON)
```json
{
  "session_id": "s1",
  "speaker_id": "mix",
  "window_index": 0,
  "timestamp": 1735432100.0,
  "duration_sec": 3.0,
  "pitch_var": 0.12,
  "energy_var": 0.45,
  "speech_rate": 3.2,
  "voiced_ratio": 0.7,
  "pitch_mean": 220.0,
  "energy_mean": 0.02
}
```

Response (JSON)
```json
{
  "ok": true,
  "phase": "calib",
  "score": null,
  "baseline": {
    "pitch_mu": 0.1,
    "energy_mu": 0.4,
    "rate_mu": 3.0,
    "pitch_mean_mu": 210.0,
    "energy_mean_mu": 0.02,
    "voiced_mu": 0.6
  },
  "features": { "pitch_var": 0.12, "energy_var": 0.45, "speech_rate": 3.2 }
}
```

### POST `/audio_chunk`
音声ファイルを送ってサーバ側で特徴量を抽出し、盛り上がり度を返します。

Request (multipart/form-data)
- `file`: 音声ファイル（webm/wav など）

Response (JSON)
```json
{
  "ok": true,
  "phase": "run",
  "score": 0.58,
  "features": { "pitch_var": 0.1, "energy_var": 0.3, "speech_rate": 3.4 }
}
```

### POST `/diarize_chunk`
音声ファイルを送って話者分離を行い、話者ラベル付き区間を返します。

Request (multipart/form-data)
- `file`: 音声ファイル（webm/wav など）
- `offset_sec` (query): 連続ストリームのオフセット秒

Response (JSON)
```json
{
  "ok": true,
  "offset_sec": 0.0,
  "segments": [
    { "start": 0.1, "end": 1.2, "speaker": "SPEAKER_00" },
    { "start": 1.3, "end": 2.0, "speaker": "SPEAKER_01" }
  ],
  "metrics": {
    "turn_changes_chunk": 1,
    "tone_change_by_speaker": { "SPEAKER_00": 0.42 },
    "features_by_speaker": {
      "SPEAKER_00": { "pitch_var": 0.12, "energy_var": 0.31, "speech_rate": 3.2 }
    }
  }
}
```

## WebSocket

### WS `/ws`
接続直後に `connected`（テキスト）を送信します。

送信例:
```json
{"type":"tick"}
```

受信例:
```json
{
  "type": "stats",
  "payload": {
    "score": 0.6,
    "label_score": null,
    "voice_score": 0.6,
    "baseline": 0.55,
    "silence": 1.2,
    "level": "mid",
    "intervene": { "trigger": false, "reason": "" }
  }
}
```

補足:
- `PING`（JSON 文字列）を送ると `PONG` を返します。
- 介入トリガー時に `questions` を送る実装余地があります（現状は未使用の可能性あり）。

### WS `/ws_diarize`
音声チャンクを base64 で送ると、話者ラベル付き区間を返します。

送信例:
```json
{
  "type": "chunk",
  "audio_base64": "<base64>",
  "suffix": ".webm",
  "offset_sec": 0.0
}
```

受信例（話者分離の即時結果）:
```json
{
  "type": "segments",
  "payload": {
    "offset_sec": 0.0,
    "segments": [
      { "start": 0.1, "end": 1.2, "speaker": "SPEAKER_00" }
    ],
    "chunk_id": 0,
    "durations": { "diarize_sec": 1.23 },
    "metrics": {
      "turn_changes_chunk": 1,
      "turn_changes_total": 5,
      "tone_change_by_speaker": { "SPEAKER_00": 0.42 },
      "features_by_speaker": {
        "SPEAKER_00": { "pitch_var": 0.12, "energy_var": 0.003, "speech_rate": 2.4 }
      }
    }
  }
}
```

## 設定（環境変数）
`.env` に以下を設定します（`.env.example` 参照）。
- `LLM_API_BASE`, `LLM_API_KEY`, `LLM_MODEL`
- `DEBUG_LLM_RAW`（1/true の場合、LLMのRAW出力を `notes` に含める）
- `OPENAI_API_KEY`（OpenAI SDK 用）
- `YAHOO_APPID`（キーフレーズ抽出 API）
- `UUDB_ROOT`（UUDB データセットのルートパス）
- `AUDIO_CHUNK_MAX_MB`（`/audio_chunk` の最大アップロードサイズ）
- `DIARIZATION_MODEL`（pyannoteのモデル名）
- `DIARIZATION_MIN_SPEAKERS` / `DIARIZATION_MAX_SPEAKERS`（話者数の範囲）
- `HUGGINGFACE_TOKEN`（pyannote モデル取得用）
