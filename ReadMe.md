# Conversation Support System

初対面の雑談を支援するための研究用プロトタイプです。  
音声特徴量から盛り上がり度を推定し、テキストから話題抽出・質問生成を行います。

## セットアップ

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## 起動

```bash
python -m uvicorn src.app.main:app --reload
```

ローカルで UI を確認: `http://localhost:8000/`  
Swagger UI: `http://localhost:8000/docs`

## 話者分離（pyannote）

話者分離は `pyannote.audio` を使います。オンライン運用前提です。

1) 依存関係の追加
```bash
pip install pyannote.audio
```

2) 環境変数（`.env`）
```
HUGGINGFACE_TOKEN=hf_xxx
DIARIZATION_MODEL=pyannote/speaker-diarization-3.1
DIARIZATION_NUM_SPEAKERS=2
```

3) エンドポイント
- HTTP: `POST /diarize_chunk`
- WebSocket: `/ws_diarize`

## 主要エンドポイント

- `POST /text_flow`: 話題抽出 + 質問生成
- `POST /topics_incremental`: 話題抽出（チャンク）
- `POST /voice_features`: ブラウザ計算の音声特徴量 → 盛り上がり度
- `POST /audio_chunk`: 音声ファイル → 盛り上がり度

詳細は `API.md` を参照してください。
