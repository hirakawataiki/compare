# Repository Guidelines

## Project Structure & Module Organization
- `src/app/`: FastAPI アプリ本体。`main.py` がエントリポイント。
- `src/audio/`: 音声処理（VAD、特徴量、ASR、盛り上がり度）。
- `src/nlp/`: 日本語NLP・話題抽出・質問生成。
- `src/pipeline/`: 会話パイプラインとスコア統合。
- `src/core/`: 設定・安全フィルタ・共通ユーティリティ。
- `src/data/`: UUDB などのデータ読み込み。
- `src/scripts/`: 補助スクリプト置き場（例: UUDB の統計確認など）。
- `ui/`: ブラウザ用の静的 UI (`index.html`)。
- `tmp_audio/`: 音声チャンクなど一時データの置き場（実験用途）。
- `requirements.txt`: Python 依存関係。
- `.env.example`: API キーやモデル設定のサンプル。

## Build, Test, and Development Commands
- `python -m venv .venv` / `source .venv/bin/activate`: 仮想環境の作成と有効化。
- `pip install -r requirements.txt`: 依存関係のインストール。
- `cp .env.example .env`: 環境変数の雛形を作成し、必要に応じて編集。
- `python -m uvicorn src.app.main:app --reload`: ローカル開発サーバ起動（`ui/index.html` を配信）。
- `curl http://localhost:8000/`: 起動確認（ブラウザでも可）。

## Coding Style & Naming Conventions
- インデントは 4 スペース、関数・変数は `snake_case`、クラスは `CamelCase`。
- API エンドポイントは `src/main.py` に集約し、処理ロジックは `src/` 配下のモジュールへ分離してください。
- 自動整形・lint ツールは未設定のため、必要なら導入方針を PR で提案してください。
- OS 依存パスは `.env` で渡し、コード内のハードコードは避けてください。

## Testing Guidelines
- 現状、自動テスト用ディレクトリやフレームワーク設定は見当たりません。動作確認は API の手動確認が中心です。
- 追加する場合は `tests/` を新設し、`pytest` の `test_*.py` 命名を推奨します。
- 手動確認の例: `/text_flow` や `/topics_incremental` への JSON POST、`/voice_features` のスコア応答。

## Commit & Pull Request Guidelines
- 既存のコミット履歴は最小限のため、明確な規約はありません。推奨は「短い命令形サマリ + 具体的な本文」です。
- PR には変更概要、動作確認手順、設定変更（`.env` など）を記載し、UI 変更があればスクリーンショットを添付してください。
- 一時データ（`tmp_audio/`）や実キーはコミットしないでください。

## Security & Configuration Tips
- 実キーは `.env` に置き、`.env.example` にダミー値を維持してください。
- ローカルパス（例: `UUDB_ROOT`）は環境依存のため、README や PR で補足説明をお願いします。

## Agent-Specific Notes
- 開発コマンドや構成が変わった場合は `AGENTS.md` を更新してください。
