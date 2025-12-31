# src/scripts/debug_voice_engagement.py

from __future__ import annotations

"""
debug_voice_engagement.py

オフラインで「音声特徴量 → 話者ごとの盛り上がり度（音声側）」を試すためのデバッグ用スクリプト。

使い方（例）:
    cd プロジェクトのルート
    python -m src.debug_voice_engagement utt1.wav utt2.wav utt3.wav ...

与えた WAV 群は「同じ話者が順番に話した発話」とみなし、

- 最初の 5 発話はキャリブレーション（baseline 更新）
- それ以降の発話では 0〜1 のスコア（speaker_score）と low/mid/high を表示

を行う。
"""

import argparse
import json

from src.audio.features_audio import load_mono_wav, extract_segment_features
from src.pipeline.engagement import VoiceEngagementCalibrator, classify_engagement


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Debug voice-based engagement scoring for one speaker."
    )
    parser.add_argument(
        "wavs",
        nargs="+",
        help="1人の話者の発話を時系列順に並べた音声ファイルのパス（WAVなど）",
    )
    args = parser.parse_args()

    # 各話者5発話でキャリブレーションする設定（あなたの希望どおり）
    calib = VoiceEngagementCalibrator(calib_utts=5)

    speaker_id = "spk1"  # とりあえず1人だけ扱う
    current_scores = {}  # speaker_id -> 最新スコア（今回は1人だけ）

    print("=== Voice engagement debug (single speaker) ===")
    print(f"calib_utts = {calib.calib_utts}")
    print(f"num_utterances = {len(args.wavs)}")
    print()

    for idx, path in enumerate(args.wavs, start=1):
        print(f"--- Utterance {idx} ---")
        print(f"file: {path}")

        # 1) 音声を読み込む
        seg = load_mono_wav(path)

        # 2) 特徴量を抽出する
        feats = extract_segment_features(seg)
        print("features =", json.dumps(feats, ensure_ascii=False))

        # 3) 話者ごとのキャリブレーション / スコア更新
        info = calib.update_speaker(speaker_id, feats)

        phase = info.get("phase")
        spk_score = info.get("speaker_score", None)

        if phase == "calib":
            # まだ基準作りの段階
            print(f"phase = calib (キャリブレーション中) / score = {spk_score}")
        else:
            # 本番フェーズ：この発話のスコアが 0〜1 で計算される
            current_scores[speaker_id] = spk_score
            level = classify_engagement(spk_score)
            print("phase = run")
            print(f"  speaker_score = {spk_score:.3f}  (level = {level})")

        print()

    print("=== done ===")


if __name__ == "__main__":
    main()
