# src/app/main.py
# FastAPI アプリ本体 + WebSocket + テキスト処理用 HTTP エンドポイント
# ＋ 任意音声ファイルから音声特徴量ベースの盛り上がり度を返す /audio_chunk

from typing import Optional, Dict, List
import asyncio
import base64
import time
import pathlib
import subprocess
import traceback
import wave
from types import SimpleNamespace

import numpy as np

from fastapi import FastAPI, WebSocket, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.websockets import WebSocketDisconnect

from src.pipeline.pipeline import ConversationPipeline
from src.nlp.text_flow import run_text_flow, extract_topics
from src.audio.features_audio import example_usage as extract_voice_features
from src.audio.voice_engagement import VoiceEngagementRuntime
from src.nlp.openai_questions import (
    GenerateQuestionsRequest,
    GenerateQuestionsResponse,
    generate_questions_for_topic,
)
from src.audio.diarization import DiarizationEngine, DiarizationError
from src.audio.speaker_embedding import SpeakerEmbeddingEngine, SpeakerEmbeddingError
from src.audio.features_audio import extract_features_for_segments, load_mono_wav
import os

app = FastAPI()

@app.post("/generate_questions", response_model=GenerateQuestionsResponse)
def generate_questions(req: GenerateQuestionsRequest):
    model = (os.getenv("LLM_MODEL") or "gpt-5-mini").strip()
    return generate_questions_for_topic(
        topic=req.topic,
        context_text=req.context_text,
        model=model,
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # file:// からの "null" origin もまとめて許可
    allow_credentials=False,    # 認証情報を送っていないので False でよい
    allow_methods=["*"],        # GET, /POST, OPTIONS など全部許可
    allow_headers=["*"],        # Content-Type など全部許可
)

# WebSocket 用のメインパイプライン（UUDBリプレイなど）
pipe = ConversationPipeline()

# 話者分離/埋め込みエンジン
diarizer = DiarizationEngine()
speaker_embedder = SpeakerEmbeddingEngine()
DIARIZATION_MIN_SPEAKERS = int(os.getenv("DIARIZATION_MIN_SPEAKERS", "1") or "1")
DIARIZATION_MAX_SPEAKERS = int(os.getenv("DIARIZATION_MAX_SPEAKERS", "2") or "2")
DIARIZATION_OVERLAP_SEC = float(os.getenv("DIARIZATION_OVERLAP_SEC", "1.0") or "1.0")
DIARIZATION_MIN_SEGMENT_SEC = float(os.getenv("DIARIZATION_MIN_SEGMENT_SEC", "0.3") or "0.3")
SPEAKER_EMBEDDING_ENABLED = (os.getenv("SPEAKER_EMBEDDING_ENABLED", "1") or "1").strip() not in ("0", "false", "False")
SPEAKER_EMBEDDING_MATCH_THRESHOLD = float(os.getenv("SPEAKER_EMBEDDING_MATCH_THRESHOLD", "0.65") or "0.65")

# /audio_chunk 用の「オンライン音声盛り上がり度ランタイム」
voice_runtime_api = VoiceEngagementRuntime(calib_utts=5)

# /voice_features 用の「ブラウザ計算済み特徴量」ランタイム（リアル音声）
voice_runtime_live = VoiceEngagementRuntime(calib_utts=6)


async def _save_upload_to_path(file: UploadFile, raw_path: pathlib.Path, max_bytes: int) -> int:
    total = 0
    try:
        with raw_path.open("wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    try:
                        raw_path.unlink()
                    except Exception:
                        pass
                    raise ValueError("uploaded file is too large")
                f.write(chunk)
    finally:
        try:
            await file.close()
        except Exception:
            pass
    return total


def _convert_to_wav(raw_path: pathlib.Path, suffix: str) -> tuple[pathlib.Path, bool, Optional[str]]:
    if suffix in [".wav", ".wave"]:
        return raw_path, False, None

    wav_path = raw_path.with_suffix(".wav")
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-i", str(raw_path),
        "-ac", "1",
        "-ar", "16000",
        str(wav_path),
    ]
    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not wav_path.exists():
        err_msg = proc.stderr.strip() or proc.stdout.strip() or "ffmpeg exited with error"
        return wav_path, True, f"ffmpeg failed (code={proc.returncode}): {err_msg}"
    return wav_path, True, None


def _write_wav(path: pathlib.Path, samples: np.ndarray, sr: int) -> None:
    """モノラルfloat波形(-1..1)を16bit PCMのWAVとして保存する。"""
    data = np.asarray(samples, dtype=np.float32)
    if data.size == 0:
        raise ValueError("empty audio")
    data = np.clip(data, -1.0, 1.0)
    pcm16 = (data * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(pcm16.tobytes())


# ========= フロント用 HTML を返す =========
@app.get("/")
def index():
    """
    ルートアクセスで ui/index.html をそのまま返す。
    uvicorn をプロジェクトルートで起動している前提:
      python -m uvicorn src.app.main:app --reload
    """
    html_path = pathlib.Path("ui") / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


# ========= テキストから話題＋質問文を生成する /text_flow =========

class TextFlowRequest(BaseModel):
    text: str
    prefs: Optional[dict] = None


@app.post("/text_flow")
async def text_flow_endpoint(req: TextFlowRequest):
    """
    まとめて「会話テキスト → 話題＋質問文」を生成するエンドポイント。
    """
    try:
        result = run_text_flow(req.text, req.prefs)
        return JSONResponse(result)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


# ========= ストリーミング用：チャンクから話題だけを抽出する /topics_incremental =========

class TopicsIncrementalRequest(BaseModel):
    text: str
    topk: int = 8


@app.post("/topics_incremental")
async def topics_incremental(req: TopicsIncrementalRequest):
    """
    音声認識で得られた「部分テキスト（チャンク）」から、
    軽量に話題候補だけを抽出して返すエンドポイント。

    ★追加: 話題が抽出された瞬間に、直前区間の平均スコアを「1回だけ」割り当てるため
         scores（topics と同じ長さ）を返す
    """
    try:
        seg = pipe.cut_segment_for_new_topic()
        avg = seg.avg_score  # None のこともある

        payload_topics = []
        topics = extract_topics(req.text, topk=req.topk)
        for t in topics:
            # topics が文字列でも dict でも耐えるように
            text = t.get("text") if isinstance(t, dict) else str(t)
            payload_topics.append({
                "text": text,
                "score_avg": avg,
                "segment": {
                    "start_window": seg.start_window,
                    "end_window": seg.end_window,
                    "n": seg.n,
                }
            })

        return {"ok": True, "topics": payload_topics}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

# ========= ブラウザ計算の音声特徴量を受け取る /voice_features =========

class VoiceFeaturesRequest(BaseModel):
    """Web Audio 側で計算した音声特徴量をまとめて送るためのリクエストモデル。"""
    session_id: Optional[str] = None   # 会話セッションID（任意）
    speaker_id: Optional[str] = "mix"  # 今は "mix" 固定でもOK
    window_index: int                  # 0,1,2,... の連番
    timestamp: Optional[float] = None  # この窓の右端の時刻（秒, 任意）
    duration_sec: float                # この窓の長さ（例: 3.0 秒）

    # --- Web Audio 側で計算したメインの特徴量 ---
    pitch_var: float                   # ピッチの分散
    energy_var: float                  # エネルギー（音量）の分散
    speech_rate: float                 # 発話スピード（指標）

    # --- 追加情報（あればベター、なくてもOK） ---
    voiced_ratio: Optional[float] = None
    pitch_mean: Optional[float] = None
    energy_mean: Optional[float] = None


@app.post("/voice_features")
async def voice_features(req: VoiceFeaturesRequest) -> dict:
    """Web Audio 側で計算した特徴量を受け取り、盛り上がり度スコアを返すエンドポイント。

    - /audio_chunk: 音声ファイルを送ってサーバ側で特徴量計算
    - /voice_features: ブラウザで特徴量を計算して JSON だけ送る（軽量）

    ★追加: live score を pipeline 側に保存し、区間平均に使う
    """
    try:
        # 必須3特徴量
        feats = {
            "pitch_var": float(req.pitch_var),
            "energy_var": float(req.energy_var),
            "speech_rate": float(req.speech_rate),
        }

        # 追加特徴量（来ていれば Runtime に渡す）
        if req.pitch_mean is not None:
            feats["pitch_mean"] = float(req.pitch_mean)
        if req.energy_mean is not None:
            feats["energy_mean"] = float(req.energy_mean)
        if req.voiced_ratio is not None:
            feats["voiced_ratio"] = float(req.voiced_ratio)

        score, phase = voice_runtime_live.update(feats)

        # タイムスタンプ決定（なければサーバ時刻）
        ts = float(req.timestamp) if req.timestamp is not None else time.time()

        # window_index==0 を「新しい音声開始」とみなして区間起点をリセット
        if int(req.window_index) == 0:
            pipe.reset_interval_cursor(now_ts=ts)

        # pipeline に live voice を渡す（tick の live優先にも効く）
        pipe.on_live_voice(
            score=None if score is None else float(score),
            phase=str(phase),
            window_index=int(req.window_index),
            timestamp=ts,
        )

        # 区間平均用の履歴に push（run で score が出た時だけ）
        if phase == "run" and score is not None:
            pipe.push_live_score(ts=ts, score=float(score))

        baseline = {
            "pitch_mu": voice_runtime_live.pitch_mu,
            "energy_mu": voice_runtime_live.energy_mu,
            "rate_mu": voice_runtime_live.rate_mu,
            "pitch_mean_mu": getattr(voice_runtime_live, "pitch_mean_mu", None),
            "energy_mean_mu": getattr(voice_runtime_live, "energy_mean_mu", None),
            "voiced_mu": getattr(voice_runtime_live, "voiced_mu", None),
        }

        pipe.record_voice_score(window_index=req.window_index, score=score)

        return {
            "ok": True,
            "session_id": req.session_id,
            "speaker_id": req.speaker_id,
            "window_index": req.window_index,
            "phase": phase,        # "calib" or "run"
            "score": score,        # run フェーズなら 0〜1, calib 中は None
            "baseline": baseline,
            "features": {
                "pitch_var": req.pitch_var,
                "energy_var": req.energy_var,
                "speech_rate": req.speech_rate,
                "voiced_ratio": req.voiced_ratio,
                "pitch_mean": req.pitch_mean,
                "energy_mean": req.energy_mean,
            },
        }
    except Exception as e:
        traceback.print_exc()
        return {"ok": False, "error": str(e)}


# ========= 音声ファイル → 音声特徴量ベース盛り上がり度 /audio_chunk =========

@app.post("/audio_chunk")
async def audio_chunk(file: UploadFile = File(...)) -> dict:
    """
    任意の音声ファイルを受け取って、
    features_audio.py で特徴量を計算し、
    VoiceEngagementRuntime で「盛り上がり度スコア」を返すエンドポイント。
    """
    tmp_dir = pathlib.Path("tmp_audio")
    tmp_dir.mkdir(exist_ok=True)

    original_name = file.filename or "chunk.webm"
    suffix = pathlib.Path(original_name).suffix.lower() or ".webm"

    raw_path = tmp_dir / f"chunk_{int(time.time() * 1000)}{suffix}"

    max_mb = int(os.getenv("AUDIO_CHUNK_MAX_MB", "20") or "20")
    max_bytes = max_mb * 1024 * 1024
    try:
        total = await _save_upload_to_path(file, raw_path, max_bytes)
    except ValueError:
        return {"ok": False, "error": f"uploaded file is too large (>{max_mb}MB)"}

    if total == 0:
        try:
            raw_path.unlink()
        except Exception:
            pass
        return {"ok": False, "error": "uploaded file is empty"}

    wav_path = raw_path
    need_delete_wav = False

    try:
        wav_path, need_delete_wav, err = _convert_to_wav(raw_path, suffix)
        if err:
            return {"ok": False, "error": err}

        feats = extract_voice_features(str(wav_path))
        score, phase = voice_runtime_api.update(feats)

        return {
            "ok": True,
            "phase": phase,
            "score": score,
            "features": feats,
        }

    except Exception as e:
        traceback.print_exc()
        return {"ok": False, "error": str(e)}

    finally:
        try:
            raw_path.unlink()
        except Exception:
            pass
        if need_delete_wav:
            try:
                wav_path.unlink()
            except Exception:
                pass


# ========= 音声ファイル → 話者分離 /diarize_chunk =========

@app.post("/diarize_chunk")
async def diarize_chunk(file: UploadFile = File(...), offset_sec: float = 0.0) -> dict:
    """
    音声ファイルを受け取り、話者ラベル付き区間を返すエンドポイント。
    """
    tmp_dir = pathlib.Path("tmp_audio")
    tmp_dir.mkdir(exist_ok=True)

    original_name = file.filename or "chunk.webm"
    suffix = pathlib.Path(original_name).suffix.lower() or ".webm"
    raw_path = tmp_dir / f"diarize_{int(time.time() * 1000)}{suffix}"

    max_mb = int(os.getenv("AUDIO_CHUNK_MAX_MB", "20") or "20")
    max_bytes = max_mb * 1024 * 1024

    try:
        total = await _save_upload_to_path(file, raw_path, max_bytes)
    except ValueError:
        return {"ok": False, "error": f"uploaded file is too large (>{max_mb}MB)"}

    if total == 0:
        try:
            raw_path.unlink()
        except Exception:
            pass
        return {"ok": False, "error": "uploaded file is empty"}

    wav_path = raw_path
    need_delete_wav = False

    try:
        wav_path, need_delete_wav, err = _convert_to_wav(raw_path, suffix)
        if err:
            return {"ok": False, "error": err}

        try:
            print(f"[DIARIZE] file={wav_path} offset={offset_sec} min={DIARIZATION_MIN_SPEAKERS} max={DIARIZATION_MAX_SPEAKERS}")
            segments = diarizer.diarize_file(
                str(wav_path),
                num_speakers=None,
                min_speakers=DIARIZATION_MIN_SPEAKERS,
                max_speakers=DIARIZATION_MAX_SPEAKERS,
            )
        except TypeError:
            # 古いpyannote向けのフォールバック
            segments = diarizer.diarize_file(str(wav_path), num_speakers=DIARIZATION_MAX_SPEAKERS)
        segments_payload = [
            {
                "start": float(seg.start) + float(offset_sec),
                "end": float(seg.end) + float(offset_sec),
                "speaker": seg.speaker,
            }
            for seg in segments
            if (float(seg.end) - float(seg.start)) >= DIARIZATION_MIN_SEGMENT_SEC
        ]
        filtered_segments = [
            seg for seg in segments
            if (float(seg.end) - float(seg.start)) >= DIARIZATION_MIN_SEGMENT_SEC
        ]
        seg_features = extract_features_for_segments(
            str(wav_path),
            filtered_segments,
            min_duration_sec=DIARIZATION_MIN_SEGMENT_SEC,
        )
        tone_change_by_speaker: Dict[str, List[float]] = {}
        features_by_speaker: Dict[str, Dict[str, List[float]]] = {}

        # 話者交代数（このチャンク内）
        turn_changes_chunk = 0
        last_speaker: Optional[str] = None
        for s in segments_payload:
            spk = s["speaker"]
            if last_speaker is not None and spk != last_speaker:
                turn_changes_chunk += 1
            last_speaker = spk

        last_features_by_speaker: Dict[str, Dict[str, float]] = {}
        for sf in seg_features:
            spk = sf["speaker"]
            feats = sf["features"]
            prev = last_features_by_speaker.get(spk)
            if prev is not None:
                delta = (
                    abs(feats.get("pitch_var", 0.0) - prev.get("pitch_var", 0.0))
                    + abs(feats.get("energy_var", 0.0) - prev.get("energy_var", 0.0))
                    + abs(feats.get("speech_rate", 0.0) - prev.get("speech_rate", 0.0))
                )
                tone_change_by_speaker.setdefault(spk, []).append(delta)
            last_features_by_speaker[spk] = feats

            feats_store = features_by_speaker.setdefault(spk, {})
            for k, v in feats.items():
                feats_store.setdefault(k, []).append(float(v))

        tone_change_avg = {
            spk: (sum(vals) / len(vals)) if vals else 0.0
            for spk, vals in tone_change_by_speaker.items()
        }
        features_avg = {
            spk: {k: (sum(vs) / len(vs)) if vs else 0.0 for k, vs in feats.items()}
            for spk, feats in features_by_speaker.items()
        }

        return {
            "ok": True,
            "offset_sec": float(offset_sec),
            "segments": segments_payload,
            "metrics": {
                "turn_changes_chunk": turn_changes_chunk,
                "tone_change_by_speaker": tone_change_avg,
                "features_by_speaker": features_avg,
            },
        }

    except DiarizationError as e:
        return {"ok": False, "error": str(e)}
    except Exception as e:
        traceback.print_exc()
        return {"ok": False, "error": str(e)}
    finally:
        try:
            raw_path.unlink()
        except Exception:
            pass
        if need_delete_wav:
            try:
                wav_path.unlink()
            except Exception:
                pass


# ========= WebSocket: 話者分離（チャンク単位） =========

@app.websocket("/ws_diarize")
async def ws_diarize(ws: WebSocket):
    await ws.accept()
    await ws.send_json({"type": "ready"})

    # NOTE: ASRは使わない方針のため、ここでは話者分離＋特徴量のみ返す

    # 話者の状態（接続ごと）
    last_speaker: Optional[str] = None
    turn_changes_total = 0
    last_features_by_speaker: Dict[str, Dict[str, float]] = {}
    prev_tail_samples: Optional[np.ndarray] = None
    prev_tail_sr = 16000
    speaker_profiles: Dict[str, Dict[str, object]] = {}
    next_speaker_id = 1

    def _new_speaker_id() -> str:
        nonlocal next_speaker_id
        label = f"SPK_{next_speaker_id:02d}"
        next_speaker_id += 1
        return label

    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        if a.size == 0 or b.size == 0:
            return -1.0
        return float(np.dot(a, b))

    def _update_profile(gid: str, emb: np.ndarray) -> None:
        prof = speaker_profiles.get(gid)
        if prof is None:
            speaker_profiles[gid] = {"embedding": emb, "n": 1}
            return
        n = int(prof.get("n", 1))
        prev = prof.get("embedding")
        if isinstance(prev, np.ndarray) and prev.size == emb.size:
            new_emb = (prev * n + emb) / float(n + 1)
            norm = float(np.linalg.norm(new_emb))
            if norm > 0:
                new_emb = new_emb / norm
            prof["embedding"] = new_emb
            prof["n"] = n + 1
        else:
            prof["embedding"] = emb
            prof["n"] = n + 1

    def _match_speakers(local_embeddings: Dict[str, np.ndarray]) -> Dict[str, str]:
        """ローカル話者 -> グローバル話者ID の対応付け。"""
        mapping: Dict[str, str] = {}
        if not local_embeddings:
            return mapping

        global_labels = list(speaker_profiles.keys())
        local_labels = list(local_embeddings.keys())

        if not global_labels:
            for lab in local_labels:
                if len(speaker_profiles) < DIARIZATION_MAX_SPEAKERS:
                    gid = _new_speaker_id()
                    mapping[lab] = gid
            return mapping

        # 1:1
        if len(local_labels) == 1:
            lab = local_labels[0]
            emb = local_embeddings[lab]
            best_gid = None
            best_sim = -1.0
            for gid in global_labels:
                sim = _cosine(emb, speaker_profiles[gid]["embedding"])  # type: ignore
                if sim > best_sim:
                    best_sim = sim
                    best_gid = gid
            if best_gid is not None:
                if best_sim < SPEAKER_EMBEDDING_MATCH_THRESHOLD and len(speaker_profiles) < DIARIZATION_MAX_SPEAKERS:
                    best_gid = _new_speaker_id()
                mapping[lab] = best_gid
            return mapping

        # 2:1 or 2:2
        if len(local_labels) == 2:
            l0, l1 = local_labels
            e0, e1 = local_embeddings[l0], local_embeddings[l1]
            if len(global_labels) == 1:
                gid0 = global_labels[0]
                s0 = _cosine(e0, speaker_profiles[gid0]["embedding"])  # type: ignore
                s1 = _cosine(e1, speaker_profiles[gid0]["embedding"])  # type: ignore
                if s0 >= s1:
                    mapping[l0] = gid0
                    if len(speaker_profiles) < DIARIZATION_MAX_SPEAKERS:
                        gid1 = _new_speaker_id()
                        mapping[l1] = gid1
                    else:
                        mapping[l1] = gid0
                else:
                    mapping[l1] = gid0
                    if len(speaker_profiles) < DIARIZATION_MAX_SPEAKERS:
                        gid1 = _new_speaker_id()
                        mapping[l0] = gid1
                    else:
                        mapping[l0] = gid0
                return mapping

            if len(global_labels) >= 2:
                g0, g1 = global_labels[0], global_labels[1]
                s00 = _cosine(e0, speaker_profiles[g0]["embedding"])  # type: ignore
                s01 = _cosine(e0, speaker_profiles[g1]["embedding"])  # type: ignore
                s10 = _cosine(e1, speaker_profiles[g0]["embedding"])  # type: ignore
                s11 = _cosine(e1, speaker_profiles[g1]["embedding"])  # type: ignore
                if (s00 + s11) >= (s01 + s10):
                    mapping[l0] = g0
                    mapping[l1] = g1
                else:
                    mapping[l0] = g1
                    mapping[l1] = g0
                return mapping

        # その他のケースはベストマッチ優先
        for lab in local_labels:
            emb = local_embeddings[lab]
            best_gid = None
            best_sim = -1.0
            for gid in global_labels:
                sim = _cosine(emb, speaker_profiles[gid]["embedding"])  # type: ignore
                if sim > best_sim:
                    best_sim = sim
                    best_gid = gid
            if best_gid is not None:
                mapping[lab] = best_gid
        return mapping

    try:
        while True:
            try:
                data = await ws.receive_json()
            except WebSocketDisconnect:
                break
            except Exception:
                await ws.send_json({"type": "error", "error": "invalid json"})
                continue

            if data == "PING" or data.get("type") == "ping":
                await ws.send_text("PONG")
                continue

            if data.get("type") != "chunk":
                await ws.send_json({"type": "error", "error": "unknown message type"})
                continue

            b64 = data.get("audio_base64")
            chunk_id = data.get("chunk_id")
            if not b64:
                await ws.send_json({"type": "error", "error": "audio_base64 is required"})
                continue

            suffix = (data.get("suffix") or ".webm").lower()
            offset_sec = float(data.get("offset_sec") or 0.0)

            tmp_dir = pathlib.Path("tmp_audio")
            tmp_dir.mkdir(exist_ok=True)
            raw_path = tmp_dir / f"ws_diarize_{int(time.time() * 1000)}{suffix}"

            try:
                audio_bytes = base64.b64decode(b64, validate=True)
            except Exception:
                await ws.send_json({"type": "error", "error": "invalid base64 audio"})
                continue

            max_mb = int(os.getenv("AUDIO_CHUNK_MAX_MB", "20") or "20")
            max_bytes = max_mb * 1024 * 1024
            if len(audio_bytes) > max_bytes:
                await ws.send_json({"type": "error", "error": f"audio chunk is too large (>{max_mb}MB)"})
                continue

            print(f"[DIARIZE] recv chunk_id={chunk_id} bytes={len(audio_bytes)} offset={offset_sec}")
            raw_path.write_bytes(audio_bytes)

            wav_path = raw_path
            need_delete_wav = False
            combined_path: Optional[pathlib.Path] = None
            cleanup_combined = False

            cleanup_in_finally = True
            try:
                wav_path, need_delete_wav, err = _convert_to_wav(raw_path, suffix)
                if err:
                    await ws.send_json({"type": "error", "error": err})
                    continue

                # オーバーラップ用に現在チャンクを読み込み（前チャンク末尾を左文脈として付与）
                current_audio = load_mono_wav(str(wav_path), target_sr=prev_tail_sr)
                chunk_duration = current_audio.duration
                context_sec = 0.0
                combined_path = wav_path
                combined_samples = current_audio.samples

                if DIARIZATION_OVERLAP_SEC > 0 and prev_tail_samples is not None and prev_tail_samples.size > 0:
                    context_sec = min(DIARIZATION_OVERLAP_SEC, float(prev_tail_samples.size) / float(prev_tail_sr))
                    combined_samples = np.concatenate([prev_tail_samples, current_audio.samples])
                    combined_path = tmp_dir / f"ws_diarize_ctx_{int(time.time() * 1000)}.wav"
                    _write_wav(combined_path, combined_samples, current_audio.sr)
                    cleanup_combined = True

                tail_len = int(DIARIZATION_OVERLAP_SEC * current_audio.sr)
                if tail_len > 0:
                    if current_audio.samples.size >= tail_len:
                        prev_tail_samples = current_audio.samples[-tail_len:]
                    else:
                        prev_tail_samples = current_audio.samples
                    prev_tail_sr = current_audio.sr
                else:
                    prev_tail_samples = None

                try:
                    t0 = time.monotonic()
                    print(f"[DIARIZE] ws file={combined_path} offset={offset_sec} min={DIARIZATION_MIN_SPEAKERS} max={DIARIZATION_MAX_SPEAKERS} chunk_id={chunk_id} ctx={context_sec}")
                    segments = diarizer.diarize_file(
                        str(combined_path),
                        num_speakers=None,
                        min_speakers=DIARIZATION_MIN_SPEAKERS,
                        max_speakers=DIARIZATION_MAX_SPEAKERS,
                    )
                    t1 = time.monotonic()
                except TypeError:
                    t0 = time.monotonic()
                    segments = diarizer.diarize_file(str(combined_path), num_speakers=DIARIZATION_MAX_SPEAKERS)
                    t1 = time.monotonic()

                # 右端（現在チャンク分）だけを有効区間として採用
                valid_start = context_sec
                valid_end = context_sec + chunk_duration
                trimmed_segments: List[SimpleNamespace] = []
                payload: List[Dict[str, object]] = []
                for seg in segments:
                    start = float(seg.start)
                    end = float(seg.end)
                    speaker = seg.speaker
                    if end <= valid_start or start >= valid_end:
                        continue
                    t_start = max(start, valid_start)
                    t_end = min(end, valid_end)
                    if (t_end - t_start) < DIARIZATION_MIN_SEGMENT_SEC:
                        continue
                    trimmed_segments.append(SimpleNamespace(start=t_start, end=t_end, speaker=speaker))
                    payload.append({
                        "start": (t_start - valid_start) + float(offset_sec),
                        "end": (t_end - valid_start) + float(offset_sec),
                        "speaker": speaker,
                    })

                # 話者埋め込みでラベルを統一（任意）
                speaker_label_map: Dict[str, str] = {}
                if SPEAKER_EMBEDDING_ENABLED and trimmed_segments:
                    try:
                        by_spk: Dict[str, List[np.ndarray]] = {}
                        for seg in trimmed_segments:
                            s = int(seg.start * current_audio.sr)
                            e = int(seg.end * current_audio.sr)
                            if e <= s:
                                continue
                            by_spk.setdefault(seg.speaker, []).append(combined_samples[s:e])

                        local_embeddings: Dict[str, np.ndarray] = {}
                        for spk, chunks in by_spk.items():
                            if not chunks:
                                continue
                            samples = np.concatenate(chunks)
                            if samples.size == 0:
                                continue
                            emb = speaker_embedder.embed(samples, current_audio.sr)
                            local_embeddings[spk] = emb

                        speaker_label_map = _match_speakers(local_embeddings)
                        for spk, emb in local_embeddings.items():
                            gid = speaker_label_map.get(spk)
                            if gid:
                                _update_profile(gid, emb)
                    except SpeakerEmbeddingError as e:
                        print(f"[DIARIZE] speaker embedding error: {e}")
                    except Exception as e:
                        print(f"[DIARIZE] speaker embedding error: {e}")

                if speaker_label_map:
                    for seg in payload:
                        seg["speaker"] = speaker_label_map.get(seg["speaker"], seg["speaker"])
                    for seg in trimmed_segments:
                        seg.speaker = speaker_label_map.get(seg.speaker, seg.speaker)

                # セグメントごとの特徴量（話者別）
                seg_features = extract_features_for_segments(
                    str(combined_path),
                    trimmed_segments,
                    min_duration_sec=DIARIZATION_MIN_SEGMENT_SEC,
                )
                tone_change_by_speaker: Dict[str, List[float]] = {}
                features_by_speaker: Dict[str, Dict[str, List[float]]] = {}

                # 話者交代数（このチャンク内 + 累計）
                turn_changes_chunk = 0
                for s in payload:
                    spk = s["speaker"]
                    if last_speaker is not None and spk != last_speaker:
                        turn_changes_chunk += 1
                        turn_changes_total += 1
                    last_speaker = spk

                for sf in seg_features:
                    spk = sf["speaker"]
                    feats = sf["features"]
                    prev = last_features_by_speaker.get(spk)
                    if prev is not None:
                        delta = (
                            abs(feats.get("pitch_var", 0.0) - prev.get("pitch_var", 0.0))
                            + abs(feats.get("energy_var", 0.0) - prev.get("energy_var", 0.0))
                            + abs(feats.get("speech_rate", 0.0) - prev.get("speech_rate", 0.0))
                        )
                        tone_change_by_speaker.setdefault(spk, []).append(delta)
                    last_features_by_speaker[spk] = feats

                    feats_store = features_by_speaker.setdefault(spk, {})
                    for k, v in feats.items():
                        feats_store.setdefault(k, []).append(float(v))

                tone_change_avg = {
                    spk: (sum(vals) / len(vals)) if vals else 0.0
                    for spk, vals in tone_change_by_speaker.items()
                }
                features_avg = {
                    spk: {k: (sum(vs) / len(vs)) if vs else 0.0 for k, vs in feats.items()}
                    for spk, feats in features_by_speaker.items()
                }
                await ws.send_json({
                    "type": "segments",
                    "payload": {
                        "offset_sec": float(offset_sec),
                        "segments": payload,
                        "chunk_id": chunk_id,
                        "durations": {
                            "diarize_sec": round(t1 - t0, 3),
                        },
                        "metrics": {
                            "turn_changes_chunk": turn_changes_chunk,
                            "turn_changes_total": turn_changes_total,
                            "tone_change_by_speaker": tone_change_avg,
                            "features_by_speaker": features_avg,
                        },
                    },
                })
                print(f"[DIARIZE] send chunk_id={chunk_id} segments={len(payload)}")
                cleanup_in_finally = True

            except DiarizationError as e:
                await ws.send_json({"type": "error", "error": str(e)})
            except Exception as e:
                traceback.print_exc()
                await ws.send_json({"type": "error", "error": str(e)})
            finally:
                if cleanup_in_finally:
                    try:
                        raw_path.unlink()
                    except Exception:
                        pass
                    if need_delete_wav:
                        try:
                            wav_path.unlink()
                        except Exception:
                            pass
                    if cleanup_combined and combined_path is not None and combined_path != wav_path:
                        try:
                            combined_path.unlink()
                        except Exception:
                            pass

    finally:
        try:
            await ws.close()
        except Exception:
            pass


# ========= WebSocket: 盛り上がり度（暫定：tick）用 =========

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    await ws.send_text("connected")

    try:
        while True:
            try:
                data = await ws.receive_json()
            except WebSocketDisconnect:
                print("connection closed (WebSocketDisconnect)")
                break

            if data == "PING":
                await ws.send_text("PONG")
                continue

            mtype = data.get("type")
            if mtype == "tick":
                stats = pipe.tick()
                await ws.send_json({"type": "stats", "payload": stats})

                # generate_questions が無い/未実装でも落とさない（安全）
                if getattr(stats.get("intervene", {}), "get", None):
                    trigger = bool(stats["intervene"].get("trigger", False))
                else:
                    trigger = bool(stats.get("intervene", {}).get("trigger", False))

                if trigger and hasattr(pipe, "generate_questions"):
                    try:
                        payload = pipe.generate_questions()  # type: ignore
                        await ws.send_json({"type": "questions", "payload": payload})
                    except Exception:
                        traceback.print_exc()

    finally:
        try:
            await ws.close()
        except Exception:
            pass
