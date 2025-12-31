# src/audio/asr_whisper.py
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional
import os


@dataclass
class ASRSegment:
    start: float
    end: float
    text: str


@lru_cache(maxsize=2)
def _get_model(model_size: str, device: str, compute_type: str):
    from faster_whisper import WhisperModel  # type: ignore
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def transcribe_segments(wav_path: str) -> List[ASRSegment]:
    model_size = (os.getenv("ASR_MODEL_SIZE") or "small").strip()
    device = (os.getenv("ASR_DEVICE") or "cpu").strip()
    compute_type = (os.getenv("ASR_COMPUTE_TYPE") or "int8").strip()
    language = (os.getenv("ASR_LANGUAGE") or "").strip() or None

    model = _get_model(model_size, device, compute_type)
    print(f"[ASR] start transcribe path={wav_path} model={model_size} device={device} compute_type={compute_type} lang={language}")
    segments, _info = model.transcribe(
        wav_path,
        beam_size=1,
        vad_filter=True,
        language=language,
    )

    results: List[ASRSegment] = []
    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        results.append(ASRSegment(start=float(seg.start), end=float(seg.end), text=text))
    print(f"[ASR] segments={len(results)}")
    return results


def assign_speakers(
    asr_segments: List[ASRSegment],
    diarization_segments: List[dict],
    unknown_speaker: str = "SPEAKER_00",
) -> List[dict]:
    """ASR セグメントに話者を割り当てる（最大重なり）。"""
    print(f"[ASR] assign speakers: asr={len(asr_segments)} diar={len(diarization_segments)}")
    assigned: List[dict] = []
    for a in asr_segments:
        best_speaker = None
        best_overlap = 0.0
        for d in diarization_segments:
            ds = float(d.get("start", 0.0))
            de = float(d.get("end", 0.0))
            overlap = min(a.end, de) - max(a.start, ds)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = d.get("speaker")
        assigned.append({
            "start": a.start,
            "end": a.end,
            "speaker": best_speaker or unknown_speaker,
            "text": a.text,
        })
    merged = _merge_utterances(assigned)
    print(f"[ASR] merged utterances={len(merged)}")
    return merged


def _merge_utterances(items: List[dict], gap_sec: float = 0.6) -> List[dict]:
    """同一話者で近接する発話を結合する。"""
    if not items:
        return []
    items = sorted(items, key=lambda x: (x.get("start", 0.0), x.get("end", 0.0)))
    merged: List[dict] = [dict(items[0])]
    for cur in items[1:]:
        prev = merged[-1]
        if (
            cur.get("speaker") == prev.get("speaker")
            and float(cur.get("start", 0.0)) - float(prev.get("end", 0.0)) <= gap_sec
        ):
            prev["end"] = cur.get("end")
            prev["text"] = (prev.get("text", "") + " " + cur.get("text", "")).strip()
        else:
            merged.append(dict(cur))
    return merged
