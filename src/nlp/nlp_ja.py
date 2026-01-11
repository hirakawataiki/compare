#Sudachi/GiNZA で形態素・固有表現抽出

# src/nlp/nlp_ja.py
from __future__ import annotations
from collections import Counter
from typing import List, Tuple
from sudachipy import tokenizer as sudachi_tokenizer
from sudachipy import dictionary as sudachi_dictionary

_tok = sudachi_dictionary.Dictionary().create()
_MODE = sudachi_tokenizer.Tokenizer.SplitMode.C

def analyze(text: str) -> List[Tuple[str, str, Tuple[str, ...]]]:
    res = []
    for m in _tok.tokenize(text or "", _MODE):
        pos = tuple(m.part_of_speech())
        res.append((m.surface(), m.normalized_form(), pos))
    return res

def extract_keywords(text: str, topk: int = 8) -> List[str]:
    counter = Counter()
    for surface, lemma, pos in analyze(text):
        key = lemma or surface
        if key:
            counter[key] += 1
    return [w for w, _ in counter.most_common(topk)]
