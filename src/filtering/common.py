from __future__ import annotations

import math
import re
from collections import Counter
from difflib import SequenceMatcher


def normalize_text(text: str) -> str:
    text = text.replace("-", "-").replace("–", "-").replace("—", "-")
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def simple_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def token_diversity(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    unique = len(set(t.lower() for t in tokens))
    return unique / len(tokens)


def punctuation_sanity_score(text: str) -> float:
    bad_patterns = [
        r"\.\.",
        r",,",
        r"\s+[,.!?;:]",
    ]
    penalty = 0.0
    for pattern in bad_patterns:
        if re.search(pattern, text):
            penalty += 0.2
    return max(0.0, 1.0 - penalty)


def length_score(n: int, low: int, high: int) -> float:
    if low <= n <= high:
        return 1.0
    if n <= 0:
        return 0.0
    dist = min(abs(n - low), abs(n - high))
    return max(0.0, 1.0 - dist / max(high, 1))


def jaccard_similarity(a_tokens: list[str], b_tokens: list[str]) -> float:
    a = set(t.lower() for t in a_tokens)
    b = set(t.lower() for t in b_tokens)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
