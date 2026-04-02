from __future__ import annotations

from typing import Any

from src.filtering.common import (
    normalize_text,
    punctuation_sanity_score,
    token_diversity,
    length_score,
    simple_similarity,
)


POSITIVE_WORDS = {
    "great",
    "excellent",
    "amazing",
    "wonderful",
    "fantastic",
    "enjoyable",
    "beautiful",
    "helpful",
    "impressive",
    "outstanding",
    "smooth",
    "love",
    "loved",
    "pleasant",
    "delightful",
    "brilliant",
    "powerful",
}

NEGATIVE_WORDS = {
    "bad",
    "awful",
    "terrible",
    "boring",
    "poor",
    "disappointing",
    "waste",
    "slow",
    "broken",
    "annoying",
    "weak",
    "hate",
    "hated",
    "frustrating",
    "predictable",
    "unpleasant",
    "worst",
    "mediocre",
}


def fluency_score_sentiment(text: str) -> float:
    norm = normalize_text(text)
    tokens = norm.split()
    return (
        0.5 * length_score(len(tokens), 8, 20)
        + 0.3 * punctuation_sanity_score(norm)
        + 0.2 * token_diversity(tokens)
    )


def label_consistency_score_sentiment(text: str, label: str) -> float:
    t = normalize_text(text).lower()
    pos_hits = sum(1 for w in POSITIVE_WORDS if w in t)
    neg_hits = sum(1 for w in NEGATIVE_WORDS if w in t)

    if label == "positive":
        if pos_hits > neg_hits:
            return 1.0
        if pos_hits == neg_hits:
            return 0.5
        return 0.0

    if label == "negative":
        if neg_hits > pos_hits:
            return 1.0
        if pos_hits == neg_hits:
            return 0.5
        return 0.0

    return 0.0


def compute_sentiment_scores(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored = []

    for i, rec in enumerate(records):
        text = rec["text"]
        label = rec["label"]

        fluency = fluency_score_sentiment(text)
        consistency = label_consistency_score_sentiment(text, label)

        max_sim = 0.0
        for j, other in enumerate(records):
            if i == j:
                continue
            sim = simple_similarity(text, other["text"])
            if sim > max_sim:
                max_sim = sim

        duplication_penalty = max_sim
        diversity = 1.0 - max_sim

        scored.append(
            {
                **rec,
                "quality_components": {
                    "fluency": round(fluency, 4),
                    "label_consistency": round(consistency, 4),
                    "diversity": round(diversity, 4),
                    "duplication": round(duplication_penalty, 4),
                },
            }
        )

    return scored
