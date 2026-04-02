from __future__ import annotations

import re
from typing import Any

from src.filtering.common import (
    normalize_text,
    punctuation_sanity_score,
    token_diversity,
    length_score,
    jaccard_similarity,
)

PERSON_TITLES = {"dr", "mr", "mrs", "ms", "prof"}


def reconstruct_text(tokens: list[str]) -> str:
    text = " ".join(tokens)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    return normalize_text(text)


def is_valid_bio(tags: list[str]) -> bool:
    prev_type = None
    for tag in tags:
        if tag == "O":
            prev_type = None
            continue

        if not (tag.startswith("B-") or tag.startswith("I-")):
            return False

        cur_type = tag[2:]
        if tag.startswith("I-") and prev_type != cur_type:
            return False

        prev_type = cur_type
    return True


def extract_entities(tokens: list[str], ner_tags: list[str]) -> list[tuple[str, str]]:
    entities = []
    current_tokens = []
    current_type = None

    for tok, tag in zip(tokens, ner_tags):
        if tag == "O":
            if current_tokens:
                entities.append((" ".join(current_tokens), current_type))
                current_tokens = []
                current_type = None
            continue

        prefix, ent_type = tag.split("-", 1)
        if prefix == "B":
            if current_tokens:
                entities.append((" ".join(current_tokens), current_type))
            current_tokens = [tok]
            current_type = ent_type
        elif prefix == "I":
            if current_tokens and current_type == ent_type:
                current_tokens.append(tok)
            else:
                return []

    if current_tokens:
        entities.append((" ".join(current_tokens), current_type))

    return entities


def fluency_score_ner(tokens: list[str]) -> float:
    text = reconstruct_text(tokens)
    return (
        0.5 * length_score(len(tokens), 8, 20)
        + 0.3 * punctuation_sanity_score(text)
        + 0.2 * token_diversity(tokens)
    )


def label_consistency_score_ner(tokens: list[str], ner_tags: list[str]) -> float:
    if len(tokens) != len(ner_tags) or not tokens:
        return 0.0

    score = 1.0

    if not is_valid_bio(ner_tags):
        return 0.0

    entities = extract_entities(tokens, ner_tags)
    if len(entities) < 2:
        score -= 0.25

    for entity_text, entity_type in entities:
        words = entity_text.split()
        if entity_type == "PER" and words:
            first = words[0].lower().rstrip(".")
            if first in PERSON_TITLES:
                score -= 0.20

        if entity_type in {"ORG", "LOC", "MISC", "PER"} and len(words) == 0:
            score -= 0.30

    if all(tag == "O" for tag in ner_tags):
        return 0.0

    return max(0.0, min(score, 1.0))


def compute_ner_scores(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    texts = [reconstruct_text(r["tokens"]) for r in records]

    scored = []
    for i, rec in enumerate(records):
        tokens = rec["tokens"]
        ner_tags = rec["ner_tags"]

        fluency = fluency_score_ner(tokens)
        consistency = label_consistency_score_ner(tokens, ner_tags)

        max_sim = 0.0
        for j, other in enumerate(records):
            if i == j:
                continue
            sim = jaccard_similarity(tokens, other["tokens"])
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
