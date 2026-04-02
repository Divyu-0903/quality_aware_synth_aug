from __future__ import annotations

import re
from typing import List, Tuple


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")


def simple_tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text)


def find_sublist_span(
    tokens: list[str], phrase_tokens: list[str]
) -> tuple[int, int] | None:
    if not phrase_tokens or len(phrase_tokens) > len(tokens):
        return None

    for i in range(len(tokens) - len(phrase_tokens) + 1):
        if tokens[i : i + len(phrase_tokens)] == phrase_tokens:
            return i, i + len(phrase_tokens) - 1
    return None


def apply_bio_tags(tokens: list[str], entity_spans: list[tuple[str, str]]) -> list[str]:
    """
    entity_spans: list of (entity_text, entity_label)
    """
    ner_tags = ["O"] * len(tokens)

    for entity_text, entity_label in entity_spans:
        ent_tokens = simple_tokenize(entity_text)
        span = find_sublist_span(tokens, ent_tokens)

        if span is None:
            raise ValueError(f"Entity '{entity_text}' not found in tokens")

        start, end = span
        ner_tags[start] = f"B-{entity_label}"
        for idx in range(start + 1, end + 1):
            ner_tags[idx] = f"I-{entity_label}"

    return ner_tags
