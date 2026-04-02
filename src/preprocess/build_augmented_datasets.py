from __future__ import annotations

import random
from pathlib import Path

from src.utils.config import load_config
from src.utils.io import load_jsonl, save_jsonl, ensure_dir
from src.utils.seed import set_seed


def sample_records(records: list[dict], n: int) -> list[dict]:
    if n >= len(records):
        return records[:]
    return random.sample(records, n)


def trim_records(records: list[dict], n: int) -> list[dict]:
    return records[: min(n, len(records))]


def reset_ids(records: list[dict]) -> list[dict]:
    out = []
    for i, rec in enumerate(records):
        new_rec = dict(rec)
        new_rec["id"] = i
        out.append(new_rec)
    return out


def main() -> None:
    cfg = load_config()
    set_seed(cfg["project"]["seed"])

    # ---------- NER ----------
    ner_raw = load_jsonl("data/raw/ner/train.jsonl")
    ner_synth = load_jsonl("data/synthetic/ner/synthetic_ner.jsonl")
    ner_filtered = load_jsonl("data/filtered/ner/synthetic_ner_filtered.jsonl")

    ner_base_n = cfg["augmentation"]["ner_base_subset"]
    ner_base = sample_records(ner_raw, ner_base_n)
    ner_naive = ner_base + trim_records(ner_synth, 100)
    ner_quality = ner_base + trim_records(ner_filtered, 49)

    ner_out = ensure_dir("data/augmented/ner")
    save_jsonl(reset_ids(ner_base), Path(ner_out) / "baseline.jsonl")
    save_jsonl(reset_ids(ner_naive), Path(ner_out) / "naive.jsonl")
    save_jsonl(reset_ids(ner_quality), Path(ner_out) / "filtered.jsonl")

    # ---------- Sentiment ----------
    sent_raw = load_jsonl("data/raw/sentiment/train.jsonl")
    sent_synth = load_jsonl("data/synthetic/sentiment/synthetic_sentiment.jsonl")
    sent_filtered = load_jsonl(
        "data/filtered/sentiment/synthetic_sentiment_filtered.jsonl"
    )

    sent_base_n = cfg["augmentation"]["sentiment_base_subset"]
    sent_base = sample_records(sent_raw, sent_base_n)
    sent_naive = sent_base + trim_records(sent_synth, 100)
    sent_quality = sent_base + trim_records(sent_filtered, 61)

    sent_out = ensure_dir("data/augmented/sentiment")
    save_jsonl(reset_ids(sent_base), Path(sent_out) / "baseline.jsonl")
    save_jsonl(reset_ids(sent_naive), Path(sent_out) / "naive.jsonl")
    save_jsonl(reset_ids(sent_quality), Path(sent_out) / "filtered.jsonl")

    print("NER baseline:", len(ner_base))
    print("NER naive:", len(ner_naive))
    print("NER filtered:", len(ner_quality))
    print("Sentiment baseline:", len(sent_base))
    print("Sentiment naive:", len(sent_naive))
    print("Sentiment filtered:", len(sent_quality))


if __name__ == "__main__":
    main()
