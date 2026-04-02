from __future__ import annotations

from pathlib import Path

from src.utils.config import load_config
from src.utils.io import load_jsonl, save_jsonl, ensure_dir
from src.filtering.score_ner import compute_ner_scores
from src.filtering.score_sentiment import compute_sentiment_scores


def final_quality_score(components: dict, weights: dict) -> float:
    return (
        weights["fluency"] * components["fluency"]
        + weights["label_consistency"] * components["label_consistency"]
        + weights["diversity"] * components["diversity"]
        - weights["duplication"] * components["duplication"]
    )


def main() -> None:
    cfg = load_config()
    weights = cfg["filtering"]["weights"]

    ner_in = "data/synthetic/ner/synthetic_ner.jsonl"
    sent_in = "data/synthetic/sentiment/synthetic_sentiment.jsonl"

    ner_out_dir = ensure_dir("data/filtered/ner")
    sent_out_dir = ensure_dir("data/filtered/sentiment")

    ner_records = load_jsonl(ner_in)
    sent_records = load_jsonl(sent_in)

    ner_scored = compute_ner_scores(ner_records)
    sent_scored = compute_sentiment_scores(sent_records)

    for rec in ner_scored:
        rec["quality_score"] = round(
            final_quality_score(rec["quality_components"], weights), 4
        )

    for rec in sent_scored:
        rec["quality_score"] = round(
            final_quality_score(rec["quality_components"], weights), 4
        )

    ner_threshold = cfg["filtering"]["thresholds"]["ner_min_quality"]
    sent_threshold = cfg["filtering"]["thresholds"]["sentiment_min_quality"]

    ner_filtered = [r for r in ner_scored if r["quality_score"] >= ner_threshold]
    sent_filtered = [r for r in sent_scored if r["quality_score"] >= sent_threshold]

    ner_scored = sorted(ner_scored, key=lambda x: x["quality_score"], reverse=True)
    sent_scored = sorted(sent_scored, key=lambda x: x["quality_score"], reverse=True)

    save_jsonl(ner_scored, Path(ner_out_dir) / "synthetic_ner_scored.jsonl")
    save_jsonl(sent_scored, Path(sent_out_dir) / "synthetic_sentiment_scored.jsonl")

    save_jsonl(ner_filtered, Path(ner_out_dir) / "synthetic_ner_filtered.jsonl")
    save_jsonl(sent_filtered, Path(sent_out_dir) / "synthetic_sentiment_filtered.jsonl")

    print(f"NER total: {len(ner_records)}")
    print(f"NER kept: {len(ner_filtered)}")
    print(f"Sentiment total: {len(sent_records)}")
    print(f"Sentiment kept: {len(sent_filtered)}")


if __name__ == "__main__":
    main()
