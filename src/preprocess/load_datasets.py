from __future__ import annotations

from pathlib import Path

from datasets import load_dataset

from src.utils.config import load_config
from src.utils.io import save_jsonl, ensure_dir, save_json
from src.utils.seed import set_seed


def convert_conll2003(split_data, id2label: dict[int, str]) -> list[dict]:
    records = []
    for idx, ex in enumerate(split_data):
        tokens = ex["tokens"]
        ner_tags = [id2label[tag_id] for tag_id in ex["ner_tags"]]
        records.append({"id": idx, "tokens": tokens, "ner_tags": ner_tags})
    return records


def convert_sst2(split_data) -> list[dict]:
    records = []
    for idx, ex in enumerate(split_data):
        label = "positive" if ex["label"] == 1 else "negative"
        records.append({"id": idx, "text": ex["sentence"], "label": label})
    return records


def main() -> None:
    config = load_config()
    set_seed(config["project"]["seed"])

    # ---------- NER ----------
    ner_cfg = config["datasets"]["ner"]
    ner_out_dir = ensure_dir(ner_cfg["output_dir"])

    ner_ds = load_dataset(ner_cfg["hf_name"])
    ner_features = ner_ds["train"].features["ner_tags"].feature
    id2label = {i: name for i, name in enumerate(ner_features.names)}

    ner_train = convert_conll2003(ner_ds["train"], id2label)
    ner_val = convert_conll2003(ner_ds["validation"], id2label)
    ner_test = convert_conll2003(ner_ds["test"], id2label)

    save_jsonl(ner_train, ner_out_dir / "train.jsonl")
    save_jsonl(ner_val, ner_out_dir / "validation.jsonl")
    save_jsonl(ner_test, ner_out_dir / "test.jsonl")
    save_json({"id2label": id2label}, ner_out_dir / "meta.json")

    # ---------- Sentiment ----------
    sent_cfg = config["datasets"]["sentiment"]
    sent_out_dir = ensure_dir(sent_cfg["output_dir"])

    sent_ds = load_dataset(sent_cfg["hf_name"], sent_cfg["hf_subset"])

    sent_train = convert_sst2(sent_ds["train"])
    sent_val = convert_sst2(sent_ds["validation"])

    save_jsonl(sent_train, sent_out_dir / "train.jsonl")
    save_jsonl(sent_val, sent_out_dir / "validation.jsonl")

    # SST-2 official GLUE test labels are hidden, so we store only train/validation for now
    save_json(
        {
            "labels": ["negative", "positive"],
            "note": "GLUE SST-2 test labels are not publicly available.",
        },
        sent_out_dir / "meta.json",
    )

    print("Datasets downloaded and saved successfully.")
    print(f"NER files: {ner_out_dir}")
    print(f"Sentiment files: {sent_out_dir}")


if __name__ == "__main__":
    main()
