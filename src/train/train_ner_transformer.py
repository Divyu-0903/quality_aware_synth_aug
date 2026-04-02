from __future__ import annotations

import argparse
from pathlib import Path

import evaluate
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from src.utils.config import load_config
from src.utils.io import load_jsonl
from src.utils.seed import set_seed


LABEL_LIST = [
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
    "B-MISC",
    "I-MISC",
]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant", choices=["baseline", "naive", "filtered"], required=True
    )
    return parser.parse_args()


def load_ner_dataset(path: str) -> Dataset:
    records = load_jsonl(path)
    clean = [{"tokens": r["tokens"], "ner_tags": r["ner_tags"]} for r in records]
    return Dataset.from_list(clean)


def tokenize_and_align_labels(examples, tokenizer, max_length):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=max_length,
    )

    labels = []
    for i, tags in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(LABEL2ID[tags[word_idx]])
            else:
                # ignore subword continuation
                label_ids.append(-100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized["labels"] = labels
    return tokenized


def main():
    args = parse_args()
    cfg = load_config()
    set_seed(cfg["project"]["seed"])

    train_path = f"data/augmented/ner/{args.variant}.jsonl"
    val_path = "data/raw/ner/validation.jsonl"

    train_ds = load_ner_dataset(train_path)
    val_ds = load_ner_dataset(val_path)

    model_name = cfg["training"]["ner_model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    tokenized_train = train_ds.map(
        lambda x: tokenize_and_align_labels(
            x, tokenizer, cfg["training"]["max_length"]
        ),
        batched=True,
    )
    tokenized_val = val_ds.map(
        lambda x: tokenize_and_align_labels(
            x, tokenizer, cfg["training"]["max_length"]
        ),
        batched=True,
    )

    seqeval = evaluate.load("seqeval")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        true_predictions = []
        true_labels = []

        for pred, lab in zip(predictions, labels):
            cur_preds = []
            cur_labels = []
            for p, l in zip(pred, lab):
                if l == -100:
                    continue
                cur_preds.append(ID2LABEL[p])
                cur_labels.append(ID2LABEL[l])
            true_predictions.append(cur_preds)
            true_labels.append(cur_labels)

        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    out_dir = Path(cfg["training"]["output_dir"]) / "ner" / args.variant

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=cfg["training"]["learning_rate"],
        per_device_train_batch_size=cfg["training"]["batch_size"],
        per_device_eval_batch_size=cfg["training"]["batch_size"],
        num_train_epochs=cfg["training"]["epochs"],
        weight_decay=cfg["training"]["weight_decay"],
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)


if __name__ == "__main__":
    main()
