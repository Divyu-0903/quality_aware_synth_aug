from __future__ import annotations

import argparse
from pathlib import Path

import evaluate
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.utils.config import load_config
from src.utils.io import load_jsonl
from src.utils.seed import set_seed


LABEL2ID = {"negative": 0, "positive": 1}
ID2LABEL = {0: "negative", 1: "positive"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant", choices=["baseline", "naive", "filtered"], required=True
    )
    return parser.parse_args()


def load_sent_dataset(path: str) -> Dataset:
    records = load_jsonl(path)
    clean = [{"text": r["text"], "label": LABEL2ID[r["label"]]} for r in records]
    return Dataset.from_list(clean)


def main():
    args = parse_args()
    cfg = load_config()
    set_seed(cfg["project"]["seed"])

    train_path = f"data/augmented/sentiment/{args.variant}.jsonl"
    val_path = "data/raw/sentiment/validation.jsonl"

    train_ds = load_sent_dataset(train_path)
    val_ds = load_sent_dataset(val_path)

    model_name = cfg["training"]["sentiment_model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=cfg["training"]["max_length"],
        )

    tokenized_train = train_ds.map(tokenize_fn, batched=True)
    tokenized_val = val_ds.map(tokenize_fn, batched=True)

    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        return {
            "accuracy": accuracy.compute(predictions=preds, references=labels)[
                "accuracy"
            ],
            "f1": f1.compute(predictions=preds, references=labels, average="binary")[
                "f1"
            ],
        }

    out_dir = Path(cfg["training"]["output_dir"]) / "sentiment" / args.variant

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
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)


if __name__ == "__main__":
    main()
