from __future__ import annotations

import json
import os
import re
from tqdm import tqdm

from src.generation.llm_client import LLMClient
from src.generation.prompts import sentiment_prompt
from src.utils.config import load_config
from src.utils.io import save_jsonl, load_jsonl, ensure_dir


def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


def main():
    cfg = load_config()

    client = LLMClient(cfg["llm"]["model_name"], cfg["llm"]["max_retries"])

    ensure_dir("data/synthetic/sentiment")
    output_path = "data/synthetic/sentiment/synthetic_sentiment.jsonl"

    data = []
    if os.path.exists(output_path):
        data = load_jsonl(output_path)

    existing = len(data)
    target_n = cfg["synthetic"]["sentiment_samples"]

    print(f"Resuming from {existing} / {target_n}")

    for _ in tqdm(range(existing, target_n)):
        system, user, _ = sentiment_prompt()

        try:
            raw = client.generate(system, user, cfg["llm"]["temperature"])
            parsed = extract_json(raw)

            if (
                parsed
                and "text" in parsed
                and "label" in parsed
                and "intensity" in parsed
            ):
                parsed["id"] = len(data)
                data.append(parsed)

                if len(data) % 20 == 0:
                    save_jsonl(data, output_path)

        except Exception as e:
            print(f"[Sentiment generation error] {e}")
            save_jsonl(data, output_path)
            continue

    save_jsonl(data, output_path)
    print("Sentiment done:", len(data))


if __name__ == "__main__":
    main()
