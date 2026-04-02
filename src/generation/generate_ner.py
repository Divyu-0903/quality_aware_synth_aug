from __future__ import annotations

import os
from tqdm import tqdm

from src.generation.llm_client import LLMClient
from src.generation.prompts import ner_prompt
from src.generation.ner_parser import to_bio
from src.utils.config import load_config
from src.utils.io import save_jsonl, load_jsonl, ensure_dir


def main():
    cfg = load_config()

    client = LLMClient(cfg["llm"]["model_name"], cfg["llm"]["max_retries"])

    ensure_dir("data/synthetic/ner")
    output_path = "data/synthetic/ner/synthetic_ner.jsonl"

    data = []
    if os.path.exists(output_path):
        data = load_jsonl(output_path)

    existing = len(data)
    target_n = cfg["synthetic"]["ner_samples"]

    print(f"Resuming from {existing} / {target_n}")

    for _ in tqdm(range(existing, target_n)):
        system, user, _ = ner_prompt()

        try:
            raw = client.generate(system, user, cfg["llm"]["temperature"])
            parsed = to_bio(raw)

            if parsed:
                tokens, tags = parsed
                data.append({"id": len(data), "tokens": tokens, "ner_tags": tags})

                if len(data) % 20 == 0:
                    save_jsonl(data, output_path)

        except Exception as e:
            print(f"[NER generation error] {e}")
            save_jsonl(data, output_path)
            continue

    save_jsonl(data, output_path)
    print("NER done:", len(data))


if __name__ == "__main__":
    main()
