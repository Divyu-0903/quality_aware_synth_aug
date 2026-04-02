from tqdm import tqdm

from src.generation.llm_client import LLMClient
from src.generation.prompts import ner_prompt
from src.generation.ner_parser import to_bio
from src.utils.config import load_config
from src.utils.io import save_jsonl, ensure_dir


def main():
    cfg = load_config()

    client = LLMClient(cfg["llm"]["model_name"], cfg["llm"]["max_retries"])

    out_dir = ensure_dir("data/synthetic/ner")

    data = []

    for i in tqdm(range(cfg["synthetic"]["ner_samples"])):
        system, user, _ = ner_prompt()

        raw = client.generate(system, user, cfg["llm"]["temperature"])

        parsed = to_bio(raw)

        if parsed:
            tokens, tags = parsed
            data.append({"id": len(data), "tokens": tokens, "ner_tags": tags})

    save_jsonl(data, "data/synthetic/ner/synthetic_ner.jsonl")
    print("NER done:", len(data))


if __name__ == "__main__":
    main()
