import json
import re
from tqdm import tqdm

from src.generation.llm_client import LLMClient
from src.generation.prompts import sentiment_prompt
from src.utils.config import load_config
from src.utils.io import save_jsonl, ensure_dir


def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            return None
    return None


def main():
    cfg = load_config()

    client = LLMClient(cfg["llm"]["model_name"], cfg["llm"]["max_retries"])

    out_dir = ensure_dir("data/synthetic/sentiment")

    data = []

    for i in tqdm(range(cfg["synthetic"]["sentiment_samples"])):
        system, user, _ = sentiment_prompt()

        raw = client.generate(system, user, cfg["llm"]["temperature"])
        parsed = extract_json(raw)

        if parsed:
            parsed["id"] = len(data)
            data.append(parsed)

    save_jsonl(data, "data/synthetic/sentiment/synthetic_sentiment.jsonl")
    print("Sentiment done:", len(data))


if __name__ == "__main__":
    main()
