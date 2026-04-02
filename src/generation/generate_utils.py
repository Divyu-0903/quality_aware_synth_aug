from __future__ import annotations

import json
import re
from typing import Any

import torch


def generate_text(
    prompt: str,
    tokenizer,
    model,
    device: str,
    max_input_length: int = 256,
    max_output_length: int = 128,
    temperature: float = 0.7,
    do_sample: bool = True,
) -> str:
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=max_input_length
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_output_length,
            do_sample=do_sample,
            temperature=temperature,
            top_p=0.9,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.strip()


def extract_json_object(text: str) -> str | None:
    text = text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        return match.group(0).strip()

    return None


def try_parse_json(text: str) -> dict[str, Any] | None:
    candidate = extract_json_object(text)
    if candidate is None:
        return None

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None
