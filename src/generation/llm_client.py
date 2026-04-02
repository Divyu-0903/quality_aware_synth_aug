from __future__ import annotations

import os
import time
from typing import Optional

from dotenv import load_dotenv
from groq import Groq

load_dotenv()


class LLMClient:
    def __init__(self, model_name: str, max_retries: int = 3):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found")

        self.client = Groq(api_key=api_key)
        self.model_name = model_name
        self.max_retries = max_retries

    def generate(self, system_prompt: str, user_prompt: str, temperature: float):
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                res = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                )
                return res.choices[0].message.content.strip()

            except Exception as e:
                last_error = e
                time.sleep(1.5 * (attempt + 1))

        raise RuntimeError(f"LLM failed: {last_error}")
