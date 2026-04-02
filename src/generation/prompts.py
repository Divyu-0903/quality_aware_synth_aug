import random

NER_PATTERNS = [
    ["PER", "LOC"],
    ["PER", "ORG"],
    ["ORG", "LOC"],
    ["PER", "ORG", "LOC"],
    ["PER", "MISC"],
]

SENTIMENT = [
    ("positive", "mild"),
    ("positive", "strong"),
    ("negative", "mild"),
    ("negative", "strong"),
]


def ner_prompt():
    labels = random.choice(NER_PATTERNS)
    label_str = ", ".join(labels)

    system = "Generate NER data with inline tags only."

    user = f"""
Generate ONE sentence.

Rules:
- Use tags: [PER], [ORG], [LOC], [MISC]
- Include: {label_str}
- 8-18 words
- Natural sentence
- Output ONLY tagged sentence

Example:
[PER]Elon Musk[/PER] visited [LOC]Paris[/LOC].
"""

    return system, user, labels


def sentiment_prompt():
    label, intensity = random.choice(SENTIMENT)

    system = "Generate sentiment dataset."

    user = f"""
Generate ONE JSON:

{{
"text": "...",
"label": "{label}",
"intensity": "{intensity}"
}}

Rules:
- 8-20 words
- Natural review
- Output ONLY JSON
"""

    return system, user, (label, intensity)
