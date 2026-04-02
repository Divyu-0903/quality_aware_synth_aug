from __future__ import annotations

NER_TEMPLATES = [
    {"template": "{PER} joined {ORG} in {LOC}.", "labels": ["PER", "ORG", "LOC"]},
    {
        "template": "{PER} visited {LOC} during a meeting with {ORG}.",
        "labels": ["PER", "LOC", "ORG"],
    },
    {"template": "{ORG} opened a new office in {LOC}.", "labels": ["ORG", "LOC"]},
    {"template": "{PER} won the {MISC}.", "labels": ["PER", "MISC"]},
    {"template": "{ORG} announced a partnership in {LOC}.", "labels": ["ORG", "LOC"]},
    {"template": "{PER} discussed {MISC} at {ORG}.", "labels": ["PER", "MISC", "ORG"]},
    {
        "template": "{PER} traveled from {LOC} to attend {MISC}.",
        "labels": ["PER", "LOC", "MISC"],
    },
    {"template": "{ORG} released information about {MISC}.", "labels": ["ORG", "MISC"]},
    {"template": "{PER} works at {ORG}.", "labels": ["PER", "ORG"]},
    {"template": "{PER} lives in {LOC}.", "labels": ["PER", "LOC"]},
    {"template": "{ORG} is based in {LOC}.", "labels": ["ORG", "LOC"]},
    {
        "template": "{MISC} was discussed by {PER} at {ORG}.",
        "labels": ["MISC", "PER", "ORG"],
    },
]
