import re

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")
TAG_PATTERN = re.compile(r"\[(PER|ORG|LOC|MISC)\](.*?)\[/\1\]")


def tokenize(text):
    return TOKEN_PATTERN.findall(text)


def strip_tags(text):
    return re.sub(r"\[/?(PER|ORG|LOC|MISC)\]", "", text)


def extract_entities(text):
    return [(m.group(2), m.group(1)) for m in TAG_PATTERN.finditer(text)]


def find_span(tokens, phrase_tokens):
    for i in range(len(tokens) - len(phrase_tokens) + 1):
        if tokens[i : i + len(phrase_tokens)] == phrase_tokens:
            return i, i + len(phrase_tokens) - 1
    return None


def to_bio(tagged_text):
    entities = extract_entities(tagged_text)
    plain = strip_tags(tagged_text)
    tokens = tokenize(plain)

    tags = ["O"] * len(tokens)

    for text, label in entities:
        span = find_span(tokens, tokenize(text))
        if span is None:
            return None

        start, end = span
        tags[start] = f"B-{label}"
        for i in range(start + 1, end + 1):
            tags[i] = f"I-{label}"

    return tokens, tags
