import tiktoken

_enc: tiktoken.Encoding | None = None


def get_encoding() -> tiktoken.Encoding:
    global _enc
    if _enc is None:
        _enc = tiktoken.get_encoding("cl100k_base")
    return _enc


def count_tokens(text: str) -> int:
    enc = get_encoding()
    return len(enc.encode(text))


def count_tokens_for_messages(messages: list[dict]) -> int:
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += count_tokens(content)
    return total
