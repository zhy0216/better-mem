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

