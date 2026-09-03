import base64

import pytest

import talkie.tokenizer as tokenizer_module
from talkie.tokenizer import IT_VOCAB_SIZE, build_tokenizer


@pytest.fixture
def byte_vocab(tmp_path, monkeypatch):
    vocab_path = tmp_path / "vocab.txt"
    lines = [
        f"{base64.b64encode(bytes([token])).decode()} {token}"
        for token in range(256)
    ]
    vocab_path.write_text("\n".join(lines) + "\n")

    monkeypatch.setattr(tokenizer_module, "BASE_VOCAB_SIZE", 257)
    monkeypatch.setattr(
        tokenizer_module,
        "_BASE_SPECIAL_TOKENS",
        {"<|endoftext|>": 256},
    )
    monkeypatch.setattr(
        tokenizer_module,
        "_IT_SPECIAL_TOKENS",
        {
            "<|endoftext|>": 256,
            "<|end|>": 257,
            "<|user|>": 258,
            "<|assistant|>": 259,
            "<|system|>": 260,
        },
    )
    return vocab_path


def test_base_tokenizer_round_trip(byte_vocab) -> None:
    tokenizer = build_tokenizer(byte_vocab, style="base")
    text = "Radio, café, and science — 1930."

    token_ids = tokenizer.encode(text)

    assert tokenizer.name == "talkie-base"
    assert tokenizer.decode(token_ids) == text
    assert tokenizer.encode_single_token("<|endoftext|>") == 256
    assert tokenizer.special_tokens_set == {"<|endoftext|>"}


def test_it_tokenizer_preserves_chat_markers(byte_vocab) -> None:
    tokenizer = build_tokenizer(byte_vocab, style="it")
    prompt = "<|system|>Brief.<|end|><|user|>Hello<|end|><|assistant|>"

    token_ids = tokenizer.encode(prompt, allowed_special="all")

    assert tokenizer.name == "talkie-it"
    assert tokenizer.decode(token_ids) == prompt
    assert token_ids[0] == 260
    assert token_ids[-1] == 259


def test_production_it_vocabulary_layout() -> None:
    assert IT_VOCAB_SIZE == 65_540
    assert tokenizer_module._IT_SPECIAL_TOKENS == {
        "<|endoftext|>": 65_535,
        "<|end|>": 65_536,
        "<|user|>": 65_537,
        "<|assistant|>": 65_538,
        "<|system|>": 65_539,
    }
