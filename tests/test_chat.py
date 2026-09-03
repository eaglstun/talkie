from talkie.chat import Message, format_chat, format_prompt, truncate_at_stop


def test_format_chat_preserves_roles_and_opens_assistant_turn() -> None:
    messages = [
        Message(role="system", content="Be concise."),
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Good day."),
        Message(role="user", content="Continue"),
    ]

    assert format_chat(messages) == (
        "<|system|>Be concise.<|end|>"
        "<|user|>Hello<|end|>"
        "<|assistant|>Good day.<|end|>"
        "<|user|>Continue<|end|>"
        "<|assistant|>"
    )


def test_format_prompt_wraps_one_user_turn() -> None:
    assert format_prompt("Hello") == "<|user|>Hello<|end|><|assistant|>"


def test_truncate_at_stop_uses_earliest_marker() -> None:
    text = "A reply<|end|>ignored<|user|>also ignored"

    assert truncate_at_stop(text) == ("A reply", True)
    assert truncate_at_stop("A complete reply") == ("A complete reply", False)
