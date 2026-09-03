import numpy as np
import pytest
import torch

try:
    import mlx.core as mx
except ImportError as exc:
    pytest.skip(f"MLX is unavailable: {exc}", allow_module_level=True)

from talkie.mlx.model import GPTConfig as MLXGPTConfig  # noqa: E402
from talkie.mlx.model import TalkieModel as MLXTalkieModel  # noqa: E402
from talkie.model import GPTConfig as TorchGPTConfig  # noqa: E402
from talkie.model import TalkieModel as TorchTalkieModel  # noqa: E402


def tiny_models() -> tuple[TorchTalkieModel, MLXTalkieModel]:
    torch.manual_seed(1930)
    torch_config = TorchGPTConfig(
        vocab_size=32,
        n_layer=2,
        n_head=4,
        n_embd=32,
        head_dim=8,
    )
    torch_model = TorchTalkieModel(
        torch_config, torch.device("cpu"), max_seq_len=32
    ).eval()
    with torch.no_grad():
        torch_model.lm_head.normal_(mean=0.0, std=0.02)

    weights = {
        name: mx.array(value.detach().float().numpy())
        for name, value in torch_model.state_dict().items()
    }
    mlx_config = MLXGPTConfig(
        vocab_size=32,
        n_layer=2,
        n_head=4,
        n_embd=32,
        head_dim=8,
        max_seq_len=32,
        dtype="float32",
        style="base",
    )
    return torch_model, MLXTalkieModel(weights, mlx_config)


def test_tiny_torch_and_mlx_forward_logits_match() -> None:
    torch_model, mlx_model = tiny_models()
    token_ids = np.array([[1, 7, 4, 12]], dtype=np.int32)

    with torch.no_grad():
        expected = torch_model(torch.from_numpy(token_ids).long()).numpy()
    actual, _ = mlx_model(mx.array(token_ids))
    mx.eval(actual)

    np.testing.assert_allclose(np.array(actual), expected, rtol=5e-4, atol=5e-5)


def test_mlx_cached_decode_matches_full_sequence() -> None:
    _, mlx_model = tiny_models()
    prefix = mx.array([[1, 7, 4]], dtype=mx.int32)
    full = mx.array([[1, 7, 4, 12]], dtype=mx.int32)

    full_logits, _ = mlx_model(full)
    _, cache = mlx_model(prefix)
    cached_logits, updated_cache = mlx_model(
        mx.array([[12]], dtype=mx.int32), cache=cache
    )
    mx.eval(full_logits, cached_logits, updated_cache)

    np.testing.assert_allclose(
        np.array(cached_logits), np.array(full_logits), rtol=5e-4, atol=5e-5
    )
    assert len(updated_cache) == mlx_model.config.n_layer
    for keys, values in updated_cache:
        assert keys.shape == (1, mlx_model.config.n_head, 4, mlx_model.config.head_dim)
        assert values.shape == keys.shape
