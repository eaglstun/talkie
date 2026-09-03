import torch

from talkie.sampling import (
    apply_top_k_top_p,
    list_top_k_tensor,
    list_top_p_tensor,
    scalar_top_k_tensor,
    scalar_top_p_tensor,
)


CPU = torch.device("cpu")


def test_top_k_supports_per_row_limits() -> None:
    logits = torch.tensor([[1.0, 4.0, 3.0, 2.0], [9.0, 8.0, 7.0, 6.0]])

    filtered = apply_top_k_top_p(logits, top_k=torch.tensor([2, 1]))

    assert torch.isfinite(filtered[0]).tolist() == [False, True, True, False]
    assert torch.isfinite(filtered[1]).tolist() == [True, False, False, False]


def test_top_p_keeps_first_token_past_threshold() -> None:
    probabilities = torch.tensor([[0.60, 0.25, 0.10, 0.05]])
    logits = probabilities.log()

    filtered = apply_top_k_top_p(logits, top_p=torch.tensor(0.70))

    assert torch.isfinite(filtered).tolist() == [[True, True, False, False]]


def test_sampling_argument_helpers_normalize_disabled_values() -> None:
    assert scalar_top_p_tensor(None, CPU) is None
    assert scalar_top_p_tensor(1.0, CPU) is None
    assert scalar_top_k_tensor(0, CPU) is None
    assert list_top_p_tensor([None, 1.0], CPU) is None
    assert list_top_k_tensor([None, 0], vocab_size=10, device=CPU) is None

    assert scalar_top_p_tensor(0.9, CPU).item() == torch.tensor(0.9).item()
    assert scalar_top_k_tensor(4, CPU).item() == 4
    assert list_top_p_tensor([0.8, None], CPU).tolist() == [
        [torch.tensor(0.8).item()],
        [1.0],
    ]
    assert list_top_k_tensor([3, None], vocab_size=10, device=CPU).tolist() == [
        3,
        10,
    ]
