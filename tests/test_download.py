from pathlib import Path
from unittest.mock import call, patch

import pytest

from talkie.download import download_model, get_model_files


@patch("talkie.download.hf_hub_download")
def test_download_model_fetches_checkpoint_and_vocab(mock_download) -> None:
    snapshot = Path("/cache/models/snapshots/revision")
    mock_download.side_effect = [
        str(snapshot / "rl-refined.pt"),
        str(snapshot / "vocab.txt"),
    ]

    result = download_model("talkie-1930-13b-it", cache_dir=Path("/cache"))

    assert result == snapshot
    assert mock_download.call_args_list == [
        call(
            repo_id="talkie-lm/talkie-1930-13b-it",
            filename="rl-refined.pt",
            cache_dir="/cache",
        ),
        call(
            repo_id="talkie-lm/talkie-1930-13b-it",
            filename="vocab.txt",
            cache_dir="/cache",
        ),
    ]


@patch("talkie.download.hf_hub_download")
def test_get_model_files_returns_both_paths(mock_download) -> None:
    checkpoint = "/cache/snapshots/revision/final.ckpt"
    vocab = "/cache/snapshots/revision/vocab.txt"
    mock_download.side_effect = [checkpoint, vocab]

    result = get_model_files("talkie-1930-13b-base")

    assert result == (Path(checkpoint), Path(vocab))
    assert mock_download.call_args_list == [
        call(
            repo_id="talkie-lm/talkie-1930-13b-base",
            filename="final.ckpt",
        ),
        call(
            repo_id="talkie-lm/talkie-1930-13b-base",
            filename="vocab.txt",
        ),
    ]


@patch("talkie.download.hf_hub_download")
def test_unknown_model_fails_before_download(mock_download) -> None:
    with pytest.raises(ValueError, match="Unknown model 'missing'"):
        get_model_files("missing")

    mock_download.assert_not_called()
