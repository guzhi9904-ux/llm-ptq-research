from __future__ import annotations

from types import SimpleNamespace

import torch

from flexrot.calibration import data


class _Stream:
    def __init__(self, rows):
        self.rows = rows
        self.shuffle_args = None

    def shuffle(self, *, seed, buffer_size):
        self.shuffle_args = (seed, buffer_size)
        return self

    def __iter__(self):
        return iter(self.rows)


class _Tokenizer:
    def __call__(self, text, **_kwargs):
        values = torch.arange(len(text), dtype=torch.long).unsqueeze(0)
        return SimpleNamespace(input_ids=values)


def test_fineweb_sampling_is_seeded_and_streaming(monkeypatch) -> None:
    stream = _Stream([{"text": "short"}, {"text": "0123456789abcdef"}])
    monkeypatch.setattr(data, "load_dataset", lambda *args, **kwargs: stream)
    first = data.load_fineweb_edu_windows(
        _Tokenizer(), seq_len=8, num_sequences=1, seed=7
    )
    second = data.load_fineweb_edu_windows(
        _Tokenizer(), seq_len=8, num_sequences=1, seed=7
    )
    assert stream.shuffle_args == (7, 1_000)
    assert torch.equal(first[0], second[0])
    assert first[0].shape == (1, 8)
