from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from meeting_transcriber.config import Config, load_config


@pytest.fixture
def config(tmp_path: Path) -> Config:
    root = Path(__file__).resolve().parents[1]
    return replace(load_config(root / "config.yaml"), input_dir=tmp_path / "input",
                   output_dir=tmp_path / "output", logs_dir=tmp_path / "logs", models_dir=tmp_path / "models")


def fake_segment(start: float, end: float, text: str = "讨论 iQube 批次。", words=None):
    return SimpleNamespace(start=start, end=end, text=text, words=words, avg_logprob=-0.2,
                           no_speech_prob=0.01, compression_ratio=1.2)

