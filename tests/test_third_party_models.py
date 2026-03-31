from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from q_lab import cli, models
from q_lab.types import ModelFamily


def test_infer_model_family_for_third_party_text_torchscript_path(
    scripted_third_party_text_model_path: Path,
) -> None:
    inferred_family = models.infer_model_family(
        str(scripted_third_party_text_model_path),
        "auto",
    )

    assert inferred_family is ModelFamily.TEXT


@pytest.mark.integration
def test_cli_runs_third_party_torchscript_vision_model_with_multiple_outputs(
    scripted_third_party_vision_model_path: Path,
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "third_party_vision.csv"
    onnx_path = tmp_path / "third_party_vision.onnx"

    exit_code = cli.main(
        [
            str(scripted_third_party_vision_model_path),
            "--task",
            "vision",
            "--image-shape",
            "3,8,8",
            "--warmup-iterations",
            "1",
            "--benchmark-iterations",
            "2",
            "--export-onnx",
            "--onnx-path",
            str(onnx_path),
            "--report-path",
            str(report_path),
        ]
    )

    assert exit_code == 0
    assert report_path.exists()
    assert onnx_path.exists()

    dataframe = pd.read_csv(report_path)

    assert set(dataframe["label"]) == {"baseline", "baseline-onnx"}
    assert set(dataframe["backend"]) == {"pytorch", "onnxruntime"}


@pytest.mark.integration
def test_cli_runs_third_party_torchscript_text_model(
    scripted_third_party_text_model_path: Path,
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "third_party_text.csv"

    exit_code = cli.main(
        [
            str(scripted_third_party_text_model_path),
            "--task",
            "text",
            "--sequence-length",
            "6",
            "--vocab-size",
            "256",
            "--warmup-iterations",
            "1",
            "--benchmark-iterations",
            "2",
            "--report-path",
            str(report_path),
        ]
    )

    assert exit_code == 0
    assert report_path.exists()

    dataframe = pd.read_csv(report_path)

    assert list(dataframe["label"]) == ["baseline"]
    assert list(dataframe["backend"]) == ["pytorch"]
    assert list(dataframe["source"]) == ["torchscript"]
