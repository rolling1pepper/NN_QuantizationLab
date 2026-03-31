from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest

from q_lab import cli, models
from q_lab.types import InputConfig, ModelFamily
from tests.helpers import KeywordVisionNet, TinyTextNet, TinyVisionNet, TwoInputTextNet


def test_infer_model_family_for_new_external_sources() -> None:
    assert models.infer_model_family("timm:resnet34", "auto") is ModelFamily.VISION
    assert models.infer_model_family("hf:acme-bert-encoder", "auto") is ModelFamily.TEXT


def test_load_model_from_python_module_factory() -> None:
    loaded_model = models.load_model(
        model_ref="python:tests.helpers:create_factory_vision_model",
        input_config=InputConfig(
            family=ModelFamily.VISION,
            batch_size=1,
            image_shape=(3, 8, 8),
            sequence_length=4,
            vocab_size=32,
        ),
        device="cpu",
    )

    assert loaded_model.source == "python_module"
    assert loaded_model.family is ModelFamily.VISION
    assert loaded_model.input_names == ("input",)


def test_load_model_from_python_file_factory(
    tmp_path: Path,
) -> None:
    factory_path = tmp_path / "factory_model.py"
    factory_path.write_text(
        "from tests.helpers import TinyTextNet\n\n"
        "def create_model(use_pretrained=False):\n"
        "    del use_pretrained\n"
        "    return TinyTextNet().eval()\n",
        encoding="utf-8",
    )
    model_ref = f"pyfile:{factory_path}::create_model"

    loaded_model = models.load_model(
        model_ref=model_ref,
        input_config=InputConfig(
            family=ModelFamily.TEXT,
            batch_size=1,
            image_shape=(3, 8, 8),
            sequence_length=6,
            vocab_size=128,
        ),
        device="cpu",
    )

    assert loaded_model.source == "python_file"
    assert loaded_model.family is ModelFamily.TEXT
    assert loaded_model.input_names == ("input_ids", "attention_mask", "token_type_ids")


def test_load_model_from_timm_reference(monkeypatch) -> None:
    fake_timm = ModuleType("timm")

    def create_model(model_name: str, pretrained: bool = False, **kwargs):
        del model_name, pretrained, kwargs
        return TinyVisionNet().eval()

    fake_timm.create_model = create_model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "timm", fake_timm)

    loaded_model = models.load_model(
        model_ref="timm:resnet34",
        input_config=InputConfig(
            family=ModelFamily.VISION,
            batch_size=1,
            image_shape=(3, 8, 8),
            sequence_length=4,
            vocab_size=32,
        ),
        device="cpu",
    )

    assert loaded_model.source == "timm"
    assert loaded_model.identifier.startswith("timm-")


def test_load_model_from_huggingface_reference(monkeypatch) -> None:
    monkeypatch.setattr(
        models.AutoConfig,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: SimpleNamespace(model_type="bert", to_dict=lambda: {"vocab_size": 128})),
    )
    monkeypatch.setattr(
        models.AutoModel,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: TinyTextNet().eval()),
    )

    loaded_model = models.load_model(
        model_ref="hf:acme-bert-encoder",
        input_config=InputConfig(
            family=ModelFamily.TEXT,
            batch_size=1,
            image_shape=(3, 8, 8),
            sequence_length=6,
            vocab_size=128,
        ),
        device="cpu",
        use_pretrained=True,
    )

    assert loaded_model.source == "huggingface"
    assert loaded_model.family is ModelFamily.TEXT
    assert loaded_model.invocation_mode is models.InvocationMode.KEYWORD


def test_load_model_from_huggingface_reference_infers_two_text_inputs(monkeypatch) -> None:
    monkeypatch.setattr(
        models.AutoConfig,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: SimpleNamespace(model_type="distilbert", to_dict=lambda: {"vocab_size": 128})),
    )
    monkeypatch.setattr(
        models.AutoModel,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: TwoInputTextNet().eval()),
    )

    loaded_model = models.load_model(
        model_ref="hf:acme-distilbert-encoder",
        input_config=InputConfig(
            family=ModelFamily.TEXT,
            batch_size=1,
            image_shape=(3, 8, 8),
            sequence_length=6,
            vocab_size=128,
        ),
        device="cpu",
        use_pretrained=True,
    )

    assert loaded_model.input_names == ("input_ids", "attention_mask")
    assert len(loaded_model.example_inputs) == 2


@pytest.mark.integration
def test_cli_runs_python_module_vision_model_with_pruning_and_static_quantization(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "python_module_vision.csv"

    exit_code = cli.main(
        [
            "python:tests.helpers:create_factory_vision_model",
            "--task",
            "vision",
            "--pruning",
            "structured",
            "--pruning-amount",
            "0.25",
            "--quantization",
            "static",
            "--image-shape",
            "3,8,8",
            "--warmup-iterations",
            "1",
            "--benchmark-iterations",
            "2",
            "--calibration-iterations",
            "1",
            "--report-path",
            str(report_path),
        ]
    )

    assert exit_code == 0
    dataframe = pd.read_csv(report_path)

    assert set(dataframe["label"]) == {"baseline", "optimized"}
    assert set(dataframe["source"]) == {"python_module"}
    assert set(dataframe["pruning"]) == {"none", "structured"}
    assert set(dataframe["quantization"]) == {"none", "static"}


@pytest.mark.integration
def test_cli_runs_python_file_text_model_with_dynamic_quantization(
    tmp_path: Path,
) -> None:
    factory_path = tmp_path / "text_factory.py"
    factory_path.write_text(
        "from tests.helpers import TinyTextNet\n\n"
        "def create_model(use_pretrained=False):\n"
        "    del use_pretrained\n"
        "    return TinyTextNet().eval()\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "python_file_text.csv"

    exit_code = cli.main(
        [
            f"pyfile:{factory_path}::create_model",
            "--task",
            "text",
            "--quantization",
            "dynamic",
            "--sequence-length",
            "6",
            "--vocab-size",
            "128",
            "--warmup-iterations",
            "1",
            "--benchmark-iterations",
            "2",
            "--report-path",
            str(report_path),
        ]
    )

    assert exit_code == 0
    dataframe = pd.read_csv(report_path)

    assert set(dataframe["label"]) == {"baseline", "optimized"}
    assert set(dataframe["source"]) == {"python_file"}
    assert set(dataframe["quantization"]) == {"none", "dynamic"}


@pytest.mark.integration
def test_cli_runs_timm_model_with_static_quantization(
    monkeypatch,
    tmp_path: Path,
) -> None:
    fake_timm = ModuleType("timm")

    def create_model(model_name: str, pretrained: bool = False, **kwargs):
        del model_name, pretrained, kwargs
        return TinyVisionNet().eval()

    fake_timm.create_model = create_model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "timm", fake_timm)

    report_path = tmp_path / "timm_report.csv"

    exit_code = cli.main(
        [
            "timm:resnet34",
            "--quantization",
            "static",
            "--image-shape",
            "3,8,8",
            "--warmup-iterations",
            "1",
            "--benchmark-iterations",
            "2",
            "--calibration-iterations",
            "1",
            "--report-path",
            str(report_path),
        ]
    )

    assert exit_code == 0
    dataframe = pd.read_csv(report_path)

    assert set(dataframe["source"]) == {"timm"}
    assert set(dataframe["quantization"]) == {"none", "static"}


@pytest.mark.integration
def test_cli_runs_huggingface_text_model_with_dynamic_quantization(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        models.AutoConfig,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: SimpleNamespace(model_type="bert", to_dict=lambda: {"vocab_size": 128})),
    )
    monkeypatch.setattr(
        models.AutoModel,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: TinyTextNet().eval()),
    )

    report_path = tmp_path / "hf_text_report.csv"

    exit_code = cli.main(
        [
            "hf:acme-bert-encoder",
            "--pretrained",
            "--quantization",
            "dynamic",
            "--sequence-length",
            "6",
            "--vocab-size",
            "128",
            "--warmup-iterations",
            "1",
            "--benchmark-iterations",
            "2",
            "--report-path",
            str(report_path),
        ]
    )

    assert exit_code == 0
    dataframe = pd.read_csv(report_path)

    assert set(dataframe["source"]) == {"huggingface"}
    assert set(dataframe["quantization"]) == {"none", "dynamic"}


@pytest.mark.integration
def test_cli_runs_huggingface_vision_model_with_export(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        models.AutoConfig,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: SimpleNamespace(model_type="vit", to_dict=lambda: {"image_size": 8, "num_channels": 3})),
    )
    monkeypatch.setattr(
        models.AutoModel,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: KeywordVisionNet().eval()),
    )

    report_path = tmp_path / "hf_vision_report.csv"
    onnx_path = tmp_path / "hf_vision.onnx"

    exit_code = cli.main(
        [
            "hf:acme-vit-encoder",
            "--pretrained",
            "--export-onnx",
            "--image-shape",
            "3,8,8",
            "--warmup-iterations",
            "1",
            "--benchmark-iterations",
            "2",
            "--onnx-path",
            str(onnx_path),
            "--report-path",
            str(report_path),
        ]
    )

    assert exit_code == 0
    assert onnx_path.exists()
    dataframe = pd.read_csv(report_path)

    assert set(dataframe["label"]) == {"baseline", "baseline-onnx"}
    assert set(dataframe["source"]) == {"huggingface", "onnx"}
