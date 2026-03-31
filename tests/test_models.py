from __future__ import annotations

from pathlib import Path

import pytest

from q_lab import models
from q_lab.types import InputConfig, InvocationMode, ModelFamily


def test_infer_model_family_for_supported_references(
    scripted_vision_model_path: Path,
    onnx_text_model_path: Path,
) -> None:
    assert models.infer_model_family("resnet18", "auto") is ModelFamily.VISION
    assert models.infer_model_family("bert", "auto") is ModelFamily.TEXT
    assert models.infer_model_family(str(scripted_vision_model_path), "vision") is ModelFamily.VISION
    assert models.infer_model_family(str(onnx_text_model_path), "auto") is ModelFamily.TEXT


def test_infer_model_family_requires_explicit_task_for_ambiguous_torchscript_path(
    scripted_vision_model_path: Path,
) -> None:
    with pytest.raises(ValueError, match="Unable to infer task type"):
        models.infer_model_family(str(scripted_vision_model_path), "auto")


def test_build_dummy_inputs_for_vision() -> None:
    example_inputs, input_names, invocation_mode = models.build_dummy_inputs(
        InputConfig(
            family=ModelFamily.VISION,
            batch_size=2,
            image_shape=(3, 16, 16),
            sequence_length=4,
            vocab_size=32,
        ),
        InvocationMode.POSITIONAL,
    )

    assert input_names == ("input",)
    assert invocation_mode is InvocationMode.POSITIONAL
    assert example_inputs[0].shape == (2, 3, 16, 16)


def test_build_dummy_inputs_for_text() -> None:
    example_inputs, input_names, invocation_mode = models.build_dummy_inputs(
        InputConfig(
            family=ModelFamily.TEXT,
            batch_size=2,
            image_shape=(3, 16, 16),
            sequence_length=7,
            vocab_size=64,
        ),
        InvocationMode.KEYWORD,
    )

    assert input_names == ("input_ids", "attention_mask", "token_type_ids")
    assert invocation_mode is InvocationMode.KEYWORD
    assert all(tensor.shape == (2, 7) for tensor in example_inputs)


def test_load_model_from_torchscript(
    scripted_vision_model_path: Path,
) -> None:
    loaded_model = models.load_model(
        model_ref=str(scripted_vision_model_path),
        input_config=InputConfig(
            family=ModelFamily.VISION,
            batch_size=1,
            image_shape=(3, 8, 8),
            sequence_length=4,
            vocab_size=32,
        ),
        device="cpu",
    )

    assert loaded_model.source == "torchscript"
    assert loaded_model.family is ModelFamily.VISION
    assert loaded_model.original_path == scripted_vision_model_path


def test_load_model_from_onnx(
    onnx_vision_model_path: Path,
) -> None:
    loaded_model = models.load_model(
        model_ref=str(onnx_vision_model_path),
        input_config=InputConfig(
            family=ModelFamily.VISION,
            batch_size=1,
            image_shape=(3, 8, 8),
            sequence_length=4,
            vocab_size=32,
        ),
        device="cpu",
    )

    assert loaded_model.source == "onnx"
    assert loaded_model.original_path == onnx_vision_model_path
    assert loaded_model.input_names == ("input",)


def test_load_model_from_patched_torchvision_registry(monkeypatch) -> None:
    monkeypatch.setitem(
        models.SUPPORTED_TORCHVISION_MODELS,
        "toyvision",
        ("Toy Vision", lambda use_pretrained: models.nn.Sequential()),
    )
    loaded_model = models.load_model(
        model_ref="toyvision",
        input_config=InputConfig(
            family=ModelFamily.VISION,
            batch_size=1,
            image_shape=(3, 8, 8),
            sequence_length=4,
            vocab_size=32,
        ),
        device="cpu",
    )

    assert loaded_model.source == "torchvision"
    assert loaded_model.identifier == "toyvision"
