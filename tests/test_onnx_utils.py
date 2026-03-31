from __future__ import annotations

import pytest
import torch

from q_lab.onnx_utils import (
    SyntheticCalibrationDataReader,
    build_onnx_dummy_inputs,
    infer_family_from_input_specs,
    resolve_onnx_providers,
)
from q_lab.types import InputConfig, InputTensorSpec, ModelFamily


def test_resolve_onnx_providers_preserves_requested_order(monkeypatch) -> None:
    monkeypatch.setattr(
        "q_lab.onnx_utils.ort.get_available_providers",
        lambda: ["CPUExecutionProvider", "CUDAExecutionProvider"],
    )

    providers = resolve_onnx_providers(("CUDAExecutionProvider", "CPUExecutionProvider"))

    assert providers == ("CUDAExecutionProvider", "CPUExecutionProvider")


def test_infer_family_from_input_specs_detects_text_model() -> None:
    family = infer_family_from_input_specs(
        model_path=None,
        input_specs=(
            InputTensorSpec(name="input_ids", dtype="tensor(int64)", shape=("batch", "sequence")),
            InputTensorSpec(name="attention_mask", dtype="tensor(int64)", shape=("batch", "sequence")),
        ),
    )

    assert family is ModelFamily.TEXT


def test_build_onnx_dummy_inputs_respects_shapes_and_dtypes() -> None:
    example_inputs, input_names = build_onnx_dummy_inputs(
        input_specs=(
            InputTensorSpec(name="input_ids", dtype="tensor(int64)", shape=("batch", "sequence")),
            InputTensorSpec(name="attention_mask", dtype="tensor(int64)", shape=("batch", "sequence")),
        ),
        config=InputConfig(
            family=ModelFamily.TEXT,
            batch_size=2,
            image_shape=(3, 8, 8),
            sequence_length=6,
            vocab_size=128,
        ),
    )

    assert input_names == ("input_ids", "attention_mask")
    assert all(tensor.shape == (2, 6) for tensor in example_inputs)
    assert all(tensor.dtype == torch.int64 for tensor in example_inputs)


def test_synthetic_calibration_data_reader_yields_expected_batches() -> None:
    reader = SyntheticCalibrationDataReader(
        input_names=("input",),
        template_inputs=(torch.randn(1, 3, 8, 8),),
        calibration_iterations=2,
    )

    first_batch = reader.get_next()
    second_batch = reader.get_next()
    third_batch = reader.get_next()

    assert first_batch is not None
    assert second_batch is not None
    assert third_batch is None
