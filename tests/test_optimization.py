from __future__ import annotations

import torch
import pytest

from q_lab.optimization import apply_pruning, optimize_model
from q_lab.models import load_model
from q_lab.types import CompressionConfig, InputConfig, ModelFamily, PruningMode, QuantizationMode


def test_apply_unstructured_pruning_reports_sparsity(tiny_vision_model) -> None:
    _, sparsity_pct = apply_pruning(
        model=tiny_vision_model,
        mode=PruningMode.UNSTRUCTURED,
        amount=0.5,
    )

    assert sparsity_pct > 0.0


def test_optimize_model_dynamic_quantizes_text_model(
    benchmark_config,
    text_loaded_model,
) -> None:
    outcome = optimize_model(
        loaded_model=text_loaded_model,
        compression=CompressionConfig(
            quantization=QuantizationMode.DYNAMIC,
            pruning=PruningMode.NONE,
            pruning_amount=0.0,
            export_onnx=False,
            onnx_path=None,
            onnx_opset=17,
        ),
        benchmark_config=benchmark_config,
    )

    module_locations = {
        module.__class__.__module__
        for module in outcome.model.modules()
    }
    assert any("quantized.dynamic" in module_path for module_path in module_locations)
    assert outcome.quantization is QuantizationMode.DYNAMIC


def test_optimize_model_static_quantizes_vision_model(
    benchmark_config,
    vision_loaded_model,
) -> None:
    outcome = optimize_model(
        loaded_model=vision_loaded_model,
        compression=CompressionConfig(
            quantization=QuantizationMode.STATIC,
            pruning=PruningMode.NONE,
            pruning_amount=0.0,
            export_onnx=False,
            onnx_path=None,
            onnx_opset=17,
        ),
        benchmark_config=benchmark_config,
    )

    with torch.inference_mode():
        output = outcome.model(vision_loaded_model.example_inputs[0])

    assert output.shape[0] == 1
    assert any("Static FX quantization" in note for note in outcome.notes)


def test_optimize_model_rejects_compression_for_torchscript_source(
    benchmark_config,
    vision_loaded_model,
) -> None:
    vision_loaded_model.source = "torchscript"

    with pytest.raises(ValueError, match="supported only for eager PyTorch models"):
        optimize_model(
            loaded_model=vision_loaded_model,
            compression=CompressionConfig(
                quantization=QuantizationMode.DYNAMIC,
                pruning=PruningMode.NONE,
                pruning_amount=0.0,
                export_onnx=False,
                onnx_path=None,
                onnx_opset=17,
            ),
            benchmark_config=benchmark_config,
        )


def test_optimize_model_dynamic_quantizes_onnx_text_model(
    benchmark_config,
    onnx_text_model_path,
) -> None:
    loaded_model = load_model(
        model_ref=str(onnx_text_model_path),
        input_config=InputConfig(
            family=ModelFamily.TEXT,
            batch_size=1,
            image_shape=(3, 8, 8),
            sequence_length=6,
            vocab_size=256,
        ),
        device="cpu",
    )

    outcome = optimize_model(
        loaded_model=loaded_model,
        compression=CompressionConfig(
            quantization=QuantizationMode.DYNAMIC,
            pruning=PruningMode.NONE,
            pruning_amount=0.0,
            export_onnx=False,
            onnx_path=None,
            onnx_opset=17,
        ),
        benchmark_config=benchmark_config,
    )

    assert outcome.artifact_path is not None and outcome.artifact_path.exists()
    assert outcome.quantization is QuantizationMode.DYNAMIC
    assert any("ONNX Runtime dynamic quantization" in note for note in outcome.notes)


def test_optimize_model_static_quantizes_onnx_vision_model(
    benchmark_config,
    onnx_vision_model_path,
) -> None:
    loaded_model = load_model(
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

    outcome = optimize_model(
        loaded_model=loaded_model,
        compression=CompressionConfig(
            quantization=QuantizationMode.STATIC,
            pruning=PruningMode.NONE,
            pruning_amount=0.0,
            export_onnx=False,
            onnx_path=None,
            onnx_opset=17,
        ),
        benchmark_config=benchmark_config,
    )

    assert outcome.artifact_path is not None and outcome.artifact_path.exists()
    assert any("static quantization" in note.lower() for note in outcome.notes)


def test_optimize_model_rejects_pruning_for_onnx_input(
    benchmark_config,
    onnx_vision_model_path,
) -> None:
    loaded_model = load_model(
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

    with pytest.raises(ValueError, match="Pruning is not supported for ONNX inputs"):
        optimize_model(
            loaded_model=loaded_model,
            compression=CompressionConfig(
                quantization=QuantizationMode.DYNAMIC,
                pruning=PruningMode.UNSTRUCTURED,
                pruning_amount=0.1,
                export_onnx=False,
                onnx_path=None,
                onnx_opset=17,
            ),
            benchmark_config=benchmark_config,
        )
