from __future__ import annotations

from dataclasses import dataclass

import torch

from q_lab.benchmark import BenchmarkEngine, canonicalize_outputs, compare_outputs


@dataclass
class NestedOutput:
    logits: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...]


def test_canonicalize_outputs_handles_nested_structures() -> None:
    nested_output = {
        "payload": NestedOutput(
            logits=torch.ones(1, 4),
            hidden_states=(torch.zeros(1, 4),),
        ),
        "tail": [torch.full((1, 4), 2.0)],
    }

    flattened = canonicalize_outputs(nested_output)

    assert len(flattened) == 3
    assert all(isinstance(tensor, torch.Tensor) for tensor in flattened)


def test_compare_outputs_reports_similarity_metrics() -> None:
    reference = (torch.tensor([[0.0, 1.0], [3.0, 0.5]]),)
    candidate = (torch.tensor([[0.1, 0.9], [2.9, 0.6]]),)

    metrics = compare_outputs(reference, candidate)

    assert metrics.accuracy_proxy_pct == 100.0
    assert metrics.cosine_similarity is not None and metrics.cosine_similarity > 0.99
    assert metrics.max_abs_diff is not None and metrics.max_abs_diff > 0.0


def test_benchmark_engine_runs_pytorch_benchmark(
    benchmark_config,
    vision_loaded_model,
) -> None:
    engine = BenchmarkEngine(benchmark_config)
    reference_outputs = engine.capture_reference_outputs(
        model=vision_loaded_model.model,
        loaded_model=vision_loaded_model,
    )
    result = engine.benchmark_pytorch(
        label="baseline",
        model=vision_loaded_model.model,
        loaded_model=vision_loaded_model,
        size_mb=engine.measure_pytorch_size_mb(vision_loaded_model.model),
        quantization="none",
        pruning="none",
        pruning_amount=0.0,
        sparsity_pct=0.0,
        reference_outputs=reference_outputs,
    )

    assert result.stats.mean_latency_ms >= 0.0
    assert result.stats.p95_latency_ms >= 0.0
    assert result.size_mb > 0.0
    assert result.fidelity.accuracy_proxy_pct == 100.0


def test_benchmark_engine_runs_onnx_benchmark(
    benchmark_config,
    onnx_vision_model_path,
) -> None:
    from q_lab.models import load_model
    from q_lab.types import InputConfig, ModelFamily

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
    engine = BenchmarkEngine(benchmark_config)
    reference_outputs = engine.capture_onnx_reference_outputs(
        session=loaded_model.model,
        loaded_model=loaded_model,
    )
    result = engine.benchmark_onnx(
        label="baseline",
        session=loaded_model.model,
        loaded_model=loaded_model,
        size_mb=engine.measure_file_size_mb(onnx_vision_model_path),
        quantization="none",
        pruning="none",
        pruning_amount=0.0,
        sparsity_pct=0.0,
        reference_outputs=reference_outputs,
        artifact_path=str(onnx_vision_model_path),
    )

    assert result.stats.mean_latency_ms >= 0.0
    assert result.size_mb > 0.0
    assert "ExecutionProvider" in result.execution_target
