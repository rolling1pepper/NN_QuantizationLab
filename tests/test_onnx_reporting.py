from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from rich.console import Console

pytest.importorskip("onnxruntime")

from q_lab.benchmark import BenchmarkEngine
from q_lab.onnx_utils import create_onnx_session, export_to_onnx
from q_lab.reporting import render_results, save_results_csv, save_results_html
from q_lab.types import (
    BenchmarkResult,
    BenchmarkStats,
    EvaluationMetrics,
    FidelityMetrics,
    RuntimeBackend,
)


@pytest.mark.integration
def test_export_to_onnx_and_benchmark(
    benchmark_config,
    vision_loaded_model,
    tmp_path: Path,
) -> None:
    engine = BenchmarkEngine(benchmark_config)
    baseline_reference = engine.capture_reference_outputs(
        model=vision_loaded_model.model,
        loaded_model=vision_loaded_model,
    )
    onnx_path = export_to_onnx(
        loaded_model=vision_loaded_model,
        model=vision_loaded_model.model,
        destination=tmp_path / "toy_vision.onnx",
        opset_version=17,
    )
    session = create_onnx_session(onnx_path)
    result = engine.benchmark_onnx(
        label="baseline-onnx",
        session=session,
        loaded_model=vision_loaded_model,
        size_mb=engine.measure_file_size_mb(onnx_path),
        quantization="none",
        pruning="none",
        pruning_amount=0.0,
        sparsity_pct=0.0,
        reference_outputs=baseline_reference,
        artifact_path=str(onnx_path),
    )

    assert onnx_path.exists()
    assert result.backend is RuntimeBackend.ONNX_RUNTIME
    assert result.size_mb > 0.0
    assert result.fidelity.cosine_similarity is not None
    assert result.fidelity.cosine_similarity > 0.99
    assert "ExecutionProvider" in result.execution_target


def test_render_results_and_save_csv(tmp_path: Path) -> None:
    result = BenchmarkResult(
        label="baseline",
        backend=RuntimeBackend.PYTORCH,
        model_name="Toy Vision",
        source="unit-test",
        batch_size=1,
        quantization="none",
        pruning="none",
        pruning_amount=0.0,
        stats=BenchmarkStats(
            mean_latency_ms=1.23,
            std_latency_ms=0.1,
            p95_latency_ms=1.4,
            throughput_items_per_sec=812.5,
            peak_memory_mb=3.2,
        ),
        size_mb=3.21,
        sparsity_pct=0.0,
        fidelity=FidelityMetrics(accuracy_proxy_pct=100.0, cosine_similarity=1.0, max_abs_diff=0.0),
        evaluation=EvaluationMetrics(sample_count=2, top1_accuracy_pct=100.0, macro_f1_pct=100.0),
        iterations=2,
        artifact_path="artifact.pt",
        notes="Unit test result.",
        execution_target="cpu",
    )
    console = Console(record=True, width=120)
    csv_path = tmp_path / "report.csv"
    html_path = tmp_path / "report.html"

    render_results(console, [result])
    save_results_csv([result], csv_path)
    save_results_html([result], html_path, manifest={"q_lab_version": "1.3.0"})

    exported_text = console.export_text()
    dataframe = pd.read_csv(csv_path)
    exported_html = html_path.read_text(encoding="utf-8")

    assert "Q-Lab Benchmark Report" in exported_text
    assert "baseline" in exported_text
    assert "pytorch" in exported_text
    assert "cpu" in exported_text
    assert "812.50" in exported_text
    assert csv_path.exists()
    assert html_path.exists()
    assert dataframe.loc[0, "label"] == "baseline"
    assert dataframe.loc[0, "batch_size"] == 1
    assert dataframe.loc[0, "execution_target"] == "cpu"
    assert "throughput_items_per_sec" in dataframe.columns
    assert "peak_memory_mb" in dataframe.columns
    assert "eval_top1_accuracy_pct" in dataframe.columns
    assert "Q-Lab Benchmark Report" in exported_html
    assert "execution_target" in exported_html
    assert "q_lab_version" in exported_html
