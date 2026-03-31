from __future__ import annotations

import statistics
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence, Tuple

import torch
from torch import Tensor, nn

from q_lab.onnx_utils import build_onnx_input_feed
from q_lab.types import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkStats,
    FidelityMetrics,
    InvocationMode,
    LoadedModel,
    RuntimeBackend,
)


def canonicalize_outputs(output: Any) -> Tuple[Tensor, ...]:
    if output is None:
        return tuple()
    if isinstance(output, Tensor):
        return (output.detach().cpu(),)
    if hasattr(output, "to_tuple"):
        return canonicalize_outputs(output.to_tuple())
    if isinstance(output, Mapping):
        flattened: list[Tensor] = []
        for key in sorted(output):
            flattened.extend(canonicalize_outputs(output[key]))
        return tuple(flattened)
    if hasattr(output, "__dataclass_fields__"):
        flattened = []
        for field_name in output.__dataclass_fields__:
            flattened.extend(canonicalize_outputs(getattr(output, field_name)))
        return tuple(flattened)
    if isinstance(output, (list, tuple)):
        flattened = []
        for item in output:
            flattened.extend(canonicalize_outputs(item))
        return tuple(flattened)
    try:
        tensor = torch.as_tensor(output)
    except (RuntimeError, TypeError, ValueError):
        return tuple()
    return (tensor.detach().cpu(),)


def clone_inputs_to_device(inputs: Sequence[Tensor], device: str) -> Tuple[Tensor, ...]:
    return tuple(tensor.detach().clone().to(device) for tensor in inputs)


def invoke_model(
    model: nn.Module,
    example_inputs: Sequence[Tensor],
    input_names: Sequence[str],
    invocation_mode: InvocationMode,
) -> Any:
    if invocation_mode is InvocationMode.KEYWORD:
        kwargs = dict(zip(input_names, example_inputs))
        return model(**kwargs)
    return model(*example_inputs)


def compare_outputs(reference: Any, candidate: Any) -> FidelityMetrics:
    reference_tensors = canonicalize_outputs(reference)
    candidate_tensors = canonicalize_outputs(candidate)
    comparable_pairs = [
        (ref.float().reshape(-1), cand.float().reshape(-1))
        for ref, cand in zip(reference_tensors, candidate_tensors)
        if ref.numel() == cand.numel()
    ]
    if not comparable_pairs:
        return FidelityMetrics(
            accuracy_proxy_pct=None,
            cosine_similarity=None,
            max_abs_diff=None,
        )

    cosine_values = []
    max_abs_diffs = []
    for reference_flat, candidate_flat in comparable_pairs:
        reference_norm = reference_flat.norm()
        candidate_norm = candidate_flat.norm()
        if reference_norm.item() == 0.0 and candidate_norm.item() == 0.0:
            cosine_values.append(1.0)
        elif reference_norm.item() == 0.0 or candidate_norm.item() == 0.0:
            cosine_values.append(0.0)
        else:
            cosine_values.append(
                torch.nn.functional.cosine_similarity(
                    reference_flat.unsqueeze(0),
                    candidate_flat.unsqueeze(0),
                ).item()
            )
        max_abs_diffs.append(torch.max(torch.abs(reference_flat - candidate_flat)).item())

    accuracy_proxy_pct = None
    for ref_tensor, candidate_tensor in zip(reference_tensors, candidate_tensors):
        if ref_tensor.shape == candidate_tensor.shape and ref_tensor.ndim >= 2:
            reference_prediction = ref_tensor.argmax(dim=-1)
            candidate_prediction = candidate_tensor.argmax(dim=-1)
            accuracy_proxy_pct = (
                reference_prediction.eq(candidate_prediction).float().mean().item() * 100.0
            )
            break

    return FidelityMetrics(
        accuracy_proxy_pct=accuracy_proxy_pct,
        cosine_similarity=sum(cosine_values) / len(cosine_values),
        max_abs_diff=max(max_abs_diffs),
    )


class BenchmarkEngine:
    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config

    def benchmark_pytorch(
        self,
        label: str,
        model: nn.Module,
        loaded_model: LoadedModel,
        size_mb: float,
        quantization: str,
        pruning: str,
        pruning_amount: float,
        sparsity_pct: float,
        reference_outputs: Any | None = None,
        artifact_path: str | None = None,
        notes: str = "",
    ) -> BenchmarkResult:
        model = model.to(self.config.device)
        model.eval()
        example_inputs = clone_inputs_to_device(
            loaded_model.example_inputs,
            self.config.device,
        )
        stats, outputs = self._run_pytorch_loop(model, loaded_model, example_inputs)
        fidelity = (
            FidelityMetrics(accuracy_proxy_pct=100.0, cosine_similarity=1.0, max_abs_diff=0.0)
            if reference_outputs is None
            else compare_outputs(reference_outputs, outputs)
        )
        return BenchmarkResult(
            label=label,
            backend=RuntimeBackend.PYTORCH,
            model_name=loaded_model.display_name,
            source=loaded_model.source,
            quantization=quantization,
            pruning=pruning,
            pruning_amount=pruning_amount,
            stats=stats,
            size_mb=size_mb,
            sparsity_pct=sparsity_pct,
            fidelity=fidelity,
            iterations=self.config.benchmark_iterations,
            artifact_path=artifact_path,
            notes=notes,
            execution_target=self.config.device,
        )

    def benchmark_onnx(
        self,
        label: str,
        session: Any,
        loaded_model: LoadedModel,
        size_mb: float,
        quantization: str,
        pruning: str,
        pruning_amount: float,
        sparsity_pct: float,
        reference_outputs: Any | None,
        artifact_path: str,
        notes: str = "",
    ) -> BenchmarkResult:
        input_feed = build_onnx_input_feed(
            input_names=loaded_model.input_names,
            example_inputs=loaded_model.example_inputs,
        )
        stats, outputs = self._run_onnx_loop(session=session, input_feed=input_feed)
        fidelity = (
            FidelityMetrics(accuracy_proxy_pct=100.0, cosine_similarity=1.0, max_abs_diff=0.0)
            if reference_outputs is None
            else compare_outputs(reference_outputs, outputs)
        )
        return BenchmarkResult(
            label=label,
            backend=RuntimeBackend.ONNX_RUNTIME,
            model_name=loaded_model.display_name,
            source="onnx",
            quantization=quantization,
            pruning=pruning,
            pruning_amount=pruning_amount,
            stats=stats,
            size_mb=size_mb,
            sparsity_pct=sparsity_pct,
            fidelity=fidelity,
            iterations=self.config.benchmark_iterations,
            artifact_path=artifact_path,
            notes=notes,
            execution_target=",".join(session.get_providers()),
        )

    def capture_reference_outputs(
        self,
        model: nn.Module,
        loaded_model: LoadedModel,
    ) -> Tuple[Tensor, ...]:
        model = model.to(self.config.device)
        model.eval()
        example_inputs = clone_inputs_to_device(
            loaded_model.example_inputs,
            self.config.device,
        )
        with torch.inference_mode():
            outputs = invoke_model(
                model=model,
                example_inputs=example_inputs,
                input_names=loaded_model.input_names,
                invocation_mode=loaded_model.invocation_mode,
            )
        return canonicalize_outputs(outputs)

    def capture_onnx_reference_outputs(
        self,
        session: Any,
        loaded_model: LoadedModel,
    ) -> Tuple[Tensor, ...]:
        input_feed = build_onnx_input_feed(
            input_names=loaded_model.input_names,
            example_inputs=loaded_model.example_inputs,
        )
        outputs = session.run(None, input_feed)
        return canonicalize_outputs(outputs)

    def _run_pytorch_loop(
        self,
        model: nn.Module,
        loaded_model: LoadedModel,
        example_inputs: Sequence[Tensor],
    ) -> Tuple[BenchmarkStats, Tuple[Tensor, ...]]:
        self._synchronize_if_needed()
        latest_outputs: Any = tuple()
        with torch.inference_mode():
            for _ in range(self.config.warmup_iterations):
                latest_outputs = invoke_model(
                    model=model,
                    example_inputs=example_inputs,
                    input_names=loaded_model.input_names,
                    invocation_mode=loaded_model.invocation_mode,
                )
                self._synchronize_if_needed()

            timings_ms = []
            for _ in range(self.config.benchmark_iterations):
                self._synchronize_if_needed()
                start = perf_counter()
                latest_outputs = invoke_model(
                    model=model,
                    example_inputs=example_inputs,
                    input_names=loaded_model.input_names,
                    invocation_mode=loaded_model.invocation_mode,
                )
                self._synchronize_if_needed()
                timings_ms.append((perf_counter() - start) * 1000.0)

        return self._build_stats(timings_ms), canonicalize_outputs(latest_outputs)

    def _run_onnx_loop(
        self,
        session: Any,
        input_feed: Mapping[str, Any],
    ) -> Tuple[BenchmarkStats, Tuple[Tensor, ...]]:
        latest_outputs: Any = tuple()
        for _ in range(self.config.warmup_iterations):
            latest_outputs = session.run(None, input_feed)

        timings_ms = []
        for _ in range(self.config.benchmark_iterations):
            start = perf_counter()
            latest_outputs = session.run(None, input_feed)
            timings_ms.append((perf_counter() - start) * 1000.0)

        return self._build_stats(timings_ms), canonicalize_outputs(latest_outputs)

    def _build_stats(self, timings_ms: Iterable[float]) -> BenchmarkStats:
        values = list(timings_ms)
        return BenchmarkStats(
            mean_latency_ms=statistics.mean(values),
            std_latency_ms=statistics.pstdev(values) if len(values) > 1 else 0.0,
            p95_latency_ms=self._compute_percentile(values, 0.95),
        )

    @staticmethod
    def measure_pytorch_size_mb(model: nn.Module) -> float:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = Path(temp_dir) / "artifact.pt"
            if isinstance(model, torch.jit.ScriptModule):
                torch.jit.save(model, artifact_path.as_posix())
            else:
                torch.save(model.state_dict(), artifact_path)
            return artifact_path.stat().st_size / (1024 * 1024)

    @staticmethod
    def measure_file_size_mb(path: Path) -> float:
        return path.stat().st_size / (1024 * 1024)

    def _synchronize_if_needed(self) -> None:
        if self.config.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    @staticmethod
    def _compute_percentile(values: Sequence[float], quantile: float) -> float:
        if not values:
            raise ValueError("Cannot compute percentile of an empty list.")
        sorted_values = sorted(values)
        index = min(int(round((len(sorted_values) - 1) * quantile)), len(sorted_values) - 1)
        return sorted_values[index]
