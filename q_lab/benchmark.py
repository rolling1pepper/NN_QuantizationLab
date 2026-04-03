from __future__ import annotations

import statistics
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence, Tuple

import torch
from torch import Tensor, nn

from q_lab.evaluation import compute_classification_metrics, extract_logits
from q_lab.onnx_utils import build_onnx_input_feed
from q_lab.types import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkStats,
    EvaluationMetrics,
    FidelityMetrics,
    InvocationMode,
    LoadedModel,
    RuntimeBackend,
)

try:
    import psutil
except ImportError:  # pragma: no cover - optional runtime metric helper
    psutil = None


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


def infer_batch_size(example_inputs: Sequence[Tensor]) -> int:
    if not example_inputs:
        return 1
    first_input = example_inputs[0]
    if isinstance(first_input, Tensor) and first_input.ndim > 0 and first_input.shape[0] > 0:
        return int(first_input.shape[0])
    return 1


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
        benchmark_input_batches: Sequence[Sequence[Tensor]] | None = None,
        fidelity_input_batches: Sequence[Sequence[Tensor]] | None = None,
        evaluation_input_batches: Sequence[Sequence[Tensor]] | None = None,
        evaluation_label_batches: Sequence[Tensor] | None = None,
    ) -> BenchmarkResult:
        model = model.to(self.config.device)
        model.eval()
        benchmark_batches = self._prepare_pytorch_batches(
            device=self.config.device,
            input_batches=benchmark_input_batches or (loaded_model.example_inputs,),
        )
        stats, _ = self._run_pytorch_loop(model, loaded_model, benchmark_batches)
        fidelity = (
            FidelityMetrics(accuracy_proxy_pct=100.0, cosine_similarity=1.0, max_abs_diff=0.0)
            if reference_outputs is None
            else compare_outputs(
                reference_outputs,
                self.capture_reference_outputs(
                    model=model,
                    loaded_model=loaded_model,
                    input_batches=fidelity_input_batches or benchmark_batches,
                ),
            )
        )
        evaluation = self.evaluate_pytorch(
            model=model,
            loaded_model=loaded_model,
            input_batches=evaluation_input_batches,
            label_batches=evaluation_label_batches,
        )
        return BenchmarkResult(
            label=label,
            backend=RuntimeBackend.PYTORCH,
            model_name=loaded_model.display_name,
            source=loaded_model.source,
            batch_size=self._infer_result_batch_size(benchmark_batches),
            quantization=quantization,
            pruning=pruning,
            pruning_amount=pruning_amount,
            stats=stats,
            size_mb=size_mb,
            sparsity_pct=sparsity_pct,
            fidelity=fidelity,
            evaluation=evaluation,
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
        benchmark_input_batches: Sequence[Sequence[Tensor]] | None = None,
        fidelity_input_batches: Sequence[Sequence[Tensor]] | None = None,
        evaluation_input_batches: Sequence[Sequence[Tensor]] | None = None,
        evaluation_label_batches: Sequence[Tensor] | None = None,
    ) -> BenchmarkResult:
        input_batches = tuple(benchmark_input_batches or (loaded_model.example_inputs,))
        input_feeds = tuple(
            build_onnx_input_feed(
                input_names=loaded_model.input_names,
                example_inputs=batch,
            )
            for batch in input_batches
        )
        stats, _ = self._run_onnx_loop(session=session, input_feeds=input_feeds)
        fidelity = (
            FidelityMetrics(accuracy_proxy_pct=100.0, cosine_similarity=1.0, max_abs_diff=0.0)
            if reference_outputs is None
            else compare_outputs(
                reference_outputs,
                self.capture_onnx_reference_outputs(
                    session=session,
                    loaded_model=loaded_model,
                    input_batches=fidelity_input_batches or input_batches,
                ),
            )
        )
        evaluation = self.evaluate_onnx(
            session=session,
            loaded_model=loaded_model,
            input_batches=evaluation_input_batches,
            label_batches=evaluation_label_batches,
        )
        return BenchmarkResult(
            label=label,
            backend=RuntimeBackend.ONNX_RUNTIME,
            model_name=loaded_model.display_name,
            source="onnx",
            batch_size=self._infer_result_batch_size(input_batches),
            quantization=quantization,
            pruning=pruning,
            pruning_amount=pruning_amount,
            stats=stats,
            size_mb=size_mb,
            sparsity_pct=sparsity_pct,
            fidelity=fidelity,
            evaluation=evaluation,
            iterations=self.config.benchmark_iterations,
            artifact_path=artifact_path,
            notes=notes,
            execution_target=",".join(session.get_providers()),
        )

    def capture_reference_outputs(
        self,
        model: nn.Module,
        loaded_model: LoadedModel,
        input_batches: Sequence[Sequence[Tensor]] | None = None,
    ) -> Tuple[Tensor, ...]:
        model = model.to(self.config.device)
        model.eval()
        prepared_batches = self._prepare_pytorch_batches(
            device=self.config.device,
            input_batches=input_batches or (loaded_model.example_inputs,),
        )
        outputs = []
        with torch.inference_mode():
            for example_inputs in prepared_batches:
                outputs.append(
                    invoke_model(
                        model=model,
                        example_inputs=example_inputs,
                        input_names=loaded_model.input_names,
                        invocation_mode=loaded_model.invocation_mode,
                    )
                )
        if len(outputs) == 1:
            return canonicalize_outputs(outputs[0])
        return canonicalize_outputs(outputs)

    def evaluate_pytorch(
        self,
        model: nn.Module,
        loaded_model: LoadedModel,
        input_batches: Sequence[Sequence[Tensor]] | None,
        label_batches: Sequence[Tensor] | None,
    ) -> EvaluationMetrics:
        if not input_batches or not label_batches:
            return EvaluationMetrics()

        prepared_batches = self._prepare_pytorch_batches(
            device=self.config.device,
            input_batches=input_batches,
        )
        raw_outputs = []
        with torch.inference_mode():
            for batch in prepared_batches:
                raw_outputs.append(
                    invoke_model(
                        model=model,
                        example_inputs=batch,
                        input_names=loaded_model.input_names,
                        invocation_mode=loaded_model.invocation_mode,
                    )
                )
        return self._compute_evaluation_metrics(raw_outputs, label_batches)

    def capture_onnx_reference_outputs(
        self,
        session: Any,
        loaded_model: LoadedModel,
        input_batches: Sequence[Sequence[Tensor]] | None = None,
    ) -> Tuple[Tensor, ...]:
        batches = tuple(input_batches or (loaded_model.example_inputs,))
        outputs = []
        for batch in batches:
            input_feed = build_onnx_input_feed(
                input_names=loaded_model.input_names,
                example_inputs=batch,
            )
            outputs.append(session.run(None, input_feed))
        if len(outputs) == 1:
            return canonicalize_outputs(outputs[0])
        return canonicalize_outputs(outputs)

    def evaluate_onnx(
        self,
        session: Any,
        loaded_model: LoadedModel,
        input_batches: Sequence[Sequence[Tensor]] | None,
        label_batches: Sequence[Tensor] | None,
    ) -> EvaluationMetrics:
        if not input_batches or not label_batches:
            return EvaluationMetrics()

        raw_outputs = []
        for batch in input_batches:
            input_feed = build_onnx_input_feed(
                input_names=loaded_model.input_names,
                example_inputs=batch,
            )
            raw_outputs.append(session.run(None, input_feed))
        return self._compute_evaluation_metrics(raw_outputs, label_batches)

    def _run_pytorch_loop(
        self,
        model: nn.Module,
        loaded_model: LoadedModel,
        input_batches: Sequence[Sequence[Tensor]],
    ) -> Tuple[BenchmarkStats, Tuple[Tensor, ...]]:
        self._synchronize_if_needed()
        self._reset_peak_memory_tracking()
        baseline_memory_bytes = self._current_process_memory_bytes()
        peak_memory_bytes = baseline_memory_bytes
        latest_outputs: Any = tuple()
        with torch.inference_mode():
            for _ in range(self.config.warmup_iterations):
                example_inputs = tuple(input_batches[0])
                latest_outputs = invoke_model(
                    model=model,
                    example_inputs=example_inputs,
                    input_names=loaded_model.input_names,
                    invocation_mode=loaded_model.invocation_mode,
                )
                self._synchronize_if_needed()

            timings_ms = []
            processed_items = 0
            for _ in range(self.config.benchmark_iterations):
                example_inputs = tuple(
                    input_batches[_ % len(input_batches)]
                )
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
                processed_items += infer_batch_size(example_inputs)
                peak_memory_bytes = self._update_peak_process_memory(peak_memory_bytes)

        return (
            self._build_stats(
                timings_ms=timings_ms,
                processed_items=processed_items,
                baseline_memory_bytes=baseline_memory_bytes,
                peak_memory_bytes=peak_memory_bytes,
            ),
            canonicalize_outputs(latest_outputs),
        )

    def _run_onnx_loop(
        self,
        session: Any,
        input_feeds: Sequence[Mapping[str, Any]],
    ) -> Tuple[BenchmarkStats, Tuple[Tensor, ...]]:
        baseline_memory_bytes = self._current_process_memory_bytes()
        peak_memory_bytes = baseline_memory_bytes
        latest_outputs: Any = tuple()
        for _ in range(self.config.warmup_iterations):
            latest_outputs = session.run(None, input_feeds[0])

        timings_ms = []
        processed_items = 0
        for _ in range(self.config.benchmark_iterations):
            input_feed = input_feeds[_ % len(input_feeds)]
            start = perf_counter()
            latest_outputs = session.run(None, input_feed)
            timings_ms.append((perf_counter() - start) * 1000.0)
            processed_items += self._infer_feed_batch_size(input_feed)
            peak_memory_bytes = self._update_peak_process_memory(peak_memory_bytes)

        return (
            self._build_stats(
                timings_ms=timings_ms,
                processed_items=processed_items,
                baseline_memory_bytes=baseline_memory_bytes,
                peak_memory_bytes=peak_memory_bytes,
            ),
            canonicalize_outputs(latest_outputs),
        )

    def _build_stats(
        self,
        timings_ms: Iterable[float],
        processed_items: int,
        baseline_memory_bytes: int | None,
        peak_memory_bytes: int | None,
    ) -> BenchmarkStats:
        values = list(timings_ms)
        total_elapsed_ms = sum(values)
        throughput_items_per_sec = None
        if total_elapsed_ms > 0.0 and processed_items > 0:
            throughput_items_per_sec = processed_items / (total_elapsed_ms / 1000.0)

        peak_memory_mb = None
        cuda_peak_bytes = self._cuda_peak_memory_bytes()
        if cuda_peak_bytes is not None:
            peak_memory_mb = cuda_peak_bytes / (1024 * 1024)
        elif baseline_memory_bytes is not None and peak_memory_bytes is not None:
            peak_memory_mb = max(0, peak_memory_bytes - baseline_memory_bytes) / (1024 * 1024)
        return BenchmarkStats(
            mean_latency_ms=statistics.mean(values),
            std_latency_ms=statistics.pstdev(values) if len(values) > 1 else 0.0,
            p95_latency_ms=self._compute_percentile(values, 0.95),
            throughput_items_per_sec=throughput_items_per_sec,
            peak_memory_mb=peak_memory_mb,
        )

    @staticmethod
    def _compute_evaluation_metrics(
        raw_outputs: Sequence[Any],
        label_batches: Sequence[Tensor],
    ) -> EvaluationMetrics:
        logits_batches = []
        for output in raw_outputs:
            logits = extract_logits(output)
            if logits is None:
                continue
            logits_batches.append(logits)
        if len(logits_batches) != len(label_batches):
            return EvaluationMetrics()
        return compute_classification_metrics(
            logits_batches=logits_batches,
            label_batches=label_batches,
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

    def _prepare_pytorch_batches(
        self,
        device: str,
        input_batches: Sequence[Sequence[Tensor]],
    ) -> Tuple[Tuple[Tensor, ...], ...]:
        return tuple(
            clone_inputs_to_device(batch, device)
            for batch in input_batches
        )

    @staticmethod
    def _infer_result_batch_size(
        input_batches: Sequence[Sequence[Tensor]],
    ) -> int:
        if not input_batches:
            return 1
        return infer_batch_size(tuple(input_batches[0]))

    @staticmethod
    def _infer_feed_batch_size(input_feed: Mapping[str, Any]) -> int:
        for value in input_feed.values():
            shape = getattr(value, "shape", ())
            if len(shape) > 0:
                return int(shape[0])
        return 1

    def _reset_peak_memory_tracking(self) -> None:
        if self.config.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def _cuda_peak_memory_bytes(self) -> int | None:
        if self.config.device.startswith("cuda") and torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated())
        return None

    @staticmethod
    def _current_process_memory_bytes() -> int | None:
        if psutil is None:
            return None
        return int(psutil.Process().memory_info().rss)

    def _update_peak_process_memory(self, current_peak: int | None) -> int | None:
        sampled_memory = self._current_process_memory_bytes()
        if sampled_memory is None:
            return current_peak
        if current_peak is None:
            return sampled_memory
        return max(current_peak, sampled_memory)

    @staticmethod
    def _compute_percentile(values: Sequence[float], quantile: float) -> float:
        if not values:
            raise ValueError("Cannot compute percentile of an empty list.")
        sorted_values = sorted(values)
        index = min(int(round((len(sorted_values) - 1) * quantile)), len(sorted_values) - 1)
        return sorted_values[index]
