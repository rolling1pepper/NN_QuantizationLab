from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


class ModelFamily(str, Enum):
    VISION = "vision"
    TEXT = "text"


class InvocationMode(str, Enum):
    POSITIONAL = "positional"
    KEYWORD = "keyword"


class QuantizationMode(str, Enum):
    NONE = "none"
    STATIC = "static"
    DYNAMIC = "dynamic"


class PruningMode(str, Enum):
    NONE = "none"
    STRUCTURED = "structured"
    UNSTRUCTURED = "unstructured"


class RuntimeBackend(str, Enum):
    PYTORCH = "pytorch"
    ONNX_RUNTIME = "onnxruntime"


class ModelFormat(str, Enum):
    PYTORCH = "pytorch"
    ONNX = "onnx"


@dataclass(frozen=True)
class InputTensorSpec:
    name: str
    dtype: str
    shape: Tuple[Any, ...]


@dataclass(frozen=True)
class BenchmarkConfig:
    warmup_iterations: int
    benchmark_iterations: int
    calibration_iterations: int
    batch_size: int
    device: str
    onnx_providers: Tuple[str, ...] = ("CPUExecutionProvider",)
    onnx_optimization_level: str = "extended"

    def __post_init__(self) -> None:
        if self.warmup_iterations < 0:
            raise ValueError("Warmup iterations must be >= 0.")
        if self.benchmark_iterations <= 0:
            raise ValueError("Benchmark iterations must be > 0.")
        if self.calibration_iterations <= 0:
            raise ValueError("Calibration iterations must be > 0.")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be > 0.")
        if not self.onnx_providers:
            raise ValueError("At least one ONNX Runtime provider must be configured.")
        if self.onnx_optimization_level not in {"disable", "basic", "extended", "all"}:
            raise ValueError("ONNX optimization level must be one of disable/basic/extended/all.")


@dataclass(frozen=True)
class CompressionConfig:
    quantization: QuantizationMode
    pruning: PruningMode
    pruning_amount: float
    export_onnx: bool
    onnx_path: Optional[Path]
    onnx_opset: int
    onnx_quantized_path: Optional[Path] = None
    onnx_quant_format: str = "qdq"
    preprocess_onnx: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.pruning_amount < 1.0:
            raise ValueError("Pruning amount must be within [0.0, 1.0).")
        if self.onnx_opset < 13:
            raise ValueError("ONNX opset must be >= 13.")
        if self.onnx_quant_format not in {"qdq", "qoperator"}:
            raise ValueError("ONNX quantization format must be 'qdq' or 'qoperator'.")


@dataclass(frozen=True)
class InputConfig:
    family: ModelFamily
    batch_size: int
    image_shape: Tuple[int, int, int]
    sequence_length: int
    vocab_size: int

    def __post_init__(self) -> None:
        channels, height, width = self.image_shape
        if channels <= 0 or height <= 0 or width <= 0:
            raise ValueError("Image shape values must be > 0.")
        if self.sequence_length <= 0:
            raise ValueError("Sequence length must be > 0.")
        if self.vocab_size <= 1:
            raise ValueError("Vocab size must be > 1.")


@dataclass
class LoadedModel:
    identifier: str
    display_name: str
    source: str
    model: Any
    family: ModelFamily
    invocation_mode: InvocationMode
    input_names: Tuple[str, ...]
    example_inputs: Tuple[Any, ...]
    format: ModelFormat = ModelFormat.PYTORCH
    input_specs: Tuple[InputTensorSpec, ...] = tuple()
    original_path: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FidelityMetrics:
    accuracy_proxy_pct: Optional[float]
    cosine_similarity: Optional[float]
    max_abs_diff: Optional[float]


@dataclass(frozen=True)
class EvaluationMetrics:
    sample_count: int = 0
    top1_accuracy_pct: Optional[float] = None
    macro_f1_pct: Optional[float] = None


@dataclass(frozen=True)
class BenchmarkStats:
    mean_latency_ms: float
    std_latency_ms: float
    p95_latency_ms: float
    throughput_items_per_sec: float | None = None
    peak_memory_mb: float | None = None


@dataclass
class OptimizationOutcome:
    model: Any
    quantization: QuantizationMode
    pruning: PruningMode
    pruning_amount: float
    sparsity_pct: float
    notes: list[str] = field(default_factory=list)
    artifact_path: Optional[Path] = None


@dataclass(frozen=True)
class InputDataset:
    batches: Tuple[Tuple[Any, ...], ...]
    label_batches: Tuple[Any, ...] = tuple()
    source_format: str = "synthetic"


@dataclass(frozen=True)
class BenchmarkResult:
    label: str
    backend: RuntimeBackend
    model_name: str
    source: str
    batch_size: int
    quantization: str
    pruning: str
    pruning_amount: float
    stats: BenchmarkStats
    size_mb: float
    sparsity_pct: float
    fidelity: FidelityMetrics
    iterations: int
    artifact_path: Optional[str]
    notes: str
    execution_target: str = ""
    evaluation: EvaluationMetrics = field(default_factory=EvaluationMetrics)

    def to_record(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "backend": self.backend.value,
            "model_name": self.model_name,
            "source": self.source,
            "batch_size": self.batch_size,
            "quantization": self.quantization,
            "pruning": self.pruning,
            "pruning_amount": round(self.pruning_amount, 4),
            "mean_latency_ms": round(self.stats.mean_latency_ms, 4),
            "std_latency_ms": round(self.stats.std_latency_ms, 4),
            "p95_latency_ms": round(self.stats.p95_latency_ms, 4),
            "throughput_items_per_sec": (
                None
                if self.stats.throughput_items_per_sec is None
                else round(self.stats.throughput_items_per_sec, 4)
            ),
            "peak_memory_mb": (
                None
                if self.stats.peak_memory_mb is None
                else round(self.stats.peak_memory_mb, 4)
            ),
            "size_mb": round(self.size_mb, 4),
            "sparsity_pct": round(self.sparsity_pct, 4),
            "accuracy_proxy_pct": (
                None
                if self.fidelity.accuracy_proxy_pct is None
                else round(self.fidelity.accuracy_proxy_pct, 4)
            ),
            "eval_sample_count": self.evaluation.sample_count,
            "eval_top1_accuracy_pct": (
                None
                if self.evaluation.top1_accuracy_pct is None
                else round(self.evaluation.top1_accuracy_pct, 4)
            ),
            "eval_macro_f1_pct": (
                None
                if self.evaluation.macro_f1_pct is None
                else round(self.evaluation.macro_f1_pct, 4)
            ),
            "cosine_similarity": (
                None
                if self.fidelity.cosine_similarity is None
                else round(self.fidelity.cosine_similarity, 6)
            ),
            "max_abs_diff": (
                None
                if self.fidelity.max_abs_diff is None
                else round(self.fidelity.max_abs_diff, 6)
            ),
            "iterations": self.iterations,
            "artifact_path": self.artifact_path or "",
            "notes": self.notes,
            "execution_target": self.execution_target,
        }
