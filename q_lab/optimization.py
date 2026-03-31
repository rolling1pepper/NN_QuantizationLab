from __future__ import annotations

import copy
from typing import Iterable, Tuple

import torch
import torch.nn.utils.prune as prune
from torch import Tensor, nn
from torch.ao.quantization import QConfigMapping, get_default_qconfig
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

from q_lab.benchmark import clone_inputs_to_device
from q_lab.onnx_utils import quantize_onnx_artifact
from q_lab.types import (
    BenchmarkConfig,
    CompressionConfig,
    LoadedModel,
    ModelFormat,
    ModelFamily,
    OptimizationOutcome,
    PruningMode,
    QuantizationMode,
)

PRUNABLE_MODULES = (nn.Conv2d, nn.Linear)


def optimize_model(
    loaded_model: LoadedModel,
    compression: CompressionConfig,
    benchmark_config: BenchmarkConfig,
) -> OptimizationOutcome:
    if loaded_model.format is ModelFormat.ONNX:
        return quantize_onnx_artifact(
            loaded_model=loaded_model,
            compression=compression,
            benchmark_config=benchmark_config,
        )

    if (
        loaded_model.source == "torchscript"
        and (
            compression.quantization is not QuantizationMode.NONE
            or compression.pruning is not PruningMode.NONE
        )
    ):
        raise ValueError(
            "Quantization and pruning are supported only for eager PyTorch models. "
            "TorchScript inputs can be benchmarked and exported to ONNX as-is."
        )

    working_model = copy.deepcopy(loaded_model.model).cpu().eval()
    notes: list[str] = []
    sparsity_pct = 0.0

    if compression.pruning is not PruningMode.NONE:
        working_model, sparsity_pct = apply_pruning(
            model=working_model,
            mode=compression.pruning,
            amount=compression.pruning_amount,
        )
        notes.append(
            "Pruning creates sparsity masks; serialized model size may stay close to the baseline."
        )

    if compression.quantization is QuantizationMode.DYNAMIC:
        working_model = apply_dynamic_quantization(working_model)
        notes.append("Dynamic quantization applied to Linear layers.")
    elif compression.quantization is QuantizationMode.STATIC:
        if loaded_model.family is not ModelFamily.VISION:
            raise ValueError("Static quantization is currently restricted to vision models.")
        working_model = apply_static_quantization(
            model=working_model,
            example_inputs=clone_inputs_to_device(loaded_model.example_inputs, "cpu"),
            calibration_iterations=benchmark_config.calibration_iterations,
        )
        notes.append(
            "Static FX quantization calibrated on synthetic inputs. Use a real calibration dataset for production accuracy."
        )

    return OptimizationOutcome(
        model=working_model,
        quantization=compression.quantization,
        pruning=compression.pruning,
        pruning_amount=compression.pruning_amount,
        sparsity_pct=sparsity_pct,
        notes=notes,
    )


def apply_pruning(
    model: nn.Module,
    mode: PruningMode,
    amount: float,
) -> Tuple[nn.Module, float]:
    modules = list(iter_prunable_modules(model))
    if not modules:
        raise ValueError("No prunable Conv2d or Linear layers were found in the model.")

    for _, module in modules:
        if mode is PruningMode.UNSTRUCTURED:
            prune.l1_unstructured(module, name="weight", amount=amount)
        elif mode is PruningMode.STRUCTURED:
            prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
        else:
            raise ValueError(f"Unsupported pruning mode '{mode.value}'.")
        prune.remove(module, "weight")

    return model, compute_parameter_sparsity_pct(model)


def apply_dynamic_quantization(model: nn.Module) -> nn.Module:
    return torch.ao.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
        inplace=False,
    )


def apply_static_quantization(
    model: nn.Module,
    example_inputs: Tuple[Tensor, ...],
    calibration_iterations: int,
) -> nn.Module:
    backend = _select_quantization_backend()
    torch.backends.quantized.engine = backend
    qconfig_mapping = QConfigMapping().set_global(get_default_qconfig(backend))
    prepared = prepare_fx(model, qconfig_mapping, example_inputs)
    with torch.inference_mode():
        for _ in range(calibration_iterations):
            prepared(*example_inputs)
    return convert_fx(prepared)


def iter_prunable_modules(model: nn.Module) -> Iterable[Tuple[str, nn.Module]]:
    for module_name, module in model.named_modules():
        if isinstance(module, PRUNABLE_MODULES) and getattr(module, "weight", None) is not None:
            yield module_name, module


def compute_parameter_sparsity_pct(model: nn.Module) -> float:
    total_params = 0
    zero_params = 0
    for parameter in model.parameters():
        total_params += parameter.numel()
        zero_params += int(parameter.eq(0).sum().item())
    if total_params == 0:
        return 0.0
    return (zero_params / total_params) * 100.0


def _select_quantization_backend() -> str:
    supported_backends = torch.backends.quantized.supported_engines
    for backend in ("x86", "fbgemm", "qnnpack"):
        if backend in supported_backends:
            return backend
    raise RuntimeError(
        "No supported quantization backend was found. "
        f"Available backends: {supported_backends}"
    )
