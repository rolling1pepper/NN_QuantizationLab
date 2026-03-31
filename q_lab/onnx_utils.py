from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from torch import Tensor, nn

from q_lab.types import (
    BenchmarkConfig,
    CompressionConfig,
    InputConfig,
    InputTensorSpec,
    InvocationMode,
    LoadedModel,
    ModelFamily,
    ModelFormat,
    OptimizationOutcome,
    PruningMode,
    QuantizationMode,
)

try:
    from onnxruntime.quantization.shape_inference import quant_pre_process
except ImportError:  # pragma: no cover - optional helper depends on ORT packaging
    quant_pre_process = None


TEXT_MODEL_HINTS = ("bert", "distilbert", "roberta", "gpt", "llama", "t5")
INPUT_NAME_FAMILY_HINTS = {
    "input_ids": ModelFamily.TEXT,
    "attention_mask": ModelFamily.TEXT,
    "token_type_ids": ModelFamily.TEXT,
    "position_ids": ModelFamily.TEXT,
    "pixel_values": ModelFamily.VISION,
    "image": ModelFamily.VISION,
    "input": ModelFamily.VISION,
}
ORT_OPTIMIZATION_LEVELS = {
    "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
}
ORT_TO_TORCH_DTYPE = {
    "tensor(float)": torch.float32,
    "tensor(float16)": torch.float16,
    "tensor(double)": torch.float64,
    "tensor(int64)": torch.int64,
    "tensor(int32)": torch.int32,
    "tensor(int16)": torch.int16,
    "tensor(int8)": torch.int8,
    "tensor(uint8)": torch.uint8,
    "tensor(bool)": torch.bool,
}


def resolve_onnx_providers(
    requested_providers: Sequence[str] | None,
) -> Tuple[str, ...]:
    normalized = tuple(
        provider.strip() for provider in (requested_providers or ("CPUExecutionProvider",)) if provider.strip()
    )
    if not normalized:
        raise ValueError("At least one ONNX Runtime provider must be specified.")

    available = set(ort.get_available_providers())
    missing = [provider for provider in normalized if provider not in available]
    if missing:
        raise ValueError(
            "Requested ONNX Runtime providers are unavailable: "
            f"{', '.join(missing)}. Available providers: {', '.join(sorted(available))}."
        )
    return normalized


def create_onnx_session(
    model_path: Path,
    providers: Sequence[str] | None = None,
    optimization_level: str = "extended",
) -> ort.InferenceSession:
    if optimization_level not in ORT_OPTIMIZATION_LEVELS:
        raise ValueError(
            "Unsupported ONNX Runtime optimization level. "
            "Use one of: disable, basic, extended, all."
        )

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ORT_OPTIMIZATION_LEVELS[optimization_level]
    resolved_providers = resolve_onnx_providers(providers)
    return ort.InferenceSession(
        model_path.as_posix(),
        sess_options=session_options,
        providers=list(resolved_providers),
    )


def extract_input_specs(session: ort.InferenceSession) -> Tuple[InputTensorSpec, ...]:
    return tuple(
        InputTensorSpec(
            name=node_arg.name,
            dtype=node_arg.type,
            shape=tuple(node_arg.shape),
        )
        for node_arg in session.get_inputs()
    )


def infer_onnx_model_family(model_path: Path) -> ModelFamily:
    lowered_name = model_path.stem.lower()
    if any(hint in lowered_name for hint in TEXT_MODEL_HINTS):
        return ModelFamily.TEXT

    session = create_onnx_session(
        model_path=model_path,
        providers=("CPUExecutionProvider",),
        optimization_level="basic",
    )
    return infer_family_from_input_specs(model_path=model_path, input_specs=extract_input_specs(session))


def infer_family_from_input_specs(
    model_path: Path | None,
    input_specs: Sequence[InputTensorSpec],
) -> ModelFamily:
    lowered_name = model_path.stem.lower() if model_path is not None else ""
    if lowered_name and any(hint in lowered_name for hint in TEXT_MODEL_HINTS):
        return ModelFamily.TEXT

    for spec in input_specs:
        family_hint = INPUT_NAME_FAMILY_HINTS.get(spec.name.lower())
        if family_hint is not None:
            return family_hint

    for spec in input_specs:
        rank = len(spec.shape)
        if rank >= 2 and spec.dtype in {"tensor(int64)", "tensor(int32)", "tensor(int16)", "tensor(int8)"}:
            return ModelFamily.TEXT
        if rank == 4 and spec.dtype in {"tensor(float)", "tensor(float16)", "tensor(double)"}:
            return ModelFamily.VISION

    raise ValueError(
        "Unable to infer task type for the ONNX model. "
        "Pass --task vision or --task text explicitly."
    )


def build_onnx_dummy_inputs(
    input_specs: Sequence[InputTensorSpec],
    config: InputConfig,
) -> Tuple[Tuple[Tensor, ...], Tuple[str, ...]]:
    tensors: list[Tensor] = []
    names: list[str] = []
    for spec in input_specs:
        resolved_shape = _resolve_onnx_shape(spec=spec, config=config)
        tensors.append(_build_onnx_tensor(spec=spec, shape=resolved_shape, config=config))
        names.append(spec.name)
    return tuple(tensors), tuple(names)


def build_onnx_input_feed(
    input_names: Sequence[str],
    example_inputs: Sequence[Tensor],
) -> dict[str, np.ndarray]:
    return {
        name: tensor.detach().cpu().numpy()
        for name, tensor in zip(input_names, example_inputs)
    }


def load_onnx_model(
    model_path: Path,
    input_config: InputConfig,
    providers: Sequence[str],
    optimization_level: str,
) -> LoadedModel:
    if model_path.suffix.lower() != ".onnx":
        raise ValueError(f"Expected an ONNX .onnx file, received '{model_path.name}'.")
    if not model_path.is_file():
        raise ValueError(f"Model path '{model_path}' is not a file.")

    session = create_onnx_session(
        model_path=model_path,
        providers=providers,
        optimization_level=optimization_level,
    )
    input_specs = extract_input_specs(session)
    example_inputs, input_names = build_onnx_dummy_inputs(input_specs=input_specs, config=input_config)
    return LoadedModel(
        identifier=model_path.stem,
        display_name=model_path.stem,
        source="onnx",
        model=session,
        family=input_config.family,
        invocation_mode=InvocationMode.POSITIONAL,
        input_names=input_names,
        example_inputs=example_inputs,
        format=ModelFormat.ONNX,
        input_specs=tuple(input_specs),
        original_path=model_path,
        metadata={
            "providers": tuple(session.get_providers()),
            "optimization_level": optimization_level,
        },
    )


def export_to_onnx(
    loaded_model: LoadedModel,
    model: nn.Module,
    destination: Path,
    opset_version: int,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    example_inputs = _move_inputs_to_model_device(
        model=model,
        example_inputs=loaded_model.example_inputs,
    )
    output_names = _infer_export_output_names(
        loaded_model=loaded_model,
        model=model,
        example_inputs=example_inputs,
    )
    dynamic_axes = _build_dynamic_axes(
        family=loaded_model.family,
        input_names=loaded_model.input_names,
        output_names=output_names,
    )
    export_kwargs = {
        "export_params": True,
        "do_constant_folding": True,
        "opset_version": opset_version,
        "input_names": list(loaded_model.input_names),
        "output_names": output_names,
        "dynamic_axes": dynamic_axes,
    }

    if isinstance(model, torch.jit.ScriptModule):
        torch.onnx.export(
            model,
            tuple(example_inputs),
            destination.as_posix(),
            **export_kwargs,
        )
        return destination

    wrapped_model = OnnxExportWrapper(
        model=model,
        input_names=loaded_model.input_names,
        invocation_mode=loaded_model.invocation_mode,
    )
    torch.onnx.export(
        wrapped_model,
        tuple(example_inputs),
        destination.as_posix(),
        **export_kwargs,
    )
    return destination


def quantize_onnx_artifact(
    loaded_model: LoadedModel,
    compression: CompressionConfig,
    benchmark_config: BenchmarkConfig,
) -> OptimizationOutcome:
    if loaded_model.format is not ModelFormat.ONNX:
        raise ValueError("ONNX quantization requires an ONNX input model.")
    if loaded_model.original_path is None:
        raise ValueError("The ONNX input path is missing.")
    if compression.pruning is not PruningMode.NONE:
        raise ValueError("Pruning is not supported for ONNX inputs.")
    if compression.quantization is QuantizationMode.NONE:
        raise ValueError("No ONNX quantization mode was requested.")

    source_path = loaded_model.original_path
    artifact_path = _resolve_onnx_quantized_path(
        source_path=source_path,
        requested_path=compression.onnx_quantized_path,
        quantization=compression.quantization,
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    working_source = source_path
    notes: list[str] = []
    if compression.preprocess_onnx:
        working_source, preprocess_note = _preprocess_onnx_for_quantization(
            source_path=source_path,
            artifact_path=artifact_path,
        )
        if preprocess_note:
            notes.append(preprocess_note)

    if compression.quantization is QuantizationMode.DYNAMIC:
        quantize_dynamic(
            model_input=working_source.as_posix(),
            model_output=artifact_path.as_posix(),
            per_channel=True,
            weight_type=QuantType.QInt8,
        )
        notes.append("ONNX Runtime dynamic quantization applied.")
    elif compression.quantization is QuantizationMode.STATIC:
        calibration_reader = SyntheticCalibrationDataReader(
            input_names=loaded_model.input_names,
            template_inputs=loaded_model.example_inputs,
            calibration_iterations=benchmark_config.calibration_iterations,
        )
        quantize_static(
            model_input=working_source.as_posix(),
            model_output=artifact_path.as_posix(),
            calibration_data_reader=calibration_reader,
            quant_format=_resolve_quant_format(compression.onnx_quant_format),
            per_channel=True,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
        )
        notes.append(
            "ONNX Runtime static quantization calibrated on synthetic inputs. "
            "Use representative production data for accurate activation ranges."
        )

    return OptimizationOutcome(
        model=artifact_path,
        quantization=compression.quantization,
        pruning=PruningMode.NONE,
        pruning_amount=0.0,
        sparsity_pct=0.0,
        notes=notes,
        artifact_path=artifact_path,
    )


class OnnxExportWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        input_names: Sequence[str],
        invocation_mode: InvocationMode,
    ) -> None:
        super().__init__()
        self.model = model
        self.input_names = tuple(input_names)
        self.invocation_mode = invocation_mode

    def forward(self, *inputs: Tensor) -> Tensor | Tuple[Tensor, ...]:
        if self.invocation_mode is InvocationMode.KEYWORD:
            outputs = self.model(**dict(zip(self.input_names, inputs)))
        else:
            outputs = self.model(*inputs)
        flattened = _flatten_export_outputs(outputs)
        if len(flattened) == 1:
            return flattened[0]
        return flattened


class SyntheticCalibrationDataReader(CalibrationDataReader):
    def __init__(
        self,
        input_names: Sequence[str],
        template_inputs: Sequence[Tensor],
        calibration_iterations: int,
    ) -> None:
        self._input_names = tuple(input_names)
        self._template_inputs = tuple(tensor.detach().cpu() for tensor in template_inputs)
        self._calibration_iterations = calibration_iterations
        self._index = 0

    def get_next(self) -> Mapping[str, np.ndarray] | None:
        if self._index >= self._calibration_iterations:
            return None

        sample = {
            name: _build_calibration_tensor(name=name, tensor=tensor, step=self._index).numpy()
            for name, tensor in zip(self._input_names, self._template_inputs)
        }
        self._index += 1
        return sample

    def rewind(self) -> None:
        self._index = 0


def _resolve_onnx_shape(
    spec: InputTensorSpec,
    config: InputConfig,
) -> Tuple[int, ...]:
    resolved: list[int] = []
    rank = len(spec.shape)
    for index, raw_dim in enumerate(spec.shape):
        if isinstance(raw_dim, int) and raw_dim > 0:
            resolved.append(raw_dim)
            continue

        if index == 0:
            resolved.append(config.batch_size)
            continue

        if config.family is ModelFamily.VISION:
            image_dims = config.image_shape
            if rank >= 4 and 1 <= index <= 3:
                resolved.append(image_dims[index - 1])
            else:
                resolved.append(1)
            continue

        if index == 1:
            resolved.append(config.sequence_length)
        else:
            resolved.append(1)
    return tuple(resolved)


def _build_onnx_tensor(
    spec: InputTensorSpec,
    shape: Sequence[int],
    config: InputConfig,
) -> Tensor:
    if spec.dtype not in ORT_TO_TORCH_DTYPE:
        raise ValueError(
            f"Unsupported ONNX input dtype '{spec.dtype}' for synthetic input generation."
        )

    dtype = ORT_TO_TORCH_DTYPE[spec.dtype]
    lowered_name = spec.name.lower()

    if dtype in {torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8}:
        high = config.vocab_size if "ids" in lowered_name else 2
        return torch.randint(low=0, high=max(high, 2), size=tuple(shape), dtype=dtype)
    if dtype is torch.bool:
        return torch.ones(tuple(shape), dtype=torch.bool)
    return torch.randn(tuple(shape), dtype=dtype)


def _move_inputs_to_model_device(
    model: nn.Module,
    example_inputs: Sequence[Tensor],
) -> Tuple[Tensor, ...]:
    try:
        first_tensor = next(model.parameters())
        target_device = first_tensor.device
    except StopIteration:
        try:
            target_device = next(model.buffers()).device
        except StopIteration:
            target_device = torch.device("cpu")
    return tuple(tensor.detach().clone().to(target_device) for tensor in example_inputs)


def _infer_export_output_names(
    loaded_model: LoadedModel,
    model: nn.Module,
    example_inputs: Sequence[Tensor],
) -> list[str]:
    if isinstance(model, torch.jit.ScriptModule):
        with torch.inference_mode():
            outputs = model(*example_inputs)
    elif loaded_model.invocation_mode is InvocationMode.KEYWORD:
        with torch.inference_mode():
            outputs = model(**dict(zip(loaded_model.input_names, example_inputs)))
    else:
        with torch.inference_mode():
            outputs = model(*example_inputs)

    flattened = _flatten_export_outputs(outputs)
    return [f"output_{index}" for index in range(len(flattened))]


def _flatten_export_outputs(outputs: Any) -> Tuple[Tensor, ...]:
    if isinstance(outputs, Tensor):
        return (outputs,)
    if hasattr(outputs, "to_tuple"):
        return _flatten_export_outputs(outputs.to_tuple())
    if isinstance(outputs, Mapping):
        flattened: list[Tensor] = []
        for key in sorted(outputs):
            flattened.extend(_flatten_export_outputs(outputs[key]))
        return tuple(flattened)
    if hasattr(outputs, "__dataclass_fields__"):
        flattened: list[Tensor] = []
        for field_name in outputs.__dataclass_fields__:
            flattened.extend(_flatten_export_outputs(getattr(outputs, field_name)))
        return tuple(flattened)
    if isinstance(outputs, (list, tuple)):
        flattened = []
        for value in outputs:
            flattened.extend(_flatten_export_outputs(value))
        return tuple(flattened)
    raise TypeError(f"Unsupported ONNX export output type: {type(outputs)!r}")


def _build_dynamic_axes(
    family: ModelFamily,
    input_names: Sequence[str],
    output_names: Sequence[str],
) -> dict[str, dict[int, str]]:
    dynamic_axes: dict[str, dict[int, str]] = {}
    for input_name in input_names:
        axes = {0: "batch_size"}
        if family is ModelFamily.TEXT:
            axes[1] = "sequence_length"
        dynamic_axes[input_name] = axes
    for output_name in output_names:
        dynamic_axes[output_name] = {0: "batch_size"}
    return dynamic_axes


def _resolve_quant_format(raw_format: str) -> QuantFormat:
    if raw_format == "qoperator":
        return QuantFormat.QOperator
    return QuantFormat.QDQ


def _resolve_onnx_quantized_path(
    source_path: Path,
    requested_path: Path | None,
    quantization: QuantizationMode,
) -> Path:
    if requested_path is not None:
        return requested_path
    return Path("artifacts") / f"{source_path.stem}-{quantization.value}-quantized.onnx"


def _preprocess_onnx_for_quantization(
    source_path: Path,
    artifact_path: Path,
) -> Tuple[Path, str]:
    if quant_pre_process is None:
        return source_path, "ONNX pre-processing helper is unavailable; quantization used the original graph."

    preprocessed_path = artifact_path.with_name(f"{artifact_path.stem}-preprocessed.onnx")
    try:
        quant_pre_process(source_path.as_posix(), preprocessed_path.as_posix())
        return preprocessed_path, "ONNX graph pre-processing completed before quantization."
    except Exception:
        if preprocessed_path.exists():
            preprocessed_path.unlink()
        return source_path, "ONNX graph pre-processing failed; quantization used the original graph."


def _build_calibration_tensor(name: str, tensor: Tensor, step: int) -> Tensor:
    lowered_name = name.lower()
    if tensor.dtype in {torch.float16, torch.float32, torch.float64}:
        noise = torch.randn_like(tensor) * 0.05
        return tensor + noise + (step * 0.01)
    if tensor.dtype in {torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8}:
        if "ids" in lowered_name:
            upper_bound = int(max(int(tensor.max().item()) + 1, 2))
            return torch.randint(0, upper_bound, tensor.shape, dtype=tensor.dtype)
        if "mask" in lowered_name:
            return torch.ones(tensor.shape, dtype=tensor.dtype)
        return tensor.clone()
    if tensor.dtype is torch.bool:
        return tensor.clone()
    return tensor.clone()
