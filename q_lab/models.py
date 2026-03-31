from __future__ import annotations

import importlib
import importlib.util
import inspect
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

import torch
from torch import nn
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    mobilenet_v3_small,
    resnet18,
    resnet50,
)
from transformers import AutoConfig, AutoModel

from q_lab.onnx_utils import infer_onnx_model_family, load_onnx_model
from q_lab.types import (
    InputConfig,
    InvocationMode,
    LoadedModel,
    ModelFamily,
    ModelFormat,
)

TorchFactory = Callable[[bool], nn.Module]

SUPPORTED_TORCHVISION_MODELS: Dict[str, Tuple[str, TorchFactory]] = {
    "resnet18": (
        "ResNet-18",
        lambda use_pretrained: resnet18(
            weights=ResNet18_Weights.DEFAULT if use_pretrained else None
        ),
    ),
    "resnet50": (
        "ResNet-50",
        lambda use_pretrained: resnet50(
            weights=ResNet50_Weights.DEFAULT if use_pretrained else None
        ),
    ),
    "mobilenet_v3_small": (
        "MobileNetV3 Small",
        lambda use_pretrained: mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.DEFAULT if use_pretrained else None
        ),
    ),
}

SUPPORTED_TRANSFORMER_MODELS: Dict[str, Tuple[str, str, str]] = {
    "bert": ("BERT Base Uncased", "bert-base-uncased", "bert"),
}

TEXT_MODEL_HINTS = ("bert", "distilbert", "roberta", "gpt", "llama", "t5")
VISION_MODEL_HINTS = (
    "vit",
    "swin",
    "deit",
    "beit",
    "convnext",
    "detr",
    "sam",
    "siglip",
    "dinov",
    "mobilevit",
    "resnet",
    "efficientnet",
)
HF_PREFIX = "hf:"
TIMM_PREFIX = "timm:"
PYTHON_PREFIXES = ("python:", "py:")
PYFILE_PREFIX = "pyfile:"


def infer_model_family(
    model_ref: str,
    requested_task: str,
    hf_trust_remote_code: bool = False,
) -> ModelFamily:
    if requested_task != "auto":
        return ModelFamily(requested_task)

    model_path = Path(model_ref)
    if model_path.exists():
        if model_path.suffix.lower() == ".onnx":
            return infer_onnx_model_family(model_path)
        lowered_name = model_path.stem.lower()
        if any(hint in lowered_name for hint in TEXT_MODEL_HINTS):
            return ModelFamily.TEXT
        raise ValueError(
            "Unable to infer task type for the TorchScript model. "
            "Pass --task vision or --task text explicitly."
        )

    lowered_ref = model_ref.lower()
    if lowered_ref in SUPPORTED_TORCHVISION_MODELS:
        return ModelFamily.VISION
    if lowered_ref in SUPPORTED_TRANSFORMER_MODELS:
        return ModelFamily.TEXT
    if lowered_ref.startswith(TIMM_PREFIX):
        return ModelFamily.VISION
    if lowered_ref.startswith(HF_PREFIX):
        return _infer_huggingface_family(
            model_ref=model_ref,
            trust_remote_code=hf_trust_remote_code,
        )
    if lowered_ref.startswith(PYFILE_PREFIX) or lowered_ref.startswith(PYTHON_PREFIXES):
        raise ValueError(
            "Unable to infer task type for Python factory references. "
            "Pass --task vision or --task text explicitly."
        )

    supported_names = sorted(
        set(SUPPORTED_TORCHVISION_MODELS) | set(SUPPORTED_TRANSFORMER_MODELS)
    )
    raise ValueError(
        f"Unsupported model reference '{model_ref}'. "
        "Supported sources: built-in names, TorchScript .pt, ONNX .onnx, "
        f"{TIMM_PREFIX}<model>, {HF_PREFIX}<model>, "
        f"{PYTHON_PREFIXES[0]}<module>:<factory>, {PYFILE_PREFIX}<path>::<factory>. "
        f"Built-ins: {', '.join(supported_names)}."
    )


def build_dummy_inputs(
    config: InputConfig,
    invocation_mode: InvocationMode,
    input_names_override: Sequence[str] | None = None,
) -> Tuple[Tuple[torch.Tensor, ...], Tuple[str, ...], InvocationMode]:
    if config.family is ModelFamily.VISION:
        channels, height, width = config.image_shape
        image = torch.randn(
            config.batch_size,
            channels,
            height,
            width,
            dtype=torch.float32,
        )
        input_names = tuple(input_names_override or ("input",))
        if len(input_names) != 1:
            raise ValueError("Vision models must expose exactly one synthetic input name.")
        return (image,), input_names, invocation_mode

    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(config.batch_size, config.sequence_length),
        dtype=torch.long,
    )
    attention_mask = torch.ones(
        config.batch_size,
        config.sequence_length,
        dtype=torch.long,
    )
    token_type_ids = torch.zeros(
        config.batch_size,
        config.sequence_length,
        dtype=torch.long,
    )
    input_names = tuple(input_names_override or ("input_ids", "attention_mask", "token_type_ids"))
    if not 1 <= len(input_names) <= 3:
        raise ValueError("Text models must expose between one and three synthetic inputs.")
    return (
        (input_ids, attention_mask, token_type_ids)[: len(input_names)],
        input_names,
        invocation_mode,
    )


def load_model(
    model_ref: str,
    input_config: InputConfig,
    device: str,
    use_pretrained: bool = False,
    onnx_providers: Sequence[str] = ("CPUExecutionProvider",),
    onnx_optimization_level: str = "extended",
    model_kwargs: Mapping[str, Any] | None = None,
    invocation_mode_override: InvocationMode | None = None,
    input_names_override: Sequence[str] | None = None,
    hf_trust_remote_code: bool = False,
) -> LoadedModel:
    model_path = Path(model_ref)
    if model_path.exists():
        if model_path.suffix.lower() == ".onnx":
            return load_onnx_model(
                model_path=model_path,
                input_config=input_config,
                providers=onnx_providers,
                optimization_level=onnx_optimization_level,
            )
        return _load_torchscript_model(
            model_path=model_path,
            input_config=input_config,
            device=device,
            invocation_mode_override=invocation_mode_override,
            input_names_override=input_names_override,
        )

    lowered_ref = model_ref.lower()
    if lowered_ref in SUPPORTED_TORCHVISION_MODELS:
        return _invoke_loader(
            _load_torchvision_model,
            model_name=lowered_ref,
            input_config=input_config,
            device=device,
            use_pretrained=use_pretrained,
            invocation_mode_override=invocation_mode_override,
            input_names_override=input_names_override,
        )
    if lowered_ref in SUPPORTED_TRANSFORMER_MODELS:
        return _invoke_loader(
            _load_transformer_model,
            model_name=lowered_ref,
            input_config=input_config,
            device=device,
            use_pretrained=use_pretrained,
            invocation_mode_override=invocation_mode_override,
            input_names_override=input_names_override,
        )
    if lowered_ref.startswith(TIMM_PREFIX):
        return _load_timm_model(
            model_ref=model_ref,
            input_config=input_config,
            device=device,
            use_pretrained=use_pretrained,
            model_kwargs=model_kwargs,
            invocation_mode_override=invocation_mode_override,
            input_names_override=input_names_override,
        )
    if lowered_ref.startswith(HF_PREFIX):
        return _load_huggingface_model(
            model_ref=model_ref,
            input_config=input_config,
            device=device,
            use_pretrained=use_pretrained,
            invocation_mode_override=invocation_mode_override,
            input_names_override=input_names_override,
            hf_trust_remote_code=hf_trust_remote_code,
        )
    if lowered_ref.startswith(PYFILE_PREFIX):
        return _load_python_file_model(
            model_ref=model_ref,
            input_config=input_config,
            device=device,
            use_pretrained=use_pretrained,
            model_kwargs=model_kwargs,
            invocation_mode_override=invocation_mode_override,
            input_names_override=input_names_override,
        )
    if lowered_ref.startswith(PYTHON_PREFIXES):
        return _load_python_module_model(
            model_ref=model_ref,
            input_config=input_config,
            device=device,
            use_pretrained=use_pretrained,
            model_kwargs=model_kwargs,
            invocation_mode_override=invocation_mode_override,
            input_names_override=input_names_override,
        )

    supported_names = sorted(
        set(SUPPORTED_TORCHVISION_MODELS) | set(SUPPORTED_TRANSFORMER_MODELS)
    )
    raise ValueError(
        f"Unsupported model reference '{model_ref}'. "
        "Expected a TorchScript .pt path, ONNX .onnx path, "
        f"{TIMM_PREFIX}<model>, {HF_PREFIX}<model>, "
        f"{PYTHON_PREFIXES[0]}<module>:<factory>, {PYFILE_PREFIX}<path>::<factory>, "
        f"or one of: {', '.join(supported_names)}."
    )


def _load_torchscript_model(
    model_path: Path,
    input_config: InputConfig,
    device: str,
    invocation_mode_override: InvocationMode | None,
    input_names_override: Sequence[str] | None,
) -> LoadedModel:
    if model_path.suffix.lower() != ".pt":
        raise ValueError(f"Expected a TorchScript .pt file, received '{model_path.name}'.")
    if not model_path.is_file():
        raise ValueError(f"Model path '{model_path}' is not a file.")

    invocation_mode = invocation_mode_override or InvocationMode.POSITIONAL
    model = torch.jit.load(model_path.as_posix(), map_location=device)
    model.eval()
    example_inputs, input_names, _ = build_dummy_inputs(
        config=input_config,
        invocation_mode=invocation_mode,
        input_names_override=input_names_override,
    )
    return LoadedModel(
        identifier=model_path.stem,
        display_name=model_path.stem,
        source="torchscript",
        model=model,
        family=input_config.family,
        invocation_mode=invocation_mode,
        input_names=input_names,
        example_inputs=example_inputs,
        original_path=model_path,
        metadata={"task": input_config.family.value},
    )


def _load_torchvision_model(
    model_name: str,
    input_config: InputConfig,
    device: str,
    use_pretrained: bool,
    invocation_mode_override: InvocationMode | None,
    input_names_override: Sequence[str] | None,
) -> LoadedModel:
    display_name, factory = SUPPORTED_TORCHVISION_MODELS[model_name]
    model = factory(use_pretrained)
    return _build_loaded_eager_model(
        identifier=model_name,
        display_name=display_name,
        source="torchvision",
        model=model,
        family=ModelFamily.VISION,
        input_config=input_config,
        device=device,
        default_invocation_mode=InvocationMode.POSITIONAL,
        default_input_names=("input",),
        invocation_mode_override=invocation_mode_override,
        input_names_override=input_names_override,
        metadata={"weights": "default" if use_pretrained else "random"},
    )


def _load_transformer_model(
    model_name: str,
    input_config: InputConfig,
    device: str,
    use_pretrained: bool,
    invocation_mode_override: InvocationMode | None,
    input_names_override: Sequence[str] | None,
) -> LoadedModel:
    display_name, checkpoint, model_type = SUPPORTED_TRANSFORMER_MODELS[model_name]
    config = (
        AutoConfig.from_pretrained(checkpoint)
        if use_pretrained
        else AutoConfig.for_model(model_type)
    )
    if use_pretrained:
        model = AutoModel.from_pretrained(checkpoint, config=config)
    else:
        model = AutoModel.from_config(config)
    return _build_loaded_eager_model(
        identifier=model_name,
        display_name=display_name,
        source="transformers",
        model=model,
        family=ModelFamily.TEXT,
        input_config=input_config,
        device=device,
        default_invocation_mode=InvocationMode.KEYWORD,
        default_input_names=("input_ids", "attention_mask", "token_type_ids"),
        invocation_mode_override=invocation_mode_override,
        input_names_override=input_names_override,
        metadata={
            "checkpoint": checkpoint,
            "weights": "default" if use_pretrained else "random",
        },
    )


def _load_huggingface_model(
    model_ref: str,
    input_config: InputConfig,
    device: str,
    use_pretrained: bool,
    invocation_mode_override: InvocationMode | None,
    input_names_override: Sequence[str] | None,
    hf_trust_remote_code: bool,
) -> LoadedModel:
    model_id = _strip_prefix(model_ref, HF_PREFIX)
    config = AutoConfig.from_pretrained(
        model_id,
        trust_remote_code=hf_trust_remote_code,
    )
    if use_pretrained:
        model = AutoModel.from_pretrained(
            model_id,
            config=config,
            trust_remote_code=hf_trust_remote_code,
        )
    else:
        model = AutoModel.from_config(config)

    default_input_names = (
        ("pixel_values",)
        if input_config.family is ModelFamily.VISION
        else ("input_ids", "attention_mask", "token_type_ids")
    )
    default_invocation_mode = InvocationMode.KEYWORD
    return _build_loaded_eager_model(
        identifier=_sanitize_identifier(f"hf-{model_id}"),
        display_name=model_id,
        source="huggingface",
        model=model,
        family=input_config.family,
        input_config=input_config,
        device=device,
        default_invocation_mode=default_invocation_mode,
        default_input_names=default_input_names,
        invocation_mode_override=invocation_mode_override,
        input_names_override=input_names_override,
        metadata={
            "checkpoint": model_id,
            "weights": "default" if use_pretrained else "random-config",
            "trust_remote_code": hf_trust_remote_code,
        },
    )


def _load_timm_model(
    model_ref: str,
    input_config: InputConfig,
    device: str,
    use_pretrained: bool,
    model_kwargs: Mapping[str, Any] | None,
    invocation_mode_override: InvocationMode | None,
    input_names_override: Sequence[str] | None,
) -> LoadedModel:
    try:
        timm = importlib.import_module("timm")
    except ImportError as error:
        raise ImportError(
            "Loading timm models requires the 'timm' package. "
            "Install dependencies with `python -m pip install -r requirements.txt`."
        ) from error

    model_name = _strip_prefix(model_ref, TIMM_PREFIX)
    model = timm.create_model(
        model_name,
        pretrained=use_pretrained,
        **dict(model_kwargs or {}),
    )
    return _build_loaded_eager_model(
        identifier=_sanitize_identifier(f"timm-{model_name}"),
        display_name=model_name,
        source="timm",
        model=model,
        family=ModelFamily.VISION,
        input_config=input_config,
        device=device,
        default_invocation_mode=InvocationMode.POSITIONAL,
        default_input_names=("input",),
        invocation_mode_override=invocation_mode_override,
        input_names_override=input_names_override,
        metadata={
            "model_name": model_name,
            "weights": "default" if use_pretrained else "random",
            "model_kwargs": dict(model_kwargs or {}),
        },
    )


def _load_python_module_model(
    model_ref: str,
    input_config: InputConfig,
    device: str,
    use_pretrained: bool,
    model_kwargs: Mapping[str, Any] | None,
    invocation_mode_override: InvocationMode | None,
    input_names_override: Sequence[str] | None,
) -> LoadedModel:
    module_name, factory_name = _parse_python_module_reference(model_ref)
    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name, None)
    if factory is None:
        raise AttributeError(
            f"Python factory '{factory_name}' was not found in module '{module_name}'."
        )
    model = _invoke_python_factory(
        factory=factory,
        use_pretrained=use_pretrained,
        model_kwargs=model_kwargs,
    )
    return _build_loaded_eager_model(
        identifier=_sanitize_identifier(f"{module_name}-{factory_name}"),
        display_name=f"{module_name}:{factory_name}",
        source="python_module",
        model=model,
        family=input_config.family,
        input_config=input_config,
        device=device,
        default_invocation_mode=_default_invocation_mode_for_family(input_config.family),
        default_input_names=_default_input_names_for_family(input_config.family),
        invocation_mode_override=invocation_mode_override,
        input_names_override=input_names_override,
        metadata={
            "module": module_name,
            "factory": factory_name,
            "weights": "default" if use_pretrained else "factory-default",
            "model_kwargs": dict(model_kwargs or {}),
        },
    )


def _load_python_file_model(
    model_ref: str,
    input_config: InputConfig,
    device: str,
    use_pretrained: bool,
    model_kwargs: Mapping[str, Any] | None,
    invocation_mode_override: InvocationMode | None,
    input_names_override: Sequence[str] | None,
) -> LoadedModel:
    file_path, factory_name = _parse_python_file_reference(model_ref)
    if not file_path.is_file():
        raise ValueError(f"Python model factory file '{file_path}' does not exist.")

    module_name = f"q_lab_user_factory_{_sanitize_identifier(file_path.stem)}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import Python factory from '{file_path}'.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    factory = getattr(module, factory_name, None)
    if factory is None:
        raise AttributeError(
            f"Python factory '{factory_name}' was not found in file '{file_path}'."
        )
    model = _invoke_python_factory(
        factory=factory,
        use_pretrained=use_pretrained,
        model_kwargs=model_kwargs,
    )
    return _build_loaded_eager_model(
        identifier=_sanitize_identifier(f"{file_path.stem}-{factory_name}"),
        display_name=f"{file_path.name}:{factory_name}",
        source="python_file",
        model=model,
        family=input_config.family,
        input_config=input_config,
        device=device,
        default_invocation_mode=_default_invocation_mode_for_family(input_config.family),
        default_input_names=_default_input_names_for_family(input_config.family),
        invocation_mode_override=invocation_mode_override,
        input_names_override=input_names_override,
        metadata={
            "file": file_path.as_posix(),
            "factory": factory_name,
            "weights": "default" if use_pretrained else "factory-default",
            "model_kwargs": dict(model_kwargs or {}),
        },
    )


def _build_loaded_eager_model(
    identifier: str,
    display_name: str,
    source: str,
    model: nn.Module,
    family: ModelFamily,
    input_config: InputConfig,
    device: str,
    default_invocation_mode: InvocationMode,
    default_input_names: Sequence[str],
    invocation_mode_override: InvocationMode | None,
    input_names_override: Sequence[str] | None,
    metadata: Mapping[str, Any] | None = None,
) -> LoadedModel:
    if not isinstance(model, nn.Module):
        raise TypeError(
            f"Expected an eager torch.nn.Module instance from source '{source}', "
            f"received '{type(model).__name__}'."
        )
    resolved_invocation_mode = invocation_mode_override or default_invocation_mode
    if input_names_override is not None:
        requested_input_names = tuple(input_names_override)
    elif resolved_invocation_mode is InvocationMode.KEYWORD:
        requested_input_names = _infer_input_names_from_model(
            model=model,
            family=family,
            default_input_names=default_input_names,
        )
    else:
        requested_input_names = tuple(default_input_names)
    example_inputs, input_names, invocation_mode = build_dummy_inputs(
        config=input_config,
        invocation_mode=resolved_invocation_mode,
        input_names_override=requested_input_names,
    )
    model = model.to(device)
    model.eval()
    return LoadedModel(
        identifier=identifier,
        display_name=display_name,
        source=source,
        model=model,
        family=family,
        invocation_mode=invocation_mode,
        input_names=input_names,
        example_inputs=example_inputs,
        format=ModelFormat.PYTORCH,
        metadata=dict(metadata or {}),
    )


def _infer_huggingface_family(
    model_ref: str,
    trust_remote_code: bool,
) -> ModelFamily:
    model_id = _strip_prefix(model_ref, HF_PREFIX)
    lowered_ref = model_id.lower()
    if any(hint in lowered_ref for hint in TEXT_MODEL_HINTS):
        return ModelFamily.TEXT
    if any(hint in lowered_ref for hint in VISION_MODEL_HINTS):
        return ModelFamily.VISION

    try:
        config = AutoConfig.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )
    except Exception as error:
        raise ValueError(
            "Unable to infer task type for the Hugging Face model. "
            "Pass --task vision or --task text explicitly."
        ) from error

    config_dict = config.to_dict()
    model_type = getattr(config, "model_type", "").lower()
    if any(hint in model_type for hint in TEXT_MODEL_HINTS):
        return ModelFamily.TEXT
    if any(hint in model_type for hint in VISION_MODEL_HINTS):
        return ModelFamily.VISION
    if "vision_config" in config_dict or "num_channels" in config_dict or "image_size" in config_dict:
        return ModelFamily.VISION
    if "vocab_size" in config_dict or "max_position_embeddings" in config_dict:
        return ModelFamily.TEXT

    raise ValueError(
        "Unable to infer task type for the Hugging Face model. "
        "Pass --task vision or --task text explicitly."
    )


def _invoke_python_factory(
    factory: Callable[..., Any],
    use_pretrained: bool,
    model_kwargs: Mapping[str, Any] | None,
) -> Any:
    signature = inspect.signature(factory)
    call_kwargs = dict(model_kwargs or {})
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )

    if "use_pretrained" in signature.parameters or accepts_kwargs:
        call_kwargs.setdefault("use_pretrained", use_pretrained)
    elif "pretrained" in signature.parameters:
        call_kwargs.setdefault("pretrained", use_pretrained)

    return factory(**call_kwargs)


def _parse_python_module_reference(model_ref: str) -> Tuple[str, str]:
    prefix = next(prefix for prefix in PYTHON_PREFIXES if model_ref.lower().startswith(prefix))
    payload = _strip_prefix(model_ref, prefix)
    if ":" in payload:
        module_name, factory_name = payload.rsplit(":", 1)
    else:
        module_name, factory_name = payload, "create_model"
    if not module_name or not factory_name:
        raise ValueError(
            "Python module references must use 'python:package.module:factory' "
            "or 'py:package.module:factory'."
        )
    return module_name, factory_name


def _parse_python_file_reference(model_ref: str) -> Tuple[Path, str]:
    payload = _strip_prefix(model_ref, PYFILE_PREFIX)
    if "::" in payload:
        path_str, factory_name = payload.rsplit("::", 1)
    else:
        path_str, factory_name = payload, "create_model"
    file_path = Path(path_str).expanduser()
    if not path_str or not factory_name:
        raise ValueError(
            "Python file references must use 'pyfile:path/to/file.py::factory'."
        )
    return file_path, factory_name


def _default_invocation_mode_for_family(family: ModelFamily) -> InvocationMode:
    if family is ModelFamily.TEXT:
        return InvocationMode.KEYWORD
    return InvocationMode.POSITIONAL


def _default_input_names_for_family(family: ModelFamily) -> Tuple[str, ...]:
    if family is ModelFamily.TEXT:
        return ("input_ids", "attention_mask", "token_type_ids")
    return ("input",)


def _infer_input_names_from_model(
    model: nn.Module,
    family: ModelFamily,
    default_input_names: Sequence[str],
) -> Tuple[str, ...]:
    try:
        signature = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return tuple(default_input_names)

    parameter_names = [
        name
        for name, parameter in signature.parameters.items()
        if name != "self"
        and parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    ]
    if not parameter_names:
        return tuple(default_input_names)

    if family is ModelFamily.VISION:
        for candidate in default_input_names:
            if candidate in parameter_names:
                return (candidate,)
        return (parameter_names[0],)

    selected = tuple(
        name
        for name in ("input_ids", "attention_mask", "token_type_ids")
        if name in parameter_names
    )
    if selected:
        return selected
    return tuple(parameter_names[:3])


def _sanitize_identifier(value: str) -> str:
    sanitized = value.replace("\\", "_").replace("/", "_").replace(":", "_")
    return sanitized.replace(".", "_")


def _strip_prefix(value: str, prefix: str) -> str:
    if value.lower().startswith(prefix):
        return value[len(prefix) :]
    return value


def _invoke_loader(loader: Callable[..., LoadedModel], **kwargs: Any) -> LoadedModel:
    signature = inspect.signature(loader)
    filtered_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }
    return loader(**filtered_kwargs)
