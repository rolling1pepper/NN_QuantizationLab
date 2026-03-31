from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Sequence, Tuple

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
from q_lab.types import InputConfig, InvocationMode, LoadedModel, ModelFamily

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


def infer_model_family(model_ref: str, requested_task: str) -> ModelFamily:
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

    supported_names = sorted(
        set(SUPPORTED_TORCHVISION_MODELS) | set(SUPPORTED_TRANSFORMER_MODELS)
    )
    raise ValueError(
        f"Unsupported model reference '{model_ref}'. "
        f"Supported built-ins: {', '.join(supported_names)}."
    )


def build_dummy_inputs(
    config: InputConfig,
    invocation_mode: InvocationMode,
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
        return (image,), ("input",), InvocationMode.POSITIONAL

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
    return (
        (input_ids, attention_mask, token_type_ids),
        ("input_ids", "attention_mask", "token_type_ids"),
        invocation_mode,
    )


def load_model(
    model_ref: str,
    input_config: InputConfig,
    device: str,
    use_pretrained: bool = False,
    onnx_providers: Sequence[str] = ("CPUExecutionProvider",),
    onnx_optimization_level: str = "extended",
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
        )

    lowered_ref = model_ref.lower()
    if lowered_ref in SUPPORTED_TORCHVISION_MODELS:
        return _load_torchvision_model(
            lowered_ref,
            input_config,
            device,
            use_pretrained=use_pretrained,
        )
    if lowered_ref in SUPPORTED_TRANSFORMER_MODELS:
        return _load_transformer_model(
            lowered_ref,
            input_config,
            device,
            use_pretrained=use_pretrained,
        )

    supported_names = sorted(
        set(SUPPORTED_TORCHVISION_MODELS) | set(SUPPORTED_TRANSFORMER_MODELS)
    )
    raise ValueError(
        f"Unsupported model reference '{model_ref}'. "
        f"Expected a TorchScript .pt path, ONNX .onnx path, or one of: {', '.join(supported_names)}."
    )


def _load_torchscript_model(
    model_path: Path,
    input_config: InputConfig,
    device: str,
) -> LoadedModel:
    if model_path.suffix.lower() != ".pt":
        raise ValueError(f"Expected a TorchScript .pt file, received '{model_path.name}'.")
    if not model_path.is_file():
        raise ValueError(f"Model path '{model_path}' is not a file.")

    model = torch.jit.load(model_path.as_posix(), map_location=device)
    model.eval()
    example_inputs, input_names, _ = build_dummy_inputs(
        config=input_config,
        invocation_mode=InvocationMode.POSITIONAL,
    )
    return LoadedModel(
        identifier=model_path.stem,
        display_name=model_path.stem,
        source="torchscript",
        model=model,
        family=input_config.family,
        invocation_mode=InvocationMode.POSITIONAL,
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
) -> LoadedModel:
    display_name, factory = SUPPORTED_TORCHVISION_MODELS[model_name]
    model = factory(use_pretrained).to(device)
    model.eval()
    example_inputs, input_names, invocation_mode = build_dummy_inputs(
        config=input_config,
        invocation_mode=InvocationMode.POSITIONAL,
    )
    return LoadedModel(
        identifier=model_name,
        display_name=display_name,
        source="torchvision",
        model=model,
        family=ModelFamily.VISION,
        invocation_mode=invocation_mode,
        input_names=input_names,
        example_inputs=example_inputs,
        metadata={"weights": "default" if use_pretrained else "random"},
    )


def _load_transformer_model(
    model_name: str,
    input_config: InputConfig,
    device: str,
    use_pretrained: bool,
) -> LoadedModel:
    display_name, checkpoint, model_type = SUPPORTED_TRANSFORMER_MODELS[model_name]
    config = (
        AutoConfig.from_pretrained(checkpoint)
        if use_pretrained
        else AutoConfig.for_model(model_type)
    )
    if use_pretrained:
        model = AutoModel.from_pretrained(checkpoint, config=config).to(device)
    else:
        model = AutoModel.from_config(config).to(device)
    model.eval()
    example_inputs, input_names, invocation_mode = build_dummy_inputs(
        config=input_config,
        invocation_mode=InvocationMode.KEYWORD,
    )
    return LoadedModel(
        identifier=model_name,
        display_name=display_name,
        source="transformers",
        model=model,
        family=ModelFamily.TEXT,
        invocation_mode=invocation_mode,
        input_names=input_names,
        example_inputs=example_inputs,
        metadata={
            "checkpoint": checkpoint,
            "weights": "default" if use_pretrained else "random",
        },
    )
