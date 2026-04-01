from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

import torch
from torch import Tensor

from q_lab.types import LoadedModel


InputBatch = Tuple[Tensor, ...]


def load_input_batches(
    path: Path,
    loaded_model: LoadedModel,
) -> Tuple[InputBatch, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = _extract_samples(payload=payload, path=path)
    if not samples:
        raise ValueError(f"Input dataset '{path}' does not contain any samples.")

    template_inputs = tuple(_ensure_tensor(template) for template in loaded_model.example_inputs)
    return tuple(
        _coerce_sample(
            sample=sample,
            input_names=loaded_model.input_names,
            template_inputs=template_inputs,
            path=path,
            sample_index=index,
        )
        for index, sample in enumerate(samples)
    )


def _extract_samples(payload: Any, path: Path) -> list[Any]:
    if isinstance(payload, dict):
        if "samples" not in payload:
            raise ValueError(
                f"Input dataset '{path}' must be a JSON array or an object with a 'samples' field."
            )
        payload = payload["samples"]

    if not isinstance(payload, list):
        raise ValueError(
            f"Input dataset '{path}' must decode to a JSON array of samples."
        )
    return payload


def _coerce_sample(
    sample: Any,
    input_names: Sequence[str],
    template_inputs: Sequence[Tensor],
    path: Path,
    sample_index: int,
) -> InputBatch:
    if isinstance(sample, Mapping):
        values = []
        for name in input_names:
            if name not in sample:
                raise ValueError(
                    f"Input dataset '{path}' sample #{sample_index} is missing key '{name}'."
                )
            values.append(sample[name])
    elif isinstance(sample, list):
        if len(sample) != len(input_names):
            raise ValueError(
                f"Input dataset '{path}' sample #{sample_index} must contain exactly "
                f"{len(input_names)} positional values."
            )
        values = list(sample)
    else:
        raise ValueError(
            f"Input dataset '{path}' sample #{sample_index} must be a JSON object or JSON array."
        )

    return tuple(
        _coerce_tensor_value(
            value=value,
            template=template,
            path=path,
            sample_index=sample_index,
            input_name=name,
        )
        for name, value, template in zip(input_names, values, template_inputs)
    )


def _coerce_tensor_value(
    value: Any,
    template: Tensor,
    path: Path,
    sample_index: int,
    input_name: str,
) -> Tensor:
    tensor = torch.as_tensor(value, dtype=template.dtype)
    template_shape = tuple(template.shape)
    expected_rank = len(template_shape)

    if expected_rank > 0 and tuple(tensor.shape) == template_shape[1:]:
        tensor = tensor.unsqueeze(0)

    if tensor.ndim != expected_rank:
        raise ValueError(
            f"Input dataset '{path}' sample #{sample_index} input '{input_name}' has rank "
            f"{tensor.ndim}, expected {expected_rank}."
        )

    if expected_rank > 1 and tuple(tensor.shape[1:]) != template_shape[1:]:
        raise ValueError(
            f"Input dataset '{path}' sample #{sample_index} input '{input_name}' has shape "
            f"{tuple(tensor.shape)}, expected batch x {template_shape[1:]}."
        )

    return tensor


def _ensure_tensor(value: Any) -> Tensor:
    if isinstance(value, Tensor):
        return value
    return torch.as_tensor(value)
