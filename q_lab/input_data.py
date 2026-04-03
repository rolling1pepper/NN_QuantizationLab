from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from q_lab.types import InputDataset, LoadedModel, ModelFamily


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
InputBatch = Tuple[Tensor, ...]


def load_input_batches(
    path: Path,
    loaded_model: LoadedModel,
) -> Tuple[InputBatch, ...]:
    return load_input_dataset(path=path, loaded_model=loaded_model).batches


def load_input_dataset(
    path: Path,
    loaded_model: LoadedModel,
) -> InputDataset:
    if path.is_dir():
        return _load_image_folder_dataset(path=path, loaded_model=loaded_model)

    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        samples = _extract_json_samples(payload=payload, path=path)
        return _build_dataset_from_records(
            records=samples,
            path=path,
            loaded_model=loaded_model,
            source_format="json",
        )
    if suffix == ".jsonl":
        samples = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return _build_dataset_from_records(
            records=samples,
            path=path,
            loaded_model=loaded_model,
            source_format="jsonl",
        )
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as file_handle:
            reader = csv.DictReader(file_handle)
            records = [dict(row) for row in reader]
        return _build_dataset_from_records(
            records=records,
            path=path,
            loaded_model=loaded_model,
            source_format="csv",
        )

    raise ValueError(
        f"Unsupported dataset path '{path}'. Use a directory image folder or a .json/.jsonl/.csv file."
    )


def _extract_json_samples(payload: Any, path: Path) -> list[Any]:
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


def _build_dataset_from_records(
    records: Sequence[Any],
    path: Path,
    loaded_model: LoadedModel,
    source_format: str,
) -> InputDataset:
    if not records:
        raise ValueError(f"Input dataset '{path}' does not contain any samples.")

    template_inputs = tuple(_ensure_tensor(template) for template in loaded_model.example_inputs)
    batches: list[InputBatch] = []
    label_batches: list[Tensor] = []
    saw_labels = False
    saw_unlabeled = False

    for sample_index, record in enumerate(records):
        batch, labels = _coerce_record(
            record=record,
            input_names=loaded_model.input_names,
            template_inputs=template_inputs,
            path=path,
            sample_index=sample_index,
        )
        batches.append(batch)
        if labels is None:
            saw_unlabeled = True
        else:
            saw_labels = True
            label_batches.append(labels)

    if saw_labels and saw_unlabeled:
        raise ValueError(
            f"Input dataset '{path}' mixes labeled and unlabeled samples. Either label every sample or none."
        )

    return InputDataset(
        batches=tuple(batches),
        label_batches=tuple(label_batches),
        source_format=source_format,
    )


def _coerce_record(
    record: Any,
    input_names: Sequence[str],
    template_inputs: Sequence[Tensor],
    path: Path,
    sample_index: int,
) -> tuple[InputBatch, Tensor | None]:
    label_value = None
    if isinstance(record, Mapping):
        values = []
        label_value = record.get("label")
        for name in input_names:
            if name not in record:
                raise ValueError(
                    f"Input dataset '{path}' sample #{sample_index} is missing key '{name}'."
                )
            values.append(_deserialize_record_value(record[name]))
    elif isinstance(record, list):
        if len(record) != len(input_names):
            raise ValueError(
                f"Input dataset '{path}' sample #{sample_index} must contain exactly "
                f"{len(input_names)} positional values."
            )
        values = list(record)
    else:
        raise ValueError(
            f"Input dataset '{path}' sample #{sample_index} must be a JSON object or JSON array."
        )

    batch = tuple(
        _coerce_tensor_value(
            value=value,
            template=template,
            path=path,
            sample_index=sample_index,
            input_name=name,
        )
        for name, value, template in zip(input_names, values, template_inputs)
    )
    labels = None if label_value is None else _coerce_label_value(
        value=_deserialize_record_value(label_value),
        batch_size=_infer_batch_size(batch),
        path=path,
        sample_index=sample_index,
    )
    return batch, labels


def _deserialize_record_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    if stripped[0] in {"[", "{", '"'}:
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return value
    if stripped.lstrip("-").isdigit():
        try:
            return int(stripped)
        except ValueError:
            return value
    return value


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


def _coerce_label_value(
    value: Any,
    batch_size: int,
    path: Path,
    sample_index: int,
) -> Tensor:
    labels = torch.as_tensor(value, dtype=torch.int64)
    if labels.ndim == 0:
        labels = labels.unsqueeze(0)
    if labels.shape[0] == 1 and batch_size > 1:
        labels = labels.repeat(batch_size)
    if labels.shape[0] != batch_size:
        raise ValueError(
            f"Input dataset '{path}' sample #{sample_index} has {labels.shape[0]} labels for "
            f"batch size {batch_size}."
        )
    return labels


def _load_image_folder_dataset(
    path: Path,
    loaded_model: LoadedModel,
) -> InputDataset:
    if loaded_model.family is not ModelFamily.VISION:
        raise ValueError("Directory datasets are currently supported only for vision models.")
    if len(loaded_model.example_inputs) != 1:
        raise ValueError("Image folder datasets require a single vision input tensor.")

    try:
        from torchvision.io import ImageReadMode, read_image
    except ImportError as error:
        raise ImportError(
            "Image-folder datasets require torchvision. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from error

    template = _ensure_tensor(loaded_model.example_inputs[0])
    channels, height, width = tuple(int(value) for value in template.shape[1:])
    class_directories = sorted(
        directory for directory in path.iterdir() if directory.is_dir()
    )
    if not class_directories:
        raise ValueError(
            f"Image folder dataset '{path}' must contain class-named subdirectories."
        )

    batches: list[InputBatch] = []
    label_batches: list[Tensor] = []
    for label_index, class_directory in enumerate(class_directories):
        image_paths = sorted(
            image_path
            for image_path in class_directory.rglob("*")
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
        )
        for image_path in image_paths:
            image = read_image(image_path.as_posix(), mode=ImageReadMode.RGB).float() / 255.0
            if image.shape[0] != channels:
                if image.shape[0] == 1 and channels == 3:
                    image = image.repeat(3, 1, 1)
                else:
                    image = image[:channels]
            if tuple(image.shape[-2:]) != (height, width):
                image = F.interpolate(
                    image.unsqueeze(0),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            batches.append((image.unsqueeze(0),))
            label_batches.append(torch.tensor([label_index], dtype=torch.int64))

    if not batches:
        raise ValueError(
            f"Image folder dataset '{path}' does not contain any readable image files."
        )

    return InputDataset(
        batches=tuple(batches),
        label_batches=tuple(label_batches),
        source_format="imagefolder",
    )


def _infer_batch_size(batch: Sequence[Tensor]) -> int:
    if not batch:
        return 1
    return int(batch[0].shape[0]) if batch[0].ndim > 0 else 1


def _ensure_tensor(value: Any) -> Tensor:
    if isinstance(value, Tensor):
        return value
    return torch.as_tensor(value)
