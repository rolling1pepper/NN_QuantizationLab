from __future__ import annotations

from typing import Any, Iterable, Sequence

import torch
from torch import Tensor

from q_lab.types import EvaluationMetrics


def extract_logits(output: Any) -> Tensor | None:
    direct_logits = _try_extract_named_logits(output)
    if direct_logits is not None:
        return direct_logits.detach().cpu()

    candidates = list(_collect_tensor_candidates(output))
    ranked = [
        candidate.detach().cpu()
        for candidate in candidates
        if candidate.ndim >= 2 and candidate.shape[-1] > 1
    ]
    if not ranked:
        return None

    return sorted(
        ranked,
        key=lambda tensor: (-int(tensor.shape[-1]), tensor.ndim),
    )[0]


def compute_classification_metrics(
    logits_batches: Sequence[Tensor],
    label_batches: Sequence[Tensor],
) -> EvaluationMetrics:
    flattened_predictions: list[Tensor] = []
    flattened_labels: list[Tensor] = []

    for logits, labels in zip(logits_batches, label_batches):
        predictions = logits.argmax(dim=-1).reshape(-1).to(torch.int64)
        labels_tensor = labels.reshape(-1).to(torch.int64)
        valid_mask = labels_tensor.ge(0)
        if valid_mask.sum().item() == 0:
            continue

        flattened_predictions.append(predictions[valid_mask])
        flattened_labels.append(labels_tensor[valid_mask])

    if not flattened_predictions or not flattened_labels:
        return EvaluationMetrics()

    predictions_tensor = torch.cat(flattened_predictions)
    labels_tensor = torch.cat(flattened_labels)
    if predictions_tensor.numel() == 0 or labels_tensor.numel() == 0:
        return EvaluationMetrics()

    accuracy_pct = predictions_tensor.eq(labels_tensor).float().mean().item() * 100.0
    macro_f1_pct = _compute_macro_f1_pct(
        predictions=predictions_tensor,
        labels=labels_tensor,
    )
    return EvaluationMetrics(
        sample_count=int(labels_tensor.numel()),
        top1_accuracy_pct=accuracy_pct,
        macro_f1_pct=macro_f1_pct,
    )


def _try_extract_named_logits(output: Any) -> Tensor | None:
    if output is None:
        return None
    if isinstance(output, Tensor):
        return output
    if isinstance(output, (list, tuple, dict)):
        logits = getattr(output, "logits", None)
        if isinstance(logits, Tensor):
            return logits
        if isinstance(output, dict):
            for preferred_key in ("logits", "pooler_output"):
                tensor = output.get(preferred_key)
                if isinstance(tensor, Tensor):
                    return tensor
        return None
    try:
        tensor = torch.as_tensor(output)
    except (RuntimeError, TypeError, ValueError):
        tensor = None
    if tensor is not None and tensor.ndim >= 2:
        return tensor
    logits = getattr(output, "logits", None)
    if isinstance(logits, Tensor):
        return logits
    if isinstance(output, dict):
        for preferred_key in ("logits", "pooler_output"):
            tensor = output.get(preferred_key)
            if isinstance(tensor, Tensor):
                return tensor
    return None


def _collect_tensor_candidates(output: Any) -> Iterable[Tensor]:
    if output is None:
        return
    if isinstance(output, Tensor):
        yield output
        return
    if hasattr(output, "logits") and isinstance(output.logits, Tensor):
        yield output.logits
    if hasattr(output, "to_tuple"):
        yield from _collect_tensor_candidates(output.to_tuple())
        return
    if isinstance(output, dict):
        for key in sorted(output):
            yield from _collect_tensor_candidates(output[key])
        return
    if hasattr(output, "__dataclass_fields__"):
        for field_name in output.__dataclass_fields__:
            yield from _collect_tensor_candidates(getattr(output, field_name))
        return
    if isinstance(output, (list, tuple)):
        for item in output:
            yield from _collect_tensor_candidates(item)
        return
    try:
        tensor = torch.as_tensor(output)
    except (RuntimeError, TypeError, ValueError):
        tensor = None
    if tensor is not None:
        yield tensor


def _compute_macro_f1_pct(predictions: Tensor, labels: Tensor) -> float:
    class_ids = torch.unique(torch.cat((predictions, labels))).tolist()
    f1_scores: list[float] = []
    for class_id in class_ids:
        predicted_mask = predictions.eq(class_id)
        label_mask = labels.eq(class_id)
        true_positives = predicted_mask.logical_and(label_mask).sum().item()
        false_positives = predicted_mask.logical_and(label_mask.logical_not()).sum().item()
        false_negatives = predicted_mask.logical_not().logical_and(label_mask).sum().item()

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0.0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0.0
        )
        if precision == 0.0 and recall == 0.0:
            f1_scores.append(0.0)
            continue
        f1_scores.append((2 * precision * recall) / (precision + recall))

    if not f1_scores:
        return 0.0
    return (sum(f1_scores) / len(f1_scores)) * 100.0
