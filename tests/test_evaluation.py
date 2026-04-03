from __future__ import annotations

import numpy as np
import torch

from q_lab.evaluation import compute_classification_metrics, extract_logits


def test_extract_logits_prefers_named_logits_like_outputs() -> None:
    output = {
        "last_hidden_state": torch.randn(1, 1, 4),
        "pooler_output": torch.tensor([[0.1, 0.9]]),
    }

    logits = extract_logits(output)

    assert logits is not None
    assert logits.shape == (1, 2)


def test_compute_classification_metrics_reports_accuracy_and_f1() -> None:
    metrics = compute_classification_metrics(
        logits_batches=(
            torch.tensor([[0.9, 0.1], [0.2, 0.8]]),
            torch.tensor([[0.7, 0.3], [0.1, 0.9]]),
        ),
        label_batches=(
            torch.tensor([0, 1]),
            torch.tensor([1, 1]),
        ),
    )

    assert metrics.sample_count == 4
    assert metrics.top1_accuracy_pct == 75.0
    assert metrics.macro_f1_pct is not None and metrics.macro_f1_pct > 70.0


def test_extract_logits_accepts_numpy_outputs() -> None:
    logits = extract_logits([np.array([[0.2, 0.8]], dtype=np.float32)])

    assert logits is not None
    assert logits.shape == (1, 2)
