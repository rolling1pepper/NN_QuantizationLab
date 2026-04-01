from __future__ import annotations

import json
from pathlib import Path

import pytest

from q_lab.input_data import load_input_batches


def test_load_input_batches_for_text_promotes_single_sample_shape(
    text_loaded_model,
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "text_inputs.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "input_ids": [1, 2, 3, 4, 5, 6],
                    "attention_mask": [1, 1, 1, 1, 1, 1],
                    "token_type_ids": [0, 0, 0, 0, 0, 0],
                }
            ]
        ),
        encoding="utf-8",
    )

    batches = load_input_batches(dataset_path, text_loaded_model)

    assert len(batches) == 1
    assert batches[0][0].shape == (1, 6)
    assert batches[0][1].shape == (1, 6)


def test_load_input_batches_for_vision_accepts_mapping_samples(
    vision_loaded_model,
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "vision_inputs.json"
    dataset_path.write_text(
        json.dumps(
            {
                "samples": [
                    {"input": [[[0.0] * 8 for _ in range(8)] for _ in range(3)]},
                    {"input": [[[1.0] * 8 for _ in range(8)] for _ in range(3)]},
                ]
            }
        ),
        encoding="utf-8",
    )

    batches = load_input_batches(dataset_path, vision_loaded_model)

    assert len(batches) == 2
    assert batches[0][0].shape == (1, 3, 8, 8)


def test_load_input_batches_rejects_missing_required_keys(
    text_loaded_model,
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "invalid_text_inputs.json"
    dataset_path.write_text(
        json.dumps([{"input_ids": [1, 2, 3, 4, 5, 6]}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing key 'attention_mask'"):
        load_input_batches(dataset_path, text_loaded_model)
