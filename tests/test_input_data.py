from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from q_lab.input_data import load_input_batches, load_input_dataset


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


def test_load_input_dataset_keeps_label_batches_for_jsonl(
    text_loaded_model,
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "text_inputs.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "input_ids": [1, 2, 3, 4, 5, 6],
                        "attention_mask": [1, 1, 1, 1, 1, 1],
                        "token_type_ids": [0, 0, 0, 0, 0, 0],
                        "label": 1,
                    }
                ),
                json.dumps(
                    {
                        "input_ids": [2, 3, 4, 5, 6, 7],
                        "attention_mask": [1, 1, 1, 1, 1, 1],
                        "token_type_ids": [0, 0, 0, 0, 0, 0],
                        "label": 0,
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    dataset = load_input_dataset(dataset_path, text_loaded_model)

    assert len(dataset.batches) == 2
    assert len(dataset.label_batches) == 2
    assert dataset.label_batches[0].tolist() == [1]


def test_load_input_dataset_reads_csv_labels(
    text_loaded_model,
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "text_inputs.csv"
    dataset_path.write_text(
        "input_ids,attention_mask,token_type_ids,label\n"
        "\"[1,2,3,4,5,6]\",\"[1,1,1,1,1,1]\",\"[0,0,0,0,0,0]\",1\n",
        encoding="utf-8",
    )

    dataset = load_input_dataset(dataset_path, text_loaded_model)

    assert len(dataset.batches) == 1
    assert dataset.label_batches[0].tolist() == [1]


def test_load_input_dataset_reads_imagefolder_with_labels(
    vision_loaded_model,
    tmp_path: Path,
) -> None:
    from torchvision.utils import save_image

    class_zero_dir = tmp_path / "zero"
    class_one_dir = tmp_path / "one"
    class_zero_dir.mkdir()
    class_one_dir.mkdir()
    save_image(torch.zeros(3, 8, 8), class_zero_dir / "sample0.png")
    save_image(torch.ones(3, 8, 8), class_one_dir / "sample1.png")

    dataset = load_input_dataset(tmp_path, vision_loaded_model)

    assert dataset.source_format == "imagefolder"
    assert len(dataset.batches) == 2
    assert {labels.item() for labels in dataset.label_batches} == {0, 1}
