from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from q_lab import cli, models
from q_lab.experiments import build_run_matrix, collect_explicit_overrides
from q_lab.types import (
    CompressionConfig,
    InvocationMode,
    LoadedModel,
    ModelFamily,
    ModelFormat,
    PruningMode,
    QuantizationMode,
)
from tests.helpers import RuleBasedVisionClassifier, TinyTextNet, TinyVisionNet


def test_parse_image_shape_accepts_valid_input() -> None:
    assert cli.parse_image_shape("3,224,224") == (3, 224, 224)


def test_parse_image_shape_rejects_invalid_input() -> None:
    with pytest.raises(ValueError, match="Image shape must be provided as C,H,W"):
        cli.parse_image_shape("3,224")


def test_parse_onnx_providers_accepts_valid_input() -> None:
    assert cli.parse_onnx_providers("CUDAExecutionProvider,CPUExecutionProvider") == (
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    )


def test_parse_input_names_accepts_valid_input() -> None:
    assert cli.parse_input_names("pixel_values") == ("pixel_values",)


def test_parse_model_kwargs_accepts_inline_json() -> None:
    assert cli.parse_model_kwargs('{"drop_path_rate": 0.1}') == {"drop_path_rate": 0.1}


def test_parse_batch_sizes_accepts_valid_input() -> None:
    assert cli.parse_batch_sizes("1,4,8", 1) == (1, 4, 8)


def test_expand_config_argv_applies_json_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "run.json"
    config_path.write_text(
        json.dumps(
            {
                "model": "resnet18",
                "warmup_iterations": 2,
                "benchmark_iterations": 5,
                "batch_sizes": [1, 4],
                "pretrained": True,
            }
        ),
        encoding="utf-8",
    )

    parser = cli.build_parser()
    args = parser.parse_args(
        cli.expand_config_argv(
            [
                "--config",
                str(config_path),
                "--benchmark-iterations",
                "7",
            ]
        )
    )

    assert args.model == "resnet18"
    assert args.warmup_iterations == 2
    assert args.benchmark_iterations == 7
    assert args.batch_sizes == "1,4"
    assert args.pretrained is True


def test_build_run_matrix_expands_matrix_axes_from_config(tmp_path: Path) -> None:
    config_path = tmp_path / "matrix.json"
    config_path.write_text(
        json.dumps(
            {
                "model": "resnet18",
                "matrix": {
                    "quantization": ["none", "dynamic"],
                    "providers": ["CPUExecutionProvider"],
                },
            }
        ),
        encoding="utf-8",
    )

    parser = cli.build_parser()
    argv = ["--config", str(config_path)]
    args = parser.parse_args(cli.expand_config_argv(argv))
    explicit_overrides, _ = collect_explicit_overrides(parser, argv)
    matrix_runs, varying_keys = build_run_matrix(
        args=args,
        raw_config=cli.load_json_config(config_path),
        explicit_overrides=explicit_overrides,
        batch_size_values=cli.parse_batch_sizes(args.batch_sizes, args.batch_size),
    )

    assert varying_keys == ("providers", "quantization")
    assert len(matrix_runs) == 2


def test_validate_arguments_rejects_invalid_pruning_amount() -> None:
    parser = cli.build_parser()

    with pytest.raises(SystemExit):
        cli._validate_arguments(
            parser=parser,
            compression_config=CompressionConfig(
                quantization=QuantizationMode.NONE,
                pruning=PruningMode.NONE,
                pruning_amount=0.2,
                export_onnx=False,
                onnx_path=None,
                onnx_opset=17,
            ),
            family=ModelFamily.VISION,
            device="cpu",
            model_format=ModelFormat.PYTORCH,
        )


def test_validate_arguments_rejects_onnx_pruning() -> None:
    parser = cli.build_parser()

    with pytest.raises(SystemExit):
        cli._validate_arguments(
            parser=parser,
            compression_config=CompressionConfig(
                quantization=QuantizationMode.NONE,
                pruning=PruningMode.UNSTRUCTURED,
                pruning_amount=0.2,
                export_onnx=False,
                onnx_path=None,
                onnx_opset=17,
            ),
            family=ModelFamily.VISION,
            device="cpu",
            model_format=ModelFormat.ONNX,
        )


@pytest.mark.integration
def test_cli_runs_torchscript_baseline_and_onnx(
    scripted_vision_model_path: Path,
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "torchscript_report.csv"
    onnx_path = tmp_path / "torchscript_model.onnx"

    exit_code = cli.main(
        [
            str(scripted_vision_model_path),
            "--task",
            "vision",
            "--image-shape",
            "3,8,8",
            "--warmup-iterations",
            "1",
            "--benchmark-iterations",
            "2",
            "--export-onnx",
            "--onnx-path",
            str(onnx_path),
            "--report-path",
            str(report_path),
        ]
    )

    assert exit_code == 0
    assert report_path.exists()
    assert onnx_path.exists()

    dataframe = pd.read_csv(report_path)

    assert set(dataframe["label"]) == {"baseline", "baseline-onnx"}
    assert set(dataframe["backend"]) == {"pytorch", "onnxruntime"}


@pytest.mark.integration
def test_cli_runs_static_quantization_for_patched_builtin_vision_model(
    monkeypatch,
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "static_report.csv"
    monkeypatch.setitem(
        models.SUPPORTED_TORCHVISION_MODELS,
        "toyvision",
        ("Toy Vision", lambda use_pretrained: TinyVisionNet()),
    )

    exit_code = cli.main(
        [
            "toyvision",
            "--quantization",
            "static",
            "--warmup-iterations",
            "1",
            "--benchmark-iterations",
            "2",
            "--calibration-iterations",
            "1",
            "--report-path",
            str(report_path),
        ]
    )

    assert exit_code == 0
    assert report_path.exists()

    dataframe = pd.read_csv(report_path)

    assert set(dataframe["label"]) == {"baseline", "optimized"}
    assert set(dataframe["quantization"]) == {"none", "static"}
    assert set(dataframe["backend"]) == {"pytorch"}


@pytest.mark.integration
def test_cli_runs_dynamic_quantization_for_patched_builtin_text_model(
    monkeypatch,
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "dynamic_report.csv"

    def fake_load_transformer_model(
        model_name: str,
        input_config,
        device: str,
        use_pretrained: bool,
    ) -> LoadedModel:
        del model_name, device, use_pretrained
        input_ids = models.torch.randint(
            low=0,
            high=128,
            size=(input_config.batch_size, input_config.sequence_length),
            dtype=models.torch.long,
        )
        attention_mask = models.torch.ones(
            (input_config.batch_size, input_config.sequence_length),
            dtype=models.torch.long,
        )
        token_type_ids = models.torch.zeros(
            (input_config.batch_size, input_config.sequence_length),
            dtype=models.torch.long,
        )
        return LoadedModel(
            identifier="toytext",
            display_name="Toy Text",
            source="transformers",
            model=TinyTextNet().eval(),
            family=ModelFamily.TEXT,
            invocation_mode=InvocationMode.KEYWORD,
            input_names=("input_ids", "attention_mask", "token_type_ids"),
            example_inputs=(input_ids, attention_mask, token_type_ids),
        )

    monkeypatch.setitem(
        models.SUPPORTED_TRANSFORMER_MODELS,
        "toytext",
        ("Toy Text", "toy-checkpoint", "bert"),
    )
    monkeypatch.setattr(models, "_load_transformer_model", fake_load_transformer_model)

    exit_code = cli.main(
        [
            "toytext",
            "--task",
            "text",
            "--quantization",
            "dynamic",
            "--warmup-iterations",
            "1",
            "--benchmark-iterations",
            "2",
            "--report-path",
            str(report_path),
        ]
    )

    assert exit_code == 0
    assert report_path.exists()

    dataframe = pd.read_csv(report_path)

    assert set(dataframe["label"]) == {"baseline", "optimized"}
    assert set(dataframe["quantization"]) == {"none", "dynamic"}


@pytest.mark.integration
def test_cli_runs_dynamic_quantization_for_onnx_text_input(
    onnx_text_model_path: Path,
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "onnx_dynamic_report.csv"
    quantized_path = tmp_path / "onnx_dynamic_quantized.onnx"

    exit_code = cli.main(
        [
            str(onnx_text_model_path),
            "--quantization",
            "dynamic",
            "--sequence-length",
            "6",
            "--vocab-size",
            "256",
            "--warmup-iterations",
            "1",
            "--benchmark-iterations",
            "2",
            "--onnx-quantized-path",
            str(quantized_path),
            "--report-path",
            str(report_path),
        ]
    )

    assert exit_code == 0
    assert report_path.exists()
    assert quantized_path.exists()

    dataframe = pd.read_csv(report_path)

    assert set(dataframe["label"]) == {"baseline", "optimized"}
    assert set(dataframe["backend"]) == {"onnxruntime"}
    assert set(dataframe["quantization"]) == {"none", "dynamic"}
    assert set(dataframe["source"]) == {"onnx"}


@pytest.mark.integration
def test_cli_runs_batch_size_sweep_from_config_for_patched_builtin_vision_model(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "run_config.json"
    report_path = tmp_path / "batch_sweep.csv"
    config_path.write_text(
        json.dumps(
            {
                "model": "toyvision",
                "quantization": "static",
                "image_shape": "3,8,8",
                "warmup_iterations": 1,
                "benchmark_iterations": 2,
                "calibration_iterations": 1,
                "batch_sizes": [1, 2],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setitem(
        models.SUPPORTED_TORCHVISION_MODELS,
        "toyvision",
        ("Toy Vision", lambda use_pretrained: TinyVisionNet()),
    )

    exit_code = cli.main(
        [
            "--config",
            str(config_path),
            "--report-path",
            str(report_path),
        ]
    )

    assert exit_code == 0
    dataframe = pd.read_csv(report_path)

    assert set(dataframe["batch_size"]) == {1, 2}
    assert set(dataframe["label"]) == {"baseline", "optimized"}
    assert "throughput_items_per_sec" in dataframe.columns


@pytest.mark.integration
def test_cli_writes_manifest_html_and_eval_metrics(
    monkeypatch,
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "eval_report.csv"
    html_path = tmp_path / "eval_report.html"
    manifest_path = tmp_path / "eval_manifest.json"
    eval_dataset_path = tmp_path / "eval_samples.json"
    eval_dataset_path.write_text(
        json.dumps(
            {
                "samples": [
                    {"input": [[[0.0] * 8 for _ in range(8)] for _ in range(3)], "label": 0},
                    {"input": [[[1.0] * 8 for _ in range(8)] for _ in range(3)], "label": 1},
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setitem(
        models.SUPPORTED_TORCHVISION_MODELS,
        "rulevision",
        ("Rule Vision", lambda use_pretrained: RuleBasedVisionClassifier()),
    )

    exit_code = cli.main(
        [
            "rulevision",
            "--eval-data-path",
            str(eval_dataset_path),
            "--html-report-path",
            str(html_path),
            "--manifest-path",
            str(manifest_path),
            "--report-path",
            str(report_path),
        ]
    )

    assert exit_code == 0
    assert html_path.exists()
    assert manifest_path.exists()

    dataframe = pd.read_csv(report_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert dataframe.loc[0, "eval_top1_accuracy_pct"] == 100.0
    assert dataframe.loc[0, "eval_macro_f1_pct"] == 100.0
    assert manifest["artifacts"]["report_csv"] == str(report_path)


@pytest.mark.integration
def test_cli_uses_json_input_datasets_for_benchmark_calibration_and_eval(
    monkeypatch,
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "dataset_driven.csv"
    dataset_path = tmp_path / "vision_samples.json"
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
    monkeypatch.setitem(
        models.SUPPORTED_TORCHVISION_MODELS,
        "toyvision",
        ("Toy Vision", lambda use_pretrained: TinyVisionNet()),
    )

    exit_code = cli.main(
        [
            "toyvision",
            "--quantization",
            "static",
            "--image-shape",
            "3,8,8",
            "--warmup-iterations",
            "1",
            "--benchmark-iterations",
            "2",
            "--calibration-iterations",
            "2",
            "--benchmark-inputs-json",
            str(dataset_path),
            "--calibration-inputs-json",
            str(dataset_path),
            "--eval-inputs-json",
            str(dataset_path),
            "--report-path",
            str(report_path),
        ]
    )

    assert exit_code == 0
    dataframe = pd.read_csv(report_path)

    assert set(dataframe["label"]) == {"baseline", "optimized"}
    assert dataframe["throughput_items_per_sec"].notna().all()


@pytest.mark.integration
def test_cli_runs_static_quantization_for_onnx_vision_input(
    onnx_vision_model_path: Path,
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "onnx_static_report.csv"
    quantized_path = tmp_path / "onnx_static_quantized.onnx"

    exit_code = cli.main(
        [
            str(onnx_vision_model_path),
            "--quantization",
            "static",
            "--image-shape",
            "3,8,8",
            "--warmup-iterations",
            "1",
            "--benchmark-iterations",
            "2",
            "--calibration-iterations",
            "1",
            "--onnx-quantized-path",
            str(quantized_path),
            "--report-path",
            str(report_path),
        ]
    )

    assert exit_code == 0
    assert report_path.exists()
    assert quantized_path.exists()

    dataframe = pd.read_csv(report_path)

    assert set(dataframe["label"]) == {"baseline", "optimized"}
    assert set(dataframe["backend"]) == {"onnxruntime"}
    assert set(dataframe["quantization"]) == {"none", "static"}
