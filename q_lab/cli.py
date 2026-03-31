from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence, Tuple

from q_lab import __version__
from q_lab.types import (
    BenchmarkConfig,
    CompressionConfig,
    InputConfig,
    InvocationMode,
    ModelFamily,
    ModelFormat,
    PruningMode,
    QuantizationMode,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="q_lab",
        description=(
            "Q-Lab benchmarks, prunes, quantizes and exports PyTorch, TorchScript "
            "and ONNX models with Rich console and CSV reporting. "
            "Supported eager sources include built-ins, timm:<model>, hf:<model>, "
            "python:<module>:<factory>, and pyfile:<path>::<factory>."
        ),
    )
    parser.add_argument(
        "model",
        help=(
            "Model reference: TorchScript .pt path, ONNX .onnx path, built-in name "
            "(resnet18, resnet50, mobilenet_v3_small, bert), timm:<name>, hf:<model>, "
            "python:<module>:<factory>, or pyfile:<path>::<factory>."
        ),
    )
    parser.add_argument(
        "--task",
        choices=("auto", "vision", "text"),
        default="auto",
        help="Input family. Use 'auto' whenever Q-Lab can infer it from the model or artifact.",
    )
    parser.add_argument(
        "--quantization",
        choices=[mode.value for mode in QuantizationMode],
        default=QuantizationMode.NONE.value,
        help="Quantization strategy. ONNX inputs use ONNX Runtime quantization.",
    )
    parser.add_argument(
        "--pruning",
        choices=[mode.value for mode in PruningMode],
        default=PruningMode.NONE.value,
        help="Pruning strategy. Supported for eager PyTorch models only.",
    )
    parser.add_argument(
        "--pruning-amount",
        type=float,
        default=0.0,
        help="Pruning ratio in [0.0, 1.0). Ignored when pruning is disabled.",
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=10,
        help="Number of warmup iterations before timing.",
    )
    parser.add_argument(
        "--benchmark-iterations",
        type=int,
        default=50,
        help="Number of timed iterations.",
    )
    parser.add_argument(
        "--calibration-iterations",
        type=int,
        default=10,
        help="Number of synthetic calibration iterations for static quantization flows.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Synthetic batch size for inference and ONNX calibration.",
    )
    parser.add_argument(
        "--image-shape",
        default="3,224,224",
        help="Comma-separated image shape C,H,W for vision models.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=128,
        help="Synthetic token sequence length for text models.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=30522,
        help="Vocabulary size for synthetic text inputs.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Execution device for PyTorch benchmarking, for example cpu or cuda.",
    )
    parser.add_argument(
        "--invocation-mode",
        choices=("auto", "positional", "keyword"),
        default="auto",
        help="Override how synthetic inputs are passed to eager PyTorch models.",
    )
    parser.add_argument(
        "--input-names",
        default=None,
        help="Comma-separated input names used for keyword invocation and ONNX export naming.",
    )
    parser.add_argument(
        "--model-kwargs-json",
        default=None,
        help=(
            "Inline JSON object or path to a JSON file with constructor kwargs for "
            "timm and python factory loaders."
        ),
    )
    parser.add_argument(
        "--hf-trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to Hugging Face config/model loading.",
    )
    parser.add_argument(
        "--providers",
        default="CPUExecutionProvider",
        help="Comma-separated ONNX Runtime providers, for example CPUExecutionProvider or CUDAExecutionProvider,CPUExecutionProvider.",
    )
    parser.add_argument(
        "--ort-optimization-level",
        choices=("disable", "basic", "extended", "all"),
        default="extended",
        help="Graph optimization level used when creating ONNX Runtime sessions.",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Load default pretrained weights for built-in torchvision/transformers models.",
    )
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Export a PyTorch/TorchScript model variant to ONNX and benchmark it with ONNX Runtime.",
    )
    parser.add_argument(
        "--onnx-path",
        type=Path,
        default=None,
        help="Destination path for ONNX export artifacts.",
    )
    parser.add_argument(
        "--onnx-quantized-path",
        type=Path,
        default=None,
        help="Destination path for quantized ONNX artifacts when the input model is already .onnx.",
    )
    parser.add_argument(
        "--onnx-opset",
        type=int,
        default=17,
        help="ONNX opset version used during export from PyTorch.",
    )
    parser.add_argument(
        "--onnx-quant-format",
        choices=("qdq", "qoperator"),
        default="qdq",
        help="Quantization format for ONNX static quantization.",
    )
    parser.add_argument(
        "--disable-onnx-preprocess",
        action="store_true",
        help="Skip ONNX graph pre-processing before ONNX Runtime quantization.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports") / "q_lab_report.csv",
        help="CSV report output path.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def parse_image_shape(raw_shape: str) -> Tuple[int, int, int]:
    parts = [segment.strip() for segment in raw_shape.split(",") if segment.strip()]
    if len(parts) != 3:
        raise ValueError("Image shape must be provided as C,H,W.")
    try:
        channels, height, width = (int(part) for part in parts)
    except ValueError as error:
        raise ValueError("Image shape values must be integers.") from error
    return channels, height, width


def parse_onnx_providers(raw_value: str) -> Tuple[str, ...]:
    providers = tuple(segment.strip() for segment in raw_value.split(",") if segment.strip())
    if not providers:
        raise ValueError("At least one ONNX Runtime provider must be supplied.")
    return providers


def parse_input_names(raw_value: str | None) -> Tuple[str, ...] | None:
    if raw_value is None:
        return None
    input_names = tuple(segment.strip() for segment in raw_value.split(",") if segment.strip())
    if not input_names:
        raise ValueError("Input names must contain at least one non-empty value.")
    return input_names


def parse_model_kwargs(raw_value: str | None) -> dict[str, Any]:
    if raw_value is None:
        return {}
    candidate_path = Path(raw_value)
    if candidate_path.exists():
        payload = candidate_path.read_text(encoding="utf-8")
    else:
        payload = raw_value
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError("--model-kwargs-json must be a JSON object or a path to a JSON file.") from error
    if not isinstance(parsed, dict):
        raise ValueError("--model-kwargs-json must decode to a JSON object.")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exit_error:
        return int(exit_error.code)

    try:
        from rich.console import Console
        from rich.traceback import install as install_rich_traceback

        from q_lab.benchmark import BenchmarkEngine
        from q_lab.models import infer_model_family, load_model
        from q_lab.reporting import render_results, save_results_csv
    except ImportError as error:
        parser.exit(
            1,
            f"Missing runtime dependency '{getattr(error, 'name', str(error))}'. "
            "Install dependencies with `python -m pip install -r requirements.txt`.\n",
        )

    install_rich_traceback(show_locals=False)
    console = Console()

    try:
        onnx_providers = parse_onnx_providers(args.providers)
        input_names_override = parse_input_names(args.input_names)
        model_kwargs = parse_model_kwargs(args.model_kwargs_json)
        invocation_mode_override = (
            None
            if args.invocation_mode == "auto"
            else InvocationMode(args.invocation_mode)
        )
        family = infer_model_family(
            args.model,
            args.task,
            hf_trust_remote_code=args.hf_trust_remote_code,
        )
        input_config = InputConfig(
            family=family,
            batch_size=args.batch_size,
            image_shape=parse_image_shape(args.image_shape),
            sequence_length=args.sequence_length,
            vocab_size=args.vocab_size,
        )
        benchmark_config = BenchmarkConfig(
            warmup_iterations=args.warmup_iterations,
            benchmark_iterations=args.benchmark_iterations,
            calibration_iterations=args.calibration_iterations,
            batch_size=args.batch_size,
            device=args.device,
            onnx_providers=onnx_providers,
            onnx_optimization_level=args.ort_optimization_level,
        )
        compression_config = CompressionConfig(
            quantization=QuantizationMode(args.quantization),
            pruning=PruningMode(args.pruning),
            pruning_amount=args.pruning_amount,
            export_onnx=args.export_onnx,
            onnx_path=_resolve_onnx_export_path(
                model_ref=args.model,
                requested_path=args.onnx_path,
            ),
            onnx_opset=args.onnx_opset,
            onnx_quantized_path=args.onnx_quantized_path,
            onnx_quant_format=args.onnx_quant_format,
            preprocess_onnx=not args.disable_onnx_preprocess,
        )
        model_format = infer_model_format(args.model)
        _validate_arguments(
            parser=parser,
            compression_config=compression_config,
            family=family,
            device=args.device,
            model_format=model_format,
        )
        engine = BenchmarkEngine(benchmark_config)

        with console.status("Loading model artifact and generating synthetic inputs..."):
            loaded_model = load_model(
                model_ref=args.model,
                input_config=input_config,
                device=benchmark_config.device,
                use_pretrained=args.pretrained,
                onnx_providers=benchmark_config.onnx_providers,
                onnx_optimization_level=benchmark_config.onnx_optimization_level,
                model_kwargs=model_kwargs,
                invocation_mode_override=invocation_mode_override,
                input_names_override=input_names_override,
                hf_trust_remote_code=args.hf_trust_remote_code,
            )

        if loaded_model.format is ModelFormat.ONNX:
            results, output_paths = _run_onnx_pipeline(
                console=console,
                engine=engine,
                loaded_model=loaded_model,
                compression_config=compression_config,
            )
        else:
            results, output_paths = _run_pytorch_pipeline(
                console=console,
                engine=engine,
                loaded_model=loaded_model,
                compression_config=compression_config,
            )

        render_results(console=console, results=results)
        save_results_csv(results=results, path=args.report_path)
        console.print(f"[bold green]CSV report saved to[/bold green] {args.report_path}")
        if output_paths.get("onnx_export") is not None:
            console.print(
                f"[bold green]ONNX artifact saved to[/bold green] {output_paths['onnx_export']}"
            )
        if output_paths.get("onnx_quantized") is not None:
            console.print(
                "[bold green]Quantized ONNX artifact saved to[/bold green] "
                f"{output_paths['onnx_quantized']}"
            )
        return 0
    except Exception as error:
        console.print(f"[bold red]Error:[/bold red] {error}")
        return 1


def _run_pytorch_pipeline(
    console,
    engine,
    loaded_model,
    compression_config: CompressionConfig,
):
    from q_lab.onnx_utils import create_onnx_session, export_to_onnx
    from q_lab.optimization import optimize_model

    results = []
    output_paths: dict[str, str | None] = {"onnx_export": None, "onnx_quantized": None}

    with console.status("Capturing PyTorch baseline outputs and running benchmark..."):
        baseline_reference = engine.capture_reference_outputs(
            model=loaded_model.model,
            loaded_model=loaded_model,
        )
        baseline_result = engine.benchmark_pytorch(
            label="baseline",
            model=loaded_model.model,
            loaded_model=loaded_model,
            size_mb=engine.measure_pytorch_size_mb(loaded_model.model),
            quantization=QuantizationMode.NONE.value,
            pruning=PruningMode.NONE.value,
            pruning_amount=0.0,
            sparsity_pct=0.0,
            reference_outputs=None,
            artifact_path=str(loaded_model.original_path) if loaded_model.original_path else None,
            notes="Reference eager/TorchScript baseline.",
        )
        results.append(baseline_result)

    model_for_export = loaded_model.model
    model_for_export_label = "baseline"
    export_quantization = QuantizationMode.NONE.value
    export_pruning = PruningMode.NONE.value
    export_pruning_amount = 0.0
    export_sparsity_pct = 0.0
    optimization_notes = ""

    if (
        compression_config.quantization is not QuantizationMode.NONE
        or compression_config.pruning is not PruningMode.NONE
    ):
        with console.status("Applying PyTorch pruning and quantization pipeline..."):
            optimization = optimize_model(
                loaded_model=loaded_model,
                compression=compression_config,
                benchmark_config=engine.config,
            )
            optimized_result = engine.benchmark_pytorch(
                label="optimized",
                model=optimization.model,
                loaded_model=loaded_model,
                size_mb=engine.measure_pytorch_size_mb(optimization.model),
                quantization=optimization.quantization.value,
                pruning=optimization.pruning.value,
                pruning_amount=optimization.pruning_amount,
                sparsity_pct=optimization.sparsity_pct,
                reference_outputs=baseline_reference,
                notes=" ".join(optimization.notes),
            )
            results.append(optimized_result)
            if optimization.quantization is QuantizationMode.NONE:
                model_for_export = optimization.model
                model_for_export_label = "optimized"
                export_quantization = optimization.quantization.value
                export_pruning = optimization.pruning.value
                export_pruning_amount = optimization.pruning_amount
                export_sparsity_pct = optimization.sparsity_pct
                optimization_notes = " ".join(optimization.notes)

    if compression_config.export_onnx and compression_config.onnx_path is not None:
        with console.status("Exporting ONNX artifact and benchmarking ONNX Runtime..."):
            exported_path = export_to_onnx(
                loaded_model=loaded_model,
                model=model_for_export,
                destination=compression_config.onnx_path,
                opset_version=compression_config.onnx_opset,
            )
            session = create_onnx_session(
                model_path=exported_path,
                providers=engine.config.onnx_providers,
                optimization_level=engine.config.onnx_optimization_level,
            )
            onnx_result = engine.benchmark_onnx(
                label=f"{model_for_export_label}-onnx",
                session=session,
                loaded_model=loaded_model,
                size_mb=engine.measure_file_size_mb(exported_path),
                quantization=export_quantization,
                pruning=export_pruning,
                pruning_amount=export_pruning_amount,
                sparsity_pct=export_sparsity_pct,
                reference_outputs=baseline_reference,
                artifact_path=str(exported_path),
                notes=(
                    f"ONNX Runtime benchmark for the {model_for_export_label} variant. "
                    f"{optimization_notes}"
                ).strip(),
            )
            results.append(onnx_result)
            output_paths["onnx_export"] = str(exported_path)

    return results, output_paths


def _run_onnx_pipeline(
    console,
    engine,
    loaded_model,
    compression_config: CompressionConfig,
):
    from q_lab.onnx_utils import create_onnx_session
    from q_lab.optimization import optimize_model

    results = []
    output_paths: dict[str, str | None] = {"onnx_export": None, "onnx_quantized": None}

    with console.status("Capturing ONNX Runtime baseline outputs and running benchmark..."):
        baseline_reference = engine.capture_onnx_reference_outputs(
            session=loaded_model.model,
            loaded_model=loaded_model,
        )
        baseline_result = engine.benchmark_onnx(
            label="baseline",
            session=loaded_model.model,
            loaded_model=loaded_model,
            size_mb=engine.measure_file_size_mb(loaded_model.original_path),
            quantization=QuantizationMode.NONE.value,
            pruning=PruningMode.NONE.value,
            pruning_amount=0.0,
            sparsity_pct=0.0,
            reference_outputs=None,
            artifact_path=str(loaded_model.original_path),
            notes="Reference ONNX Runtime baseline.",
        )
        results.append(baseline_result)

    if compression_config.quantization is not QuantizationMode.NONE:
        with console.status("Applying ONNX Runtime quantization pipeline..."):
            optimization = optimize_model(
                loaded_model=loaded_model,
                compression=compression_config,
                benchmark_config=engine.config,
            )
            optimized_session = create_onnx_session(
                model_path=optimization.artifact_path,
                providers=engine.config.onnx_providers,
                optimization_level=engine.config.onnx_optimization_level,
            )
            optimized_result = engine.benchmark_onnx(
                label="optimized",
                session=optimized_session,
                loaded_model=loaded_model,
                size_mb=engine.measure_file_size_mb(optimization.artifact_path),
                quantization=optimization.quantization.value,
                pruning=optimization.pruning.value,
                pruning_amount=optimization.pruning_amount,
                sparsity_pct=optimization.sparsity_pct,
                reference_outputs=baseline_reference,
                artifact_path=str(optimization.artifact_path),
                notes=" ".join(optimization.notes),
            )
            results.append(optimized_result)
            output_paths["onnx_quantized"] = str(optimization.artifact_path)

    return results, output_paths


def infer_model_format(model_ref: str) -> ModelFormat:
    model_path = Path(model_ref)
    if model_path.suffix.lower() == ".onnx":
        return ModelFormat.ONNX
    return ModelFormat.PYTORCH


def _resolve_onnx_export_path(model_ref: str, requested_path: Path | None) -> Path | None:
    if requested_path is not None:
        return requested_path
    model_path = Path(model_ref)
    artifact_name = f"{model_path.stem if model_path.suffix else model_ref}-export.onnx"
    return Path("artifacts") / artifact_name


def _validate_arguments(
    parser: argparse.ArgumentParser,
    compression_config: CompressionConfig,
    family: ModelFamily,
    device: str,
    model_format: ModelFormat,
) -> None:
    if compression_config.pruning is PruningMode.NONE and compression_config.pruning_amount != 0.0:
        parser.error("--pruning-amount must stay at 0.0 when --pruning none is used.")
    if compression_config.pruning is not PruningMode.NONE and compression_config.pruning_amount <= 0.0:
        parser.error("--pruning-amount must be > 0.0 when pruning is enabled.")
    if (
        compression_config.onnx_quantized_path is not None
        and compression_config.quantization is QuantizationMode.NONE
    ):
        parser.error("--onnx-quantized-path requires --quantization dynamic or static.")

    if model_format is ModelFormat.ONNX:
        if compression_config.pruning is not PruningMode.NONE:
            parser.error("Pruning is not supported for ONNX input models.")
        if compression_config.export_onnx:
            parser.error("ONNX input models are already deployment artifacts; remove --export-onnx.")
        return

    if compression_config.onnx_quantized_path is not None:
        parser.error("--onnx-quantized-path is only valid when the input model is an ONNX artifact.")
    if (
        compression_config.quantization is QuantizationMode.STATIC
        and family is not ModelFamily.VISION
    ):
        parser.error("Static quantization is only supported for vision models in this CLI.")
    if (
        compression_config.quantization is not QuantizationMode.NONE
        and not device.lower().startswith("cpu")
    ):
        parser.error("Quantized PyTorch benchmarking is supported only on CPU. Use --device cpu.")
    if (
        compression_config.export_onnx
        and compression_config.quantization is not QuantizationMode.NONE
    ):
        parser.error(
            "Exporting PyTorch-quantized models to ONNX is not supported. "
            "Export the baseline float model first, then run Q-Lab on the .onnx artifact."
        )
