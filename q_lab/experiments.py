from __future__ import annotations

import itertools
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


SUPPORTED_MATRIX_KEYS = {
    "batch_size",
    "batch_sizes",
    "device",
    "export_onnx",
    "hf_auto_class",
    "ort_optimization_level",
    "pretrained",
    "providers",
    "pruning",
    "pruning_amount",
    "quantization",
}


def extract_config_path(argv: Sequence[str] | None) -> Path | None:
    tokens = list(argv or [])
    for index, token in enumerate(tokens):
        if token == "--config" and index + 1 < len(tokens):
            return Path(tokens[index + 1])
        if token.startswith("--config="):
            return Path(token.split("=", 1)[1])
    return None


def collect_explicit_overrides(parser, argv: Sequence[str] | None) -> tuple[set[str], bool]:
    option_to_dest = {}
    zero_arg_options = set()
    for action in parser._actions:
        for option in action.option_strings:
            option_to_dest[option] = action.dest
            if action.nargs == 0:
                zero_arg_options.add(option)

    explicit_dests: set[str] = set()
    explicit_model = False
    tokens = list(argv or [])
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("--"):
            option_name = token.split("=", 1)[0]
            destination = option_to_dest.get(option_name)
            if destination is not None:
                explicit_dests.add(destination)
            if "=" in token or option_name in zero_arg_options:
                index += 1
            else:
                index += 2
            continue

        explicit_model = True
        index += 1

    return explicit_dests, explicit_model


def build_run_matrix(
    args,
    raw_config: dict[str, Any] | None,
    explicit_overrides: set[str],
    batch_size_values: Sequence[int],
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    matrix_config = dict((raw_config or {}).get("matrix", {}) or {})
    if not isinstance(matrix_config, dict):
        raise ValueError("The optional 'matrix' config section must be a JSON object.")

    axes: dict[str, list[Any]] = {}
    if len(batch_size_values) > 1:
        axes["batch_size"] = list(batch_size_values)

    if (
        "batch_size" not in explicit_overrides
        and "batch_sizes" not in explicit_overrides
        and "batch_size" in matrix_config
    ):
        axes["batch_size"] = _coerce_matrix_axis("batch_size", matrix_config["batch_size"])
    if (
        "batch_size" not in explicit_overrides
        and "batch_sizes" not in explicit_overrides
        and "batch_sizes" in matrix_config
    ):
        axes["batch_size"] = _coerce_matrix_axis("batch_size", matrix_config["batch_sizes"])

    for key, raw_axis in matrix_config.items():
        if key in {"batch_size", "batch_sizes"}:
            continue
        if key not in SUPPORTED_MATRIX_KEYS:
            raise ValueError(
                f"Unsupported matrix axis '{key}'. Supported axes: {', '.join(sorted(SUPPORTED_MATRIX_KEYS))}."
            )
        if key in explicit_overrides:
            continue
        axes[key] = _coerce_matrix_axis(key, raw_axis)

    if not axes:
        return [{}], tuple()

    varying_keys = tuple(sorted(axes))
    runs = [
        dict(zip(varying_keys, values))
        for values in itertools.product(*(axes[key] for key in varying_keys))
    ]
    return runs, varying_keys


def build_run_suffix(
    run_overrides: dict[str, Any],
    varying_keys: Sequence[str],
) -> str:
    if not varying_keys:
        return ""
    parts = []
    for key in varying_keys:
        value = run_overrides.get(key)
        if value is None:
            continue
        normalized_key = key.replace("_", "-")
        normalized_value = _sanitize_component(str(value).replace(",", "_"))
        parts.append(f"{normalized_key}-{normalized_value}")
    if not parts:
        return ""
    return "-".join(parts)


def build_manifest(
    *,
    args,
    raw_config: dict[str, Any] | None,
    run_matrix_overrides: Sequence[dict[str, Any]],
    varying_keys: Sequence[str],
    results: Sequence[Any],
    report_path: Path,
    html_report_path: Path | None,
    output_paths: dict[str, str | None],
    q_lab_version: str,
) -> dict[str, Any]:
    return {
        "q_lab_version": q_lab_version,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cwd": str(Path.cwd()),
        "python": platform.python_version(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "libraries": _collect_library_versions(),
        "git": _collect_git_metadata(),
        "cli_args": {
            key: _serialize_manifest_value(value)
            for key, value in vars(args).items()
        },
        "raw_config": raw_config or {},
        "matrix": {
            "varying_keys": list(varying_keys),
            "runs": [
                {key: _serialize_manifest_value(value) for key, value in run.items()}
                for run in run_matrix_overrides
            ],
        },
        "artifacts": {
            "report_csv": str(report_path),
            "report_html": None if html_report_path is None else str(html_report_path),
            "onnx_export": output_paths.get("onnx_export"),
            "onnx_quantized": output_paths.get("onnx_quantized"),
        },
        "results": [result.to_record() for result in results],
    }


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def _coerce_matrix_axis(key: str, raw_axis: Any) -> list[Any]:
    if not isinstance(raw_axis, list) or not raw_axis:
        raise ValueError(f"Matrix axis '{key}' must be a non-empty JSON array.")
    if key == "batch_size":
        values = [int(value) for value in raw_axis]
        if any(value <= 0 for value in values):
            raise ValueError("Matrix batch_size values must all be > 0.")
        return values
    if key == "pruning_amount":
        values = [float(value) for value in raw_axis]
        if any(not 0.0 <= value < 1.0 for value in values):
            raise ValueError("Matrix pruning_amount values must stay within [0.0, 1.0).")
        return values
    if key in {"pretrained", "export_onnx"}:
        return [bool(value) for value in raw_axis]
    if key == "providers":
        return [
            ",".join(str(item) for item in value) if isinstance(value, (list, tuple)) else str(value)
            for value in raw_axis
        ]
    return [str(value) for value in raw_axis]


def _collect_library_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for module_name in ("torch", "torchvision", "transformers", "onnx", "onnxruntime", "pandas"):
        try:
            module = __import__(module_name)
        except Exception:
            versions[module_name] = None
            continue
        versions[module_name] = getattr(module, "__version__", None)
    return versions


def _collect_git_metadata() -> dict[str, Any]:
    commit = _run_git_command(("git", "rev-parse", "HEAD"))
    status = _run_git_command(("git", "status", "--porcelain"))
    return {
        "commit": commit,
        "dirty": bool(status.strip()) if status is not None else None,
    }


def _run_git_command(command: Sequence[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            cwd=Path.cwd(),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _serialize_manifest_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_serialize_manifest_value(item) for item in value]
    if isinstance(value, list):
        return [_serialize_manifest_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _serialize_manifest_value(item)
            for key, item in value.items()
        }
    return value


def _sanitize_component(value: str) -> str:
    sanitized = value.replace("\\", "-").replace("/", "-").replace(" ", "-")
    return "".join(character for character in sanitized if character.isalnum() or character in {"-", "_", "."})
