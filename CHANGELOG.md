# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic Versioning.

## [1.3.0] - 2026-04-01

### Added

- Config-driven experiment matrix execution through the optional `matrix` section in JSON run configs.
- Label-aware evaluation metrics with `eval_top1_accuracy_pct`, `eval_macro_f1_pct`, and `eval_sample_count`.
- Dataset adapters for `.json`, `.jsonl`, `.csv`, and vision `ImageFolder` directories.
- Optional reproducibility manifests through `--manifest-path`.
- Optional standalone HTML reports through `--html-report-path`.

### Changed

- The dataset pipeline now understands labeled evaluation records instead of only unlabeled input batches.
- Benchmark rows now capture both fidelity-to-baseline signals and task-style evaluation metrics when labels are available.
- Input dataset flags accept generic data paths instead of only JSON-specific files, while the old JSON flag names remain as compatible aliases.

## [1.2.0] - 2026-04-01

### Added

- JSON run configuration support through `--config`, with explicit CLI flags overriding config values.
- Batch-size sweep support through `--batch-sizes`, including batched ONNX artifact naming for multi-run exports.
- Dataset-backed benchmark, calibration, and fidelity inputs through `--benchmark-inputs-json`, `--calibration-inputs-json`, and `--eval-inputs-json`.
- Throughput and peak-memory metrics in Rich reports and CSV exports.
- Task-aware Hugging Face loading through `--hf-auto-class` for generic models, classifiers, token classifiers, masked language models, and causal language models.
- Automated coverage for config expansion, input-data loading, Hugging Face auto-class selection, and dataset-driven CLI runs.

### Changed

- Extended the benchmark engine so timed loops can cycle over multiple input batches instead of a single synthetic example.
- Static quantization flows for both PyTorch and ONNX can now calibrate against user-provided datasets rather than only synthetic repeated inputs.
- Report tables and CSV exports now include `batch_size`, `throughput_items_per_sec`, and `peak_memory_mb`.

## [1.1.0] - 2026-03-31

### Added

- Support for arbitrary eager PyTorch models through `python:<module>:<factory>` references.
- Support for arbitrary eager PyTorch models through `pyfile:<path>::<factory>` references.
- Support for external `timm:<model>` architectures with full eager PyTorch optimization and export flows.
- Support for generic `hf:<model>` Hugging Face model loading for third-party architectures beyond the built-in registry.
- CLI controls for `--invocation-mode`, `--input-names`, `--model-kwargs-json`, and `--hf-trust-remote-code`.
- Integration and unit coverage for new external eager model sources.

### Changed

- Expanded the support matrix so full pruning, eager quantization, ONNX export, and ONNX Runtime benchmark functionality now applies to third-party eager PyTorch models, not only built-in models.
- Clarified that ONNX pruning remains intentionally unsupported and that raw checkpoint formats still require an architecture loader.

## [1.0.0] - 2026-03-30

### Added

- Multi-backend support for built-in PyTorch models, TorchScript `.pt` artifacts, and ONNX `.onnx` artifacts.
- ONNX Runtime input benchmarking with provider selection and optimization-level controls.
- ONNX Runtime dynamic and static quantization flows, including synthetic calibration support and optional graph pre-processing.
- Rich report output with execution-target visibility and CSV export for every benchmark row.
- Third-party TorchScript and ONNX integration coverage in the automated test suite.
- Packaging metadata through `pyproject.toml`, a Docker runtime image, and GitHub Actions CI.

### Changed

- Promoted Q-Lab to a stable `1.0.0` release with a documented support matrix and recommended deployment workflow.
- Clarified CLI argument semantics, supported inputs and outputs, and production limitations in the README.

### Notes

- ONNX pruning remains intentionally unsupported because sparse speedups require backend-specific runtime support and graph rewrites beyond the scope of a portable CLI.
