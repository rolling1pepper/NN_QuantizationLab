# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic Versioning.

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
