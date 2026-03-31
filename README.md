# Q-Lab

Q-Lab is a production-oriented CLI utility for profiling, pruning, quantizing, exporting, and validating neural-network inference artifacts across PyTorch, TorchScript, and ONNX Runtime. It measures latency with warmup, tracks serialized artifact size, estimates sparsity, compares optimized outputs against a reference baseline, prints a Rich console report, and saves the same results to CSV.

## What Q-Lab Supports

| Input type | Baseline benchmark | Pruning | Quantization | ONNX export | ONNX Runtime benchmark |
| --- | --- | --- | --- | --- | --- |
| Built-in PyTorch models (`resnet18`, `resnet50`, `mobilenet_v3_small`, `bert`) | Yes | Yes | Yes | Yes | Yes |
| Arbitrary eager PyTorch via `python:<module>:<factory>` | Yes | Yes | Yes | Yes | Yes |
| Arbitrary eager PyTorch via `pyfile:<path>::<factory>` | Yes | Yes | Yes | Yes | Yes |
| Third-party `timm:<model>` architectures | Yes | Yes | Yes | Yes | Yes |
| Third-party `hf:<model>` architectures | Yes | Yes, when prunable layers exist | Yes | Yes | Yes |
| TorchScript `.pt` | Yes | No | No | Yes | Yes |
| ONNX `.onnx` | Yes | No | Yes, via ONNX Runtime | Not needed | Yes |

Key constraints:

- PyTorch pruning and PyTorch quantization are supported for eager built-in models.
- The same eager PyTorch optimization pipeline is now available for third-party eager models loaded through Python factories, `timm`, and generic Hugging Face references.
- ONNX input models use a separate ONNX Runtime quantization pipeline.
- ONNX pruning is intentionally not implemented because dense ONNX runtimes usually do not benefit from zeroed weights without backend-specific sparse execution.
- Exporting already-quantized PyTorch models to ONNX is not supported. Export the float model first, then run Q-Lab on the exported `.onnx` artifact.
- Raw checkpoint formats such as bare `state_dict` files are supported through an architecture loader, not as standalone direct inputs.

## Installation

Python `3.11` or `3.12` is recommended for the pinned dependency set in this repository.

Install from source as a package:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install .
```

Install with exact pinned dependency files:

```bash
python -m pip install -r requirements.txt
```

For development, tests, and packaging checks:

```bash
python -m pip install -e .[dev]
```

If you prefer pinned dev dependencies instead of extras:

```bash
python -m pip install -r requirements-dev.txt
```

After installation, the CLI is available both as a module and as a console entry point:

```bash
q-lab --help
python -m q_lab --help
```

## How It Works

Each CLI run follows the same high-level flow:

1. Load a model from a built-in registry, a TorchScript `.pt` artifact, or an ONNX `.onnx` artifact.
2. Generate synthetic inputs for the selected workload family (`vision` or `text`).
3. Run warmup iterations, then benchmark timed inference iterations.
4. Measure serialized artifact size.
5. Optionally apply pruning and quantization.
6. Optionally export a PyTorch/TorchScript model to ONNX and benchmark the exported artifact.
7. Compare optimized outputs against the baseline with cosine similarity, max absolute difference, and a simple prediction-agreement proxy.
8. Render a Rich table in the console and save all rows to CSV.

## CLI Usage

```bash
python -m q_lab --help
```

### Inputs

Q-Lab accepts:

- A built-in model name such as `resnet50` or `bert`
- A Python factory reference such as `python:my_package.models:create_model`
- A Python file factory reference such as `pyfile:C:\models\factory.py::create_model`
- A `timm` model reference such as `timm:convnext_tiny`
- A Hugging Face model reference such as `hf:google/vit-base-patch16-224`
- A TorchScript path such as `models\classifier.pt`
- An ONNX path such as `artifacts\encoder.onnx`
- Synthetic input controls such as `--image-shape`, `--sequence-length`, `--vocab-size`, and `--batch-size`

### Outputs

Each run produces:

- A Rich console table with benchmark rows such as `baseline`, `optimized`, or `baseline-onnx`
- A CSV report saved to `--report-path`
- An ONNX export artifact when `--export-onnx` is used on PyTorch/TorchScript input
- A quantized ONNX artifact when `--quantization dynamic|static` is used on ONNX input

CSV/report columns include:

- `label`
- `backend`
- `execution_target`
- `mean_latency_ms`
- `p95_latency_ms`
- `size_mb`
- `sparsity_pct`
- `accuracy_proxy_pct`
- `cosine_similarity`
- `max_abs_diff`
- `artifact_path`
- `notes`

## Argument Reference

| Argument | Description |
| --- | --- |
| `model` | Built-in model name, TorchScript `.pt`, or ONNX `.onnx` input |
| `--task {auto,vision,text}` | Select or infer the workload family |
| `--quantization {none,static,dynamic}` | Choose quantization mode |
| `--pruning {none,structured,unstructured}` | Choose pruning mode for eager PyTorch models |
| `--pruning-amount` | Pruning ratio in `[0.0, 1.0)` |
| `--warmup-iterations` | Number of untimed warmup passes |
| `--benchmark-iterations` | Number of timed inference passes |
| `--calibration-iterations` | Number of synthetic calibration batches for static quantization |
| `--batch-size` | Synthetic batch size |
| `--image-shape C,H,W` | Synthetic image input shape for vision models |
| `--sequence-length` | Synthetic token length for text models |
| `--vocab-size` | Token sampling range for synthetic text inputs |
| `--device` | PyTorch execution device such as `cpu` or `cuda` |
| `--invocation-mode {auto,positional,keyword}` | Override how synthetic inputs are passed to eager models |
| `--input-names` | Comma-separated eager-model input names |
| `--model-kwargs-json` | JSON object or JSON file path with constructor kwargs for `timm` and Python factory loaders |
| `--hf-trust-remote-code` | Enable `trust_remote_code=True` for Hugging Face loading |
| `--providers` | Comma-separated ONNX Runtime providers |
| `--ort-optimization-level` | ORT graph optimization level: `disable`, `basic`, `extended`, or `all` |
| `--pretrained` | Load default pretrained weights for built-in models |
| `--export-onnx` | Export a PyTorch or TorchScript variant to ONNX and benchmark it with ORT |
| `--onnx-path` | Output path for exported ONNX artifacts |
| `--onnx-quantized-path` | Output path for quantized ONNX artifacts |
| `--onnx-opset` | Opset used for PyTorch-to-ONNX export |
| `--onnx-quant-format {qdq,qoperator}` | Static ONNX quantization output format |
| `--disable-onnx-preprocess` | Skip ONNX graph pre-processing before ORT quantization |
| `--report-path` | CSV output path |

## Examples

Benchmark a pretrained PyTorch ResNet-50 baseline:

```bash
python -m q_lab resnet50 --pretrained --warmup-iterations 10 --benchmark-iterations 50 --report-path reports\resnet50.csv
```

Apply PyTorch static quantization to a vision model:

```bash
python -m q_lab resnet18 --quantization static --calibration-iterations 12 --warmup-iterations 5 --benchmark-iterations 25 --report-path reports\resnet18_static.csv
```

Benchmark and export a third-party TorchScript artifact:

```bash
python -m q_lab external_models\custom_backbone.pt --task vision --image-shape 3,224,224 --export-onnx --onnx-path artifacts\custom_backbone.onnx --report-path reports\custom_backbone.csv
```

Run ONNX Runtime dynamic quantization on an existing ONNX text model:

```bash
python -m q_lab artifacts\encoder.onnx --quantization dynamic --sequence-length 128 --vocab-size 30522 --providers CPUExecutionProvider --onnx-quantized-path artifacts\encoder-dynamic.onnx --report-path reports\encoder_dynamic.csv
```

Run ONNX Runtime static quantization on an existing ONNX vision model:

```bash
python -m q_lab artifacts\resnet18-export.onnx --quantization static --image-shape 3,224,224 --calibration-iterations 8 --onnx-quant-format qdq --onnx-quantized-path artifacts\resnet18-static.onnx --report-path reports\resnet18_onnx_static.csv
```

Use a specific ONNX Runtime provider stack:

```bash
python -m q_lab artifacts\model.onnx --providers CUDAExecutionProvider,CPUExecutionProvider --ort-optimization-level all --report-path reports\ort_cuda.csv
```

Benchmark and optimize an arbitrary eager PyTorch model exposed from a Python module:

```bash
python -m q_lab python:my_package.models:create_model --task vision --pruning structured --pruning-amount 0.25 --quantization static --image-shape 3,224,224 --model-kwargs-json '{"num_classes": 1000}' --report-path reports\python_factory.csv
```

Benchmark an eager PyTorch model from a standalone Python file:

```bash
python -m q_lab pyfile:C:\models\factory.py::create_model --task text --quantization dynamic --sequence-length 128 --vocab-size 30522 --report-path reports\python_file.csv
```

Benchmark and export a third-party timm architecture:

```bash
python -m q_lab timm:convnext_tiny --pretrained --task vision --export-onnx --onnx-path artifacts\convnext_tiny.onnx --report-path reports\convnext_tiny.csv
```

Benchmark a generic Hugging Face architecture:

```bash
python -m q_lab hf:google/vit-base-patch16-224 --pretrained --task vision --input-names pixel_values --invocation-mode keyword --report-path reports\hf_vit.csv
```

## Recommended Workflow

If your end target is ONNX deployment, use a two-step flow:

1. Export a float PyTorch or TorchScript model to ONNX.
2. Re-run Q-Lab on the exported `.onnx` artifact with ONNX Runtime quantization enabled.

Example:

```bash
python -m q_lab resnet18 --export-onnx --onnx-path artifacts\resnet18-export.onnx --report-path reports\resnet18_export.csv
python -m q_lab artifacts\resnet18-export.onnx --quantization static --image-shape 3,224,224 --calibration-iterations 8 --report-path reports\resnet18_ort_quant.csv
```

## Testing

Run the full automated suite:

```bash
pytest -q
```

Run only integration scenarios:

```bash
pytest -q -m integration
```

Run only the ONNX-focused scenarios:

```bash
pytest -q tests\test_onnx_utils.py tests\test_onnx_reporting.py tests\test_cli.py -k onnx
```

Build distributable artifacts locally:

```bash
python -m build
```

## Docker

Build the runtime image:

```bash
docker build -t q-lab:latest .
```

Run the CLI inside the container:

```bash
docker run --rm -v ${PWD}/reports:/app/reports -v ${PWD}/artifacts:/app/artifacts q-lab:latest --help
```

Example containerized benchmark:

```bash
docker run --rm -v ${PWD}/reports:/app/reports q-lab:latest resnet18 --benchmark-iterations 5 --warmup-iterations 2 --report-path /app/reports/resnet18.csv
```

## CI And Release Hygiene

The repository now includes:

- `pyproject.toml` for package metadata and the `q-lab` console script
- `.github/workflows/ci.yml` for automated install, compile, test, and build validation on Python `3.11` and `3.12`
- `.github/dependabot.yml` for automated dependency update proposals
- `.github/ISSUE_TEMPLATE/*` and `.github/pull_request_template.md` for standardized collaboration
- `CONTRIBUTING.md` for contribution rules and local validation steps
- `SECURITY.md` for private vulnerability reporting guidance
- `CODE_OF_CONDUCT.md` for community participation standards
- `RELEASE_CHECKLIST.md` for maintainers preparing a public release
- `Dockerfile` for reproducible runtime packaging
- `MANIFEST.in` for predictable source-distribution contents
- `.editorconfig` for consistent editor defaults
- `CHANGELOG.md` for release tracking
- `.gitignore` and `.dockerignore` to keep the repository and container context clean

## License

Q-Lab is distributed under the MIT License. See `LICENSE` for the full text.

## Notes

- Built-in models use random weights by default to avoid implicit downloads. Add `--pretrained` when you want default pretrained checkpoints.
- Third-party eager PyTorch models loaded through `python:`, `pyfile:`, `timm:`, and `hf:` now use the same pruning, quantization, export, and benchmark pipeline as built-in eager models.
- Static quantization in both PyTorch and ONNX paths uses synthetic calibration data in this repository. That is sufficient for pipeline validation, but real production calibration should use representative data.
- `Accuracy Proxy %` is a regression signal computed on synthetic inputs. It is not a replacement for task accuracy on labeled datasets.
- ONNX input family inference uses graph metadata and common input-name heuristics. When inference is ambiguous, pass `--task vision` or `--task text`.
- Pruning only affects layers with explicit dense weights such as `Conv2d` and `Linear`. Models without prunable layers will report that limitation instead of silently pretending to optimize.
