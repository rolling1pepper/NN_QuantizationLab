# Q-Lab

Q-Lab is a production-oriented CLI utility for profiling, pruning, quantizing, exporting, and validating neural-network inference artifacts across PyTorch, TorchScript, and ONNX Runtime. It measures latency with warmup, reports throughput and peak memory, tracks serialized artifact size, estimates sparsity, compares optimized outputs against a reference baseline, computes label-aware evaluation metrics when datasets provide labels, prints a Rich console report, and saves the same results to CSV, HTML, and reproducibility manifests.

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
- Generic config-driven runs, experiment matrices, and dataset-backed benchmark/calibration/evaluation flows are supported through JSON files.

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
2. Generate synthetic inputs for the selected workload family (`vision` or `text`) or load explicit benchmark/calibration/evaluation inputs from JSON, JSONL, CSV, or vision image folders.
3. Run warmup iterations, then benchmark timed inference iterations, optionally sweeping across a config-driven experiment matrix.
4. Measure serialized artifact size, throughput, and peak host or CUDA memory.
5. Optionally apply pruning and quantization.
6. Optionally export a PyTorch/TorchScript model to ONNX and benchmark the exported artifact.
7. Compare optimized outputs against the baseline with cosine similarity, max absolute difference, and a simple prediction-agreement proxy.
8. If labels are available, compute classification-style evaluation metrics such as top-1 accuracy and macro F1.
9. Render a Rich table in the console and save all rows to CSV, plus optional HTML and manifest artifacts.

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
- A JSON run config through `--config`
- Optional dataset paths for benchmark, calibration, and evaluation inputs

### Outputs

Each run produces:

- A Rich console table with benchmark rows such as `baseline`, `optimized`, or `baseline-onnx`
- A CSV report saved to `--report-path`
- An HTML report saved to `--html-report-path` when requested
- A reproducibility manifest saved to `--manifest-path` when requested
- An ONNX export artifact when `--export-onnx` is used on PyTorch/TorchScript input
- A quantized ONNX artifact when `--quantization dynamic|static` is used on ONNX input

CSV/report columns include:

- `label`
- `backend`
- `execution_target`
- `mean_latency_ms`
- `p95_latency_ms`
- `throughput_items_per_sec`
- `peak_memory_mb`
- `size_mb`
- `sparsity_pct`
- `accuracy_proxy_pct`
- `eval_top1_accuracy_pct`
- `eval_macro_f1_pct`
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
| `--batch-sizes` | Comma-separated batch-size sweep such as `1,4,8` |
| `--image-shape C,H,W` | Synthetic image input shape for vision models |
| `--sequence-length` | Synthetic token length for text models |
| `--vocab-size` | Token sampling range for synthetic text inputs |
| `--device` | PyTorch execution device such as `cpu` or `cuda` |
| `--invocation-mode {auto,positional,keyword}` | Override how synthetic inputs are passed to eager models |
| `--input-names` | Comma-separated eager-model input names |
| `--model-kwargs-json` | JSON object or JSON file path with constructor kwargs for `timm` and Python factory loaders |
| `--benchmark-data-path` | Dataset path used for timed inference batches |
| `--calibration-data-path` | Dataset path used for static quantization calibration |
| `--eval-data-path` | Dataset path used for baseline-versus-optimized fidelity and label-aware evaluation |
| `--hf-trust-remote-code` | Enable `trust_remote_code=True` for Hugging Face loading |
| `--hf-auto-class` | Select the Hugging Face auto model class for `hf:<model>` inputs |
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
| `--html-report-path` | HTML report output path |
| `--manifest-path` | Reproducibility manifest JSON output path |

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

Run a config-driven batch-size sweep:

```bash
python -m q_lab --config configs\resnet18_static.json --report-path reports\resnet18_sweep.csv
```

Run a config-driven experiment matrix and emit HTML plus a manifest:

```bash
python -m q_lab --config configs\vit_matrix.json --report-path reports\vit_matrix.csv --html-report-path reports\vit_matrix.html --manifest-path reports\vit_matrix_manifest.json
```

Benchmark and export a third-party timm architecture:

```bash
python -m q_lab timm:convnext_tiny --pretrained --task vision --export-onnx --onnx-path artifacts\convnext_tiny.onnx --report-path reports\convnext_tiny.csv
```

Benchmark a generic Hugging Face architecture:

```bash
python -m q_lab hf:google/vit-base-patch16-224 --pretrained --task vision --input-names pixel_values --invocation-mode keyword --report-path reports\hf_vit.csv
```

Benchmark a Hugging Face sequence-classification model with the task-aware loader:

```bash
python -m q_lab hf:distilbert-base-uncased --pretrained --hf-auto-class sequence-classification --task text --quantization dynamic --report-path reports\hf_classifier.csv
```

Run dataset-backed static quantization, fidelity evaluation, and label-aware metrics:

```bash
python -m q_lab resnet18 --quantization static --benchmark-data-path data\vision_samples.json --calibration-data-path data\vision_samples.json --eval-data-path data\vision_samples.json --report-path reports\dataset_driven.csv
```

## JSON Config And Dataset Formats

Example run config:

```json
{
  "model": "resnet18",
  "quantization": "static",
  "image_shape": "3,224,224",
  "warmup_iterations": 5,
  "benchmark_iterations": 30,
  "calibration_iterations": 8,
  "batch_sizes": [1, 4, 8],
  "html_report_path": "reports/resnet18_sweep.html",
  "manifest_path": "reports/resnet18_sweep_manifest.json",
  "report_path": "reports/resnet18_sweep.csv"
}
```

Example config with an experiment matrix:

```json
{
  "model": "resnet18",
  "task": "vision",
  "matrix": {
    "batch_size": [1, 4],
    "quantization": ["none", "static"],
    "providers": ["CPUExecutionProvider"]
  },
  "report_path": "reports/resnet18_matrix.csv"
}
```

Example input dataset for a vision model:

```json
{
  "samples": [
    {
      "input": [[[0.0, 0.0], [0.0, 0.0]]]
    }
  ]
}
```

Example input dataset for a text model:

```json
[
  {
    "input_ids": [101, 2023, 2003, 1037, 3231],
    "attention_mask": [1, 1, 1, 1, 1],
    "token_type_ids": [0, 0, 0, 0, 0]
  }
]
```

Example CSV text dataset with labels:

```csv
input_ids,attention_mask,token_type_ids,label
"[101,2023,2003,1037,3231]","[1,1,1,1,1]","[0,0,0,0,0]",1
```

Dataset notes:

- Supported dataset inputs are `.json`, `.jsonl`, `.csv`, and vision image-folder directories.
- Top-level JSON can be either a list of samples or an object with a `samples` array.
- Each sample can be a mapping keyed by input name or a positional list matching the model input order.
- Samples can include an optional `label` field for evaluation.
- If a sample omits the batch dimension, Q-Lab promotes it to batch size `1`.
- Non-batch dimensions must match the loaded model input template.
- Vision image folders follow the usual class-subdirectory layout: one folder per label class, containing image files.

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
- When `--benchmark-data-path`, `--calibration-data-path`, or `--eval-data-path` are supplied, Q-Lab uses those user-provided samples instead of repeating one synthetic tensor.
- `--batch-sizes` repeats the full run for each requested batch size and appends batch-specific suffixes to generated ONNX artifacts when needed.
- The optional `matrix` config section expands one run config into a Cartesian product of experiment variants.
- `Accuracy Proxy %` is still a regression signal against the baseline, while `Eval Acc %` and `Eval F1 %` become available when labels are present.
- ONNX input family inference uses graph metadata and common input-name heuristics. When inference is ambiguous, pass `--task vision` or `--task text`.
- Pruning only affects layers with explicit dense weights such as `Conv2d` and `Linear`. Models without prunable layers will report that limitation instead of silently pretending to optimize.
