# Contributing To Q-Lab

Thanks for contributing to Q-Lab.

## Scope

Contributions are welcome for:

- bug fixes
- benchmark correctness improvements
- ONNX Runtime and PyTorch compatibility improvements
- test coverage
- documentation and examples

If a change is large, architectural, or user-visible, open an issue first so the direction can be agreed before implementation starts.

## Development Setup

Use Python `3.11` or `3.12`.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

If you prefer pinned dependencies:

```bash
python -m pip install -r requirements-dev.txt
```

## Branching And Commits

- Create a dedicated branch for each change.
- Keep pull requests focused on one topic.
- Write clear commit messages in the imperative mood.
- Do not mix unrelated refactors with functional fixes.

Good examples:

- `fix onnx static quantization calibration feed`
- `add integration coverage for third-party torchscript models`

## Code Standards

- Follow PEP 8 and keep type hints in place.
- Prefer small, composable functions over large monolithic blocks.
- Handle edge cases explicitly.
- Keep CLI behavior backward compatible unless the change is intentional and documented.
- Update README and tests when user-facing behavior changes.

## Testing

Run the full suite before opening a pull request:

```bash
pytest -q
```

Targeted runs are fine during development, but the final branch should pass the whole suite.

Useful subsets:

```bash
pytest -q -m integration
pytest -q tests\test_onnx_utils.py tests\test_onnx_reporting.py tests\test_cli.py -k onnx
```

If packaging metadata changes, also run:

```bash
python -m build
```

## Pull Request Checklist

Before opening a PR, confirm that:

- tests pass locally
- new behavior is covered by tests
- README or docs were updated if needed
- generated files and local artifacts are not committed unnecessarily
- the PR description explains the problem, the change, and any limitations

## Reporting Bugs

When filing a bug, include:

- the exact command you ran
- model type and backend
- expected behavior
- actual behavior
- relevant logs or traceback
- environment details such as Python version, OS, and CPU/GPU runtime

## Security

Do not open public issues for sensitive vulnerabilities. Follow the guidance in `SECURITY.md`.
