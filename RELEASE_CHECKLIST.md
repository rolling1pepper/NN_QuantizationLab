# Release Checklist

Use this checklist before cutting a public Q-Lab release.

## Validation

- [ ] Run `pytest -q`
- [ ] Run `python -m build`
- [ ] Run a smoke CLI check with at least one PyTorch model and one ONNX model
- [ ] Confirm README examples still match the current CLI

## Documentation

- [ ] Update `CHANGELOG.md`
- [ ] Update version in `q_lab/__init__.py`
- [ ] Update version in `pyproject.toml`
- [ ] Review `README.md` for any user-facing changes

## Repository Hygiene

- [ ] Confirm `LICENSE`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, and `CONTRIBUTING.md` are current
- [ ] Ensure issue and PR templates still reflect the current workflow
- [ ] Verify CI passes on supported Python versions

## Packaging

- [ ] Inspect built `sdist` and wheel artifacts
- [ ] Confirm package metadata is correct
- [ ] Confirm console entry point `q-lab` works after installation
