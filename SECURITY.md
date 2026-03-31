# Security Policy

## Supported Versions

Security fixes are applied to the most recent stable release line and the current development branch.

| Version | Supported |
| --- | --- |
| `1.x` | Yes |
| `main` / latest default branch | Yes |
| `< 1.0.0` | No |

## Reporting A Vulnerability

Please do not report security vulnerabilities through public GitHub issues, pull requests, or discussion threads.

If you believe you have found a security issue:

1. Share the report privately with the project maintainers through a non-public channel.
2. Include a clear description of the issue, affected versions, reproduction steps, and potential impact.
3. If possible, include a minimal proof of concept and any suggested remediation.

What to include in the report:

- affected command or workflow
- model type or artifact format involved
- exact Q-Lab version
- Python version and operating system
- whether the issue affects PyTorch, TorchScript, ONNX, or ONNX Runtime execution
- logs, traceback, or any relevant artifacts

What to expect from the maintainers:

- acknowledgement of the report after review
- triage of severity and affected scope
- a remediation plan when the issue is confirmed
- coordinated public disclosure after a fix is available, when appropriate

Please avoid publishing exploit details before maintainers have had a reasonable opportunity to investigate and release a fix.
