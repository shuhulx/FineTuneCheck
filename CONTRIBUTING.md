# Contributing to FineTuneCheck

Thanks for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/shuhulx/finetunecheck.git
cd finetunecheck
pip install -e ".[dev]"
pre-commit install
```

## Running Tests

```bash
make test        # run tests
make lint        # run linters
make format      # auto-format
make ci          # full CI check
```

## Pull Requests

1. Fork the repo and create a branch
2. Make your changes
3. Run `make ci` to verify
4. Submit a PR

Please open an issue first for major changes.
