# Contributing to FineTuneCheck

Thanks for taking the time to help. Bug fixes, clearer docs, new probes, and careful test cases
are all welcome.

## Development Setup

```bash
git clone https://github.com/shuhulx/finetunecheck.git
cd finetunecheck
pip install -e ".[dev,deep,mcp]"
pre-commit install
```

## Running Tests

```bash
make test        # run the test suite
make lint        # run Ruff
make format      # format the code
make ci          # run the local lint and test checks
```

## Pull Requests

1. Fork the repository and create a branch.
2. Make one focused change.
3. Add or update tests when behavior changes.
4. Run `make ci`.
5. Open a pull request and explain the problem your change solves.

For a large change, open an issue first so we can agree on the direction before you spend a lot
of time on it.
