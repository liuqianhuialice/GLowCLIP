# Contributing

Contributions should keep the test split isolated from model, hyperparameter, and
threshold selection.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
make check
```

## Pull requests

1. Add tests for behavior changes.
2. Run `make check`.
3. Document configuration or CLI changes in `README.md`.
4. Report validation metrics separately from untouched test metrics.
5. Do not commit datasets, model caches, predictions, or checkpoints.
