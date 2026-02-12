# Contributing to DataWhisperer

Thanks for your interest. Here's how to help.

## Reporting Bugs

Open an issue on GitHub with:

- What you did (steps to reproduce)
- What you expected
- What actually happened
- Your environment (OS, Python version, Docker version)
- If possible, attach the dataset that caused the issue (or a minimal example)

## Code Style

- Follow the existing patterns in the codebase
- Use type hints where they add clarity
- Docstrings for public functions only
- Use `logging` instead of `print`
- Keep functions under ~50 lines
- Use f-strings for formatting

## Submitting PRs

1. Fork the repo and create a branch from `main`
2. Make your changes
3. Add or update tests for any new functionality
4. Run the test suite: `pytest tests/`
5. Make sure your code doesn't break existing tests
6. Open a PR with a clear description of the change

## Testing

- Use pytest
- Mock external services (Ollama, file I/O) in tests
- Aim to test critical paths, not trivial getters
- Run tests with: `pytest tests/ -v`

## Development Setup

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pytest tests/ -v
```

## What to Work On

Check the GitHub issues for open tasks. Good first contributions:

- Adding new visualization types
- Improving time series analysis
- Better error messages
- Documentation improvements
- New sample datasets
