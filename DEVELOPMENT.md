# Development Setup Guide

This guide explains how to set up the development environment for the OASIS Agentic Pipeline project.

## Prerequisites

- Python 3.10 or higher
- Git
- Virtual environment (recommended)

## Quick Setup

Run the automatic setup script:

```bash
python scripts/setup_dev.py
```

This script will:
1. Install development dependencies
2. Set up pre-commit hooks
3. Run initial code formatting
4. Perform security scanning

## Manual Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 3. Install Pre-commit Hooks

```bash
pre-commit install
```

## Development Tools

### Code Quality

The project uses **Ruff** for linting and formatting:

```bash
# Check code quality
ruff check .

# Auto-fix issues
ruff check . --fix

# Format code
ruff format .

# Check formatting without changes
ruff format --check .
```

### Security Scanning

Security tools are included in the development dependencies:

```bash
# Run Bandit security scanner
bandit -r src/

# Run Safety dependency scanner
safety check

# Run pip-audit
pip-audit
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_security.py

# Run with verbose output
pytest -v

# Run only unit tests
pytest -m unit

# Skip slow tests
pytest -m "not slow"
```

### Pre-commit Hooks

Pre-commit hooks automatically run before each commit:

- Ruff linting and formatting
- Security scanning (Bandit, Safety)
- File checks (trailing whitespace, JSON/YAML validation)
- Type checking (mypy)

To run hooks manually:

```bash
# Run all hooks on all files
pre-commit run --all-files

# Run specific hook
pre-commit run ruff --all-files
```

## Code Style

The project follows these code style guidelines:

- **Line length**: 100 characters
- **Indentation**: 4 spaces
- **Quotes**: Double quotes
- **Imports**: Sorted and grouped (handled by Ruff)

## Testing Strategy

The test suite includes:

1. **Unit Tests**: Individual component testing
2. **Integration Tests**: Multi-component workflow testing
3. **Security Tests**: Authentication, validation, and security middleware
4. **Performance Tests**: Benchmarking and optimization validation
5. **Data Validation Tests**: Input sanitization and validation

Test files are located in the `tests/` directory and follow the naming convention `test_*.py`.

## Security Best Practices

1. **Never commit secrets**: Use environment variables for sensitive data
2. **Run security scans**: Use the included security tools before committing
3. **Validate inputs**: All user inputs must be validated using the validators in `src/api/validators.py`
4. **Use authentication**: Protected endpoints require authentication
5. **Follow OWASP guidelines**: Security headers and best practices are enforced

## CI/CD Pipeline

The project includes a comprehensive CI/CD pipeline in `.github/workflows/ci-cd.yml`:

1. **Linting**: Ruff checks
2. **Security**: Bandit, Safety, pip-audit
3. **Testing**: Pytest with coverage
4. **Performance**: Benchmark tests
5. **Build**: Docker image building

## Troubleshooting

### Pre-commit hooks fail

If pre-commit hooks fail, you can:

1. Fix the issues they report
2. Skip hooks temporarily (not recommended): `git commit --no-verify`
3. Update hooks: `pre-commit autoupdate`

### Ruff formatting issues

If Ruff formatting causes issues:

1. Check `pyproject.toml` for configuration
2. Run `ruff format .` to fix formatting
3. Disable specific rules in `pyproject.toml` if needed

### Test failures

If tests fail:

1. Run tests with verbose output: `pytest -v`
2. Check test logs in `pytest.log`
3. Ensure all dependencies are installed
4. Verify environment variables are set

## Contributing

1. Create a feature branch
2. Make your changes
3. Run tests: `pytest`
4. Run linting: `ruff check . --fix`
5. Format code: `ruff format .`
6. Commit (pre-commit hooks will run automatically)
7. Push and create a pull request

## Additional Resources

- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Pytest Documentation](https://docs.pytest.org/)
- [Pre-commit Documentation](https://pre-commit.com/)
- [Bandit Documentation](https://bandit.readthedocs.io/)

## Environment Variables

See `.env.example` for required environment variables. Copy it to `.env` and fill in your values:

```bash
cp .env.example .env
```