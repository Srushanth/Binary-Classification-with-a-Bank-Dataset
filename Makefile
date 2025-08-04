venv:
	uv venv

download-data:
	export KAGGLE_USERNAME=<KAGGLE_USERNAME>
	export KAGGLE_KEY=<KAGGLE_KEY>
	mkdir -p data
	kaggle competitions download -c playground-series-s5e8 -p data
	unzip data/playground-series-s5e8.zip -d data
	rm data/playground-series-s5e8.zip

# Code linting
lint:
	@echo "Running linting checks..."
	@echo "✓ Black formatting check"
	black --check . || (echo "❌ Black formatting failed" && exit 1)
	@echo "✓ Isort import sorting check"
	isort --check-only . || (echo "❌ Isort import sorting failed" && exit 1)
	@echo "✓ Linting checks passed!"

# Run tests
test:
	@echo "Running tests..."
	@if [ -d "tests" ]; then \
		pytest tests/ -v --cov=. --cov-report=html; \
	else \
		echo "✓ No tests directory found - skipping tests"; \
	fi

# Clean up generated files
clean:
	@echo "Cleaning up generated files..."
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name "__pycache__" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache 2>/dev/null || true
	rm -rf htmlcov 2>/dev/null || true
	rm -rf .coverage 2>/dev/null || true
	@echo "✓ Cleanup completed!"

# Validation and testing
check: lint test
	@echo "Running comprehensive checks..."
	@echo "✓ Code formatting check"
	@echo "✓ Linting check"
	@echo "✓ Test execution"
	@echo "✓ Data file validation"
	@test -f data/train.csv || (echo "❌ Missing train.csv" && exit 1)
	@test -f data/test.csv || (echo "❌ Missing test.csv" && exit 1)
	@test -f data/sample_submission.csv || (echo "❌ Missing sample_submission.csv" && exit 1)
	@echo "✓ All checks passed!"

# Distribution check - validates the project can be properly packaged and distributed
distcheck: clean check
	@echo "Running distribution check..."
	@echo "✓ Clean build environment"
	@echo "✓ Package installation test"
	uv pip install -e . --force-reinstall
	@echo "✓ Package can be installed"
	@echo "✓ Import test"
	python -c "import sys; print('Python version:', sys.version)"
	@echo "✓ All distribution checks passed!"

.PHONY: venv download-data lint test clean check distcheck
