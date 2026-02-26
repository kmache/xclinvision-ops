.PHONY: help install dev-install data download-data train evaluate test lint format docker-build docker-up docker-down clean

help:
	@echo "XClinVision - Explainable Medical Imaging AI Platform"
	@echo ""
	@echo "Available targets:"
	@echo "  install         Install production dependencies"
	@echo "  dev-install     Install development dependencies"
	@echo "  download-data   Download and prepare datasets"
	@echo "  preprocess      Run data preprocessing pipeline"
	@echo "  train           Train model with default config"
	@echo "  evaluate        Run model evaluation"
	@echo "  test            Run test suite"
	@echo "  lint            Run linters (black, isort, flake8, mypy)"
	@echo "  format          Format code with black and isort"
	@echo "  docker-build    Build Docker images"
	@echo "  docker-up       Start services with docker-compose"
	@echo "  docker-down     Stop services"
	@echo "  mlflow-ui       Start MLflow tracking UI"
	@echo "  dashboard       Launch Streamlit clinician dashboard"
	@echo "  api             Start FastAPI backend server"
	@echo "  clean           Clean build artifacts"

install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"
	pre-commit install

download-data:
	@echo "Downloading datasets..."
	python scripts/download_data.py

preprocess:
	@echo "Running preprocessing pipeline..."
	python scripts/preprocess_data.py

train:
	@echo "Starting model training..."
	python scripts/train.py

evaluate:
	@echo "Running evaluation..."
	python scripts/evaluate.py

tune:
	@echo "Starting hyperparameter tuning..."
	python scripts/hyperparam_search.py

test:
	pytest tests/ -v --cov=xclinvision --cov-report=term-missing

lint:
	black --check src/ tests/ app/
	isort --check-only src/ tests/ app/
	flake8 src/ tests/ app/
	mypy src/

format:
	black src/ tests/ app/
	isort src/ tests/ app/

docker-build:
	docker-compose -f deployment/docker-compose.yml build

docker-up:
	docker-compose -f deployment/docker-compose.yml up -d

docker-down:
	docker-compose -f deployment/docker-compose.yml down

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000

dashboard:
	streamlit run app/frontend/main.py

api:
	uvicorn app.backend.main:app --host 0.0.0.0 --port 8000 --reload

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .coverage htmlcov/
