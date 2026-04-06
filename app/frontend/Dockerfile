FROM python:3.10-slim

WORKDIR /app

# Install system dependencies + uv (fast Python package installer)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN apt update && apt install -y \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY app/frontend/requirements.txt /app/frontend-requirements.txt
RUN uv pip install --system --no-cache -r /app/frontend-requirements.txt pyyaml

# Copy configs (needed by frontend config.py to load class names)
COPY configs/ /app/configs/

# Copy docs and evaluation outputs (needed by audit page for model card and metrics)
COPY docs/ /app/docs/
COPY outputs/ /app/outputs/

# Copy the frontend application code
COPY app/frontend/ /app/app/frontend/

WORKDIR /app/app/frontend

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"] 
