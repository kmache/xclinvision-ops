FROM python:3.10-slim

WORKDIR /app

# Install system dependencies + uv (fast Python package installer)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN apt update && apt install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxcb1 \
    libxext6 \
    libxrender-dev \
    curl \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    libgdk-pixbuf-xlib-2.0-dev \
    shared-mime-info \
    libcairo2 \
    libffi-dev \
    && (gdk-pixbuf-query-loaders > /usr/lib/x86_64-linux-gnu/gdk-pixbuf-2.0/2.10.0/loaders.cache 2>/dev/null || true) \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (backend + core library)
COPY app/backend/requirements.txt /app/backend-requirements.txt
RUN uv pip install --system --no-cache -r /app/backend-requirements.txt

# Install the xclinvision core package
COPY pyproject.toml /app/pyproject.toml
COPY src/ /app/src/
RUN uv pip install --system --no-cache -e .

# Copy configs used by the core library at runtime
COPY configs/ /app/configs/

# Copy the backend application code
COPY app/backend/ /app/app/backend/

# Copy docs (for model card endpoint)
COPY docs/ /app/docs/

WORKDIR /app/app/backend

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]