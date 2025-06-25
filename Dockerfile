FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Create non-root user, setup environment, and install package
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy application code
COPY . .

# Create virtual environment, install package, and change ownership in one layer
RUN uv venv && \
    SETUPTOOLS_SCM_PRETEND_VERSION=0.1.2 uv pip install --no-cache-dir -e '.[llamafile_remote]' && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Set environment variables to ensure uv environment is used
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# No CMD needed - @task.kubernetes will execute Python functions directly
