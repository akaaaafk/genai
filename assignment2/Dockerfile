# Use an official Python runtime as a parent image
FROM python:3.12-slim-bookworm

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Download and run the uv installer, then remove it
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Set the working directory
WORKDIR /code

# Copy dependency manifests and install locked dependencies (fastapi, uvicorn, etc.)
COPY pyproject.toml uv.lock /code/
RUN uv sync --frozen

# torch / torchvision / pillow are required by assignment2 but are not in the lockfile,
# so install the CPU builds explicitly into the project environment
RUN uv pip install --no-cache \
    --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision \
    && uv pip install --no-cache pillow numpy

# Copy the assignment code (includes models/best_model.pth needed at startup)
COPY assignment2 /code/assignment2

# Run from the assignment directory so the local `model_combined` package is importable
WORKDIR /code/assignment2
ENV PYTHONPATH=/code/assignment2

# Use --no-sync so the explicitly installed torch wheels are not removed by a re-sync
CMD ["uv", "run", "--no-sync", "fastapi", "run", "app/main.py", "--port", "80"]
