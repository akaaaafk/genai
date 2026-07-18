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

# The lockfile pins torch 2.12.1 / torchvision 0.27.1, but the Linux resolution pulls the
# heavy CUDA build. Install the matching CPU builds explicitly, with versions pinned so the
# image is reproducible and the build never grabs a different torch release.
RUN uv pip install --no-cache \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.12.1 torchvision==0.27.1

# Copy the assignment code (includes models/final_ebm.pth and models/best_diffusion.pth)
COPY assignment4 /code/assignment4

# Run from the assignment directory so the local `imgmodel` package is importable
WORKDIR /code/assignment4
ENV PYTHONPATH=/code/assignment4

# Use --no-sync so the explicitly installed torch wheels are not removed by a re-sync
CMD ["uv", "run", "--no-sync", "fastapi", "run", "app/main.py", "--port", "80"]
