FROM python:3.12-slim AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy package metadata and source (hatchling needs README + src to build)
COPY pyproject.toml README.md ./
COPY src/ src/

# Install Python dependencies + the package itself
RUN pip install --no-cache-dir .

# Copy runtime config
COPY config/ config/
COPY prompts/ prompts/

# Create data directory
RUN mkdir -p /app/data

# Run the application
CMD ["python", "-m", "mai_gram.main"]
