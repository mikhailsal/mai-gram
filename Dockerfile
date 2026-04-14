FROM python:3.12-slim AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir .

# Copy source code and config
COPY src/ src/
COPY config/ config/
COPY prompts/ prompts/

# Create data directory
RUN mkdir -p /app/data

# Run the application
CMD ["python", "-m", "mai_gram.main"]
