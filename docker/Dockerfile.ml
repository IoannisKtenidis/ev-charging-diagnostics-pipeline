FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch (CPU-only version for optimization and lightweight size)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install other Python dependencies
RUN pip install mlflow bentoml ocpp websockets

# Expose target service ports
EXPOSE 3000 5000 9000

# Default command placeholder
CMD ["python"]
