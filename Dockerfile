# # Use NVIDIA CUDA base image
# FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# # Set working directory
# WORKDIR /app

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     python3-pip \
#     python3-dev \
#     && rm -rf /var/lib/apt/lists/*

# # Install Python packages
# COPY requirements.txt .
# RUN pip3 install --no-cache-dir -r requirements.txt

# # Copy application code
# COPY . .

# # Create model cache directory
# RUN mkdir -p /app/model-cache

# # Set environment variables
# ENV MODEL_CACHE_DIR=/app/model-cache
# ENV PYTHONUNBUFFERED=1
# ENV RAY_ADDRESS=auto

# # Expose ports
# EXPOSE 8000 6379 8265

# # Start the service
# CMD ["python3", "main.py"]

# Use Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create model cache directory
RUN mkdir -p /app/model-cache

# Set environment variables
ENV MODEL_CACHE_DIR=/app/model-cache
ENV PYTHONUNBUFFERED=1

# Expose ports
EXPOSE 8000 6379 8265

# Start the service
CMD ["python3", "-c", "import ray; ray.init(dashboard_host='0.0.0.0', include_dashboard=True); from main import main; main()"]