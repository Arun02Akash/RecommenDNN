# Base image
FROM python:3.6-slim   # slim Python base instead of full Ubuntu 16.04

# Set environment variables
ENV KERAS_BACKEND=theano \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (exact versions for reproducibility)
RUN pip install --no-cache-dir \
        numpy==1.11.0 \
        h5py \
        Theano==0.8.0 \
        keras==1.0.7

# Create working directory
WORKDIR /home

# Default command
CMD ["/bin/bash"]
