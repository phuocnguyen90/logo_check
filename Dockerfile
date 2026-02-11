FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Set up work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    tesseract-ocr \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy project files
COPY logo_similarity/ logo_similarity/
COPY scripts/ scripts/
COPY README.md .

# Create needed directories
RUN mkdir -p data/ models/ indexes/ logs/

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Run API
CMD ["python3", "logo_similarity/api/app.py"]
