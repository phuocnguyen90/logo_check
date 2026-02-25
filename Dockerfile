# Slim Python image for minimum cold start
FROM python:3.10-slim

# Prevent python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

# Install minimal system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install production requirements
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy only the necessary code
COPY logo_similarity/ logo_similarity/

# Note: In production/serverless, models and indexes should be 
# either baked into the image or volume-mounted.
# Here we create the structure.
RUN mkdir -p data/ models/ indexes/

# Expose FastAPI port
EXPOSE 8000

# Run with uvicorn (using shell form to expand $PORT)
CMD uvicorn logo_similarity.api.app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
