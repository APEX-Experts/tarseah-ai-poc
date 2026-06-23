# Use a slim Python 3.11 image for a small footprint and security
FROM python:3.11-slim

# Set environment variables
# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

# Set working directory
WORKDIR /app

# Install system dependencies (curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to take advantage of Docker layer caching
COPY src/requirement.txt /app/src/requirement.txt

# Install dependencies
RUN pip install --no-cache-dir -r /app/src/requirement.txt

# Copy the rest of the application files
COPY src/ /app/src/

# Create storage and assets directory and set ownership to a non-root user
RUN mkdir -p /app/src/storage /app/src/Assets && \
    useradd -u 1000 -m appuser && \
    chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser

# Expose port
EXPOSE 9676

# Run FastAPI app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9676"]
