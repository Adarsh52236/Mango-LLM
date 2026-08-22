FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (if any are needed for torch or tokenizers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create cache directory for Hugging Face with proper permissions
# (Hugging Face Spaces runs as a non-root user with ID 1000)
RUN mkdir -p /app/.cache && chmod 777 /app/.cache
ENV HF_HOME=/app/.cache

# Expose port
EXPOSE 7860

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
