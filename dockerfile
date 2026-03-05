FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install minimal system deps (remove build-essential if not using xgboost)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements-prod.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements-prod.txt

# Copy only required folders
COPY app ./app
COPY src/ src/
COPY models ./models
COPY logger ./logger

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]