FROM python:3.11-slim

WORKDIR /app

# System deps for tree-sitter and psycopg
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# API (default command)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
