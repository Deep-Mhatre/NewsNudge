FROM python:3.10-slim

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy backend source
COPY backend/ /app/

# Install python deps
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

EXPOSE 8000

CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]
