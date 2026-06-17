FROM python:3.11-slim

WORKDIR /app

# Read-only server: dashboard + chat over pre-built data. PDF ingest/OCR runs off-server,
# so tesseract/poppler are not installed here.

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default mount points (overridden at runtime by docker-compose bind mounts).
RUN mkdir -p knowledge extracted_data .chroma_db

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
