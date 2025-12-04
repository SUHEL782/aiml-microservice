# service/Dockerfile
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app /app
# Copy model if training produced it in repo; otherwise mount at runtime
COPY ../training/models /app/models

ENV MODEL_PATH=/app/models/text_classifier.joblib
ENV INFERENCE_API_KEY=changeme

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
