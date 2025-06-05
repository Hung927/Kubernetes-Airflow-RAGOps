FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    ollama==0.4.7 \
    qdrant-client==1.13.3 \
    python-dotenv==1.1.0 \
    fastapi \
    uvicorn \
    pydantic \
    requests

RUN mkdir -p /app/dags

COPY retrieval.py /app/
COPY retrieval_api.py /app/

HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -f http://localhost:8000/ || exit 1

EXPOSE 8000

CMD ["uvicorn", "retrieval_api:app", "--host", "0.0.0.0", "--port", "8000"]