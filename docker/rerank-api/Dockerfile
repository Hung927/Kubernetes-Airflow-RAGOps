FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    FlagEmbedding==1.3.4 \
    fastapi \
    uvicorn \
    pydantic \
    requests

RUN mkdir -p /app/dags/.cache

COPY rerank.py /app/
COPY rerank_api.py /app/
COPY .cache /app/dags/.cache

HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -f http://localhost:8001/ || exit 1

EXPOSE 8001

CMD ["uvicorn", "rerank_api:app", "--host", "0.0.0.0", "--port", "8001"]