FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    datasets==3.5.0 \
    ragas==0.2.14 \
    langchain-openai==0.3.14 \
    fastapi \
    uvicorn \
    pydantic \
    requests

RUN mkdir -p /app/data

COPY ragas_evaluator.py /app/
COPY ragas_api.py /app/
COPY qa_pairs.json /app/data/

HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -f http://localhost:8003/ || exit 1

EXPOSE 8003

CMD ["uvicorn", "ragas_api:app", "--host", "0.0.0.0", "--port", "8003"]