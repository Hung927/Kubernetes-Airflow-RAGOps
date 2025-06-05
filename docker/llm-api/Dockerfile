FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    ollama==0.4.7 \
    langchain-ollama==0.2.3 \
    python-dotenv==1.1.0 \
    fastapi \
    uvicorn \
    pydantic \
    requests

RUN mkdir -p /app/dags

COPY llm.py /app/
COPY llm_api.py /app/
COPY prompt_config.py /app/

HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -f http://localhost:8000/ || exit 1

EXPOSE 8002

CMD ["uvicorn", "llm_api:app", "--host", "0.0.0.0", "--port", "8002"]