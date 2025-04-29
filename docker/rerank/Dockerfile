FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir FlagEmbedding==1.3.4

RUN mkdir -p /app/dags/.cache

COPY rerank.py /app/
COPY rerank_run.py /app/
# COPY .cache /app/dags/.cache

RUN echo '#!/bin/bash\n\
echo "Starting reranker container with arguments: $@"\n\
echo "Environment:"\n\
env | sort\n\
python rerank_run.py "$@"' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# 容器入口點
ENTRYPOINT ["/app/entrypoint.sh"]