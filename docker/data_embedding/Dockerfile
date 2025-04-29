FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    ollama==0.4.7 \
    qdrant-client==1.13.3 \
    python-dotenv==1.1.0

RUN mkdir -p /app/data /app/dags/data

COPY data_embedding.py /app/
COPY data_embedding_run.py /app/

# 建立啟動腳本
RUN echo '#!/bin/bash\n\
echo "Starting data embedding container with arguments: $@"\n\
echo "Environment:"\n\
env | sort\n\
python data_embedding_run.py "$@"' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# 容器入口點
ENTRYPOINT ["/app/entrypoint.sh"]