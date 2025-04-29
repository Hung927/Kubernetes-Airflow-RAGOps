FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir ollama==0.4.7 qdrant-client==1.13.3 python-dotenv==1.1.0

RUN mkdir -p /app/dags

COPY retrieval.py /app/
COPY retrieval_run.py /app/

RUN echo '#!/bin/bash\n\
echo "Starting container with arguments: $@"\n\
echo "Environment:"\n\
env | sort\n\
echo "Directory structure:"\n\
find /airflow -type d | sort\n\
echo "Executing main script:"\n\
python retrieval_run.py "$@"' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# 容器入口點
ENTRYPOINT ["/app/entrypoint.sh"]