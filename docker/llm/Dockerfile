FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir ollama==0.4.7 langchain-ollama==0.2.3 python-dotenv==1.1.0

RUN mkdir -p /app/dags

COPY llm.py /app/
COPY prompt_config.py /app/
COPY llm_run.py /app/

RUN echo '#!/bin/bash\n\
echo "Starting container with arguments: $@"\n\
echo "Environment:"\n\
env | sort\n\
python llm_run.py "$@"' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]