FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir datasets==3.5.0 ragas==0.2.14 langchain-openai==0.3.14

RUN mkdir -p /app/data

COPY ragas_evaluator.py /app/
COPY ragas_run.py /app/
COPY qa_pairs.json /app/data/

RUN echo '#!/bin/bash\n\
echo "Starting RAGAS container with arguments: $@"\n\
echo "Environment:"\n\
env | sort\n\
python ragas_run.py "$@"' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]