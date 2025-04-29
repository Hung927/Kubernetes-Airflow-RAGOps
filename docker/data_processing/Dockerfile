FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir marker-pdf==1.6.2 langchain_text_splitters==0.3.8

RUN mkdir -p /app/data /app/data/pdf

COPY data_processing.py /app/
COPY data_processing_run.py /app/

RUN echo '#!/bin/bash\n\
echo "Starting data processing container with arguments: $@"\n\
echo "Environment:"\n\
env | sort\n\
python data_processing_run.py "$@"' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]