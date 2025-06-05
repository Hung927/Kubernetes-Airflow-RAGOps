# import os
# import sys
# import json
# import logging
# import argparse
# import requests
# from requests.exceptions import RequestException

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# def main():
#     parser = argparse.ArgumentParser(description="Call retrieval API")
#     parser.add_argument('--types', type=str, required=True, choices=['expert', 'similarity', 'keyword'],
#                         help='Retrieval type (expert, similarity, keyword)')
#     parser.add_argument("--user-question", type=str, required=True, help="User question for retrieval")
#     parser.add_argument("--document-types", type=str, default="squad", help="Document types to retrieve")
#     parser.add_argument("--api-host", type=str, default="retrieval-api.default.svc.cluster.local", help="API host")
#     parser.add_argument("--api-port", type=int, default=8000, help="API port")
#     parser.add_argument("--topk", type=int, default=10, help="Number of top results to retrieve")
#     parser.add_argument("--embed-model", type=str, default="imac/zpoint_large_embedding_zh", 
#                         help="Embedding model")
#     parser.add_argument('--keyword-list', type=str, default=None,
#                         help='Keyword list for keyword retrieval', required=False)
    
#     args = parser.parse_args()
#     logging.info(f"Arguments: {args}")
    
#     api_url = f"http://{args.api_host}:{args.api_port}/retrieve"
    
#     payload = {
#         "types": args.types,
#         "document_types": args.document_types,
#         "topk": args.topk,
#         "embed_model": args.embed_model,
#         "user_question": args.user_question,
#         "keyword_list": args.keyword_list
#     }
    
#     print(f"Sending request to {api_url} with payload: {payload}")
    
#     try:
#         response = requests.post(api_url, json=payload, timeout=30)
#         response.raise_for_status()
#         result = response.json().get("result")
        
#         os.makedirs("/airflow/xcom", exist_ok=True)
#         with open('/airflow/xcom/return.json', 'w') as f:
#             json.dump(result, f)
#             print("Results saved successfully")
        
#     except RequestException as e:
#         print(f"API request error: {str(e)}")
#         if hasattr(e.response, 'text'):
#             print(f"Response text: {e.response.text}")
#         sys.exit(1)
#     except Exception as e:
#         print(f"Unexpected error: {str(e)}")
#         sys.exit(1)

# if __name__ == "__main__":
#     main()


# Dockerfile
# FROM python:3.10-slim

# WORKDIR /app

# RUN pip install --no-cache-dir requests

# COPY api_client.py /app/

# ENV QDRANT_URL=

# ENTRYPOINT ["python", "api_client.py"]