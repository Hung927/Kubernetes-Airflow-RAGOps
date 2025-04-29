import os
import sys
import json
import logging
import argparse
from data_embedding import Data_Embedding

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description='Data Embedding in Kubernetes')
    parser.add_argument('--data-context-path', default='/app/dags/data/data_context.json', 
                      help='Path to data context file')
    parser.add_argument('--embed-model', default='imac/zpoint_large_embedding_zh',
                      help='Embedding model to use')
    
    args = parser.parse_args()
    
    try:
        logging.info("Starting data embedding task...")
        logging.info(f"Data context path: {args.data_context_path}")
        logging.info(f"Embedding model: {args.embed_model}")
        logging.info(f"Ollama URL: {os.getenv('OLLAMA_HOST')}")
        logging.info(f"Qdrant URL: {os.getenv('QDRANT_URL')}")
        
        data_embedding_obj = Data_Embedding(
            embed_model=args.embed_model,
            data_context_path=args.data_context_path
        )
        
        result = data_embedding_obj.documents_embedding()
        
        return result
        
    except Exception as e:
        logging.error(f"Error in data embedding: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()