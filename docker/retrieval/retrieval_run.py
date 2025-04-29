import os
import sys
import json
import logging
import argparse
from retrieval import Retrieval

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main():
    """
    Main function to run the retrieval tasks.
    
    args:
        --types (str): Type of retrieval task to run. Options are 'expert', 'similarity', 'keyword'.
        --document-types (str): Document types to retrieve.
        --topk (int): Number of top results to retrieve.
        --embed-model (str): Embedding model to use.
        --user-question (str): User question for retrieval.
        --keyword-list (str): Keyword list for keyword retrieval.
    """
    parser = argparse.ArgumentParser(description='Run retrieval tasks')
    parser.add_argument('--types', type=str, required=True, choices=['expert', 'similarity', 'keyword'],
                      help='Retrieval type (expert, similarity, keyword)')
    parser.add_argument('--document-types', type=str, default='squad', 
                      help='Document types to retrieve')
    parser.add_argument('--topk', type=int, default=10,
                      help='Number of top results to retrieve')
    parser.add_argument('--embed-model', type=str, default='imac/zpoint_large_embedding_zh',
                      help='Embedding model to use')
    parser.add_argument('--user-question', type=str, default=None,
                      help='User question for retrieval')
    parser.add_argument('--keyword-list', type=str, default=None,
                      help='Keyword list for keyword retrieval', required=False)
    
    args = parser.parse_args()
    
    try:
        retrieval_obj = Retrieval(
            embed_model=args.embed_model
        )
        logging.info(f"Retrieval object created with: {args}")
        
        class MockTi:
            def __init__(self, user_question: str, keyword_list: list = None):
                self.user_question = user_question
                self.keyword_list = keyword_list

            def xcom_pull(self, task_ids: str, key: str=None):
                if task_ids == 'random_question_task' and key == 'return_value':
                    return self.user_question or "What is the current number of electors currently in a Scottish Parliament constituency?"
                elif task_ids == 'keyword_extraction_task' and key == 'return_value':
                    try:
                        # 嘗試將字符串解析為 Python 列表
                        import ast
                        keywords = ast.literal_eval(self.keyword_list)
                        logging.info(f"Parsed keyword list: {keywords}")
                        return keywords
                    except:
                        # 如果解析失敗，返回原始字符串
                        logging.warning(f"Could not parse keyword list: {self.keyword_list}")
                        return self.keyword_list
                return None
        
        mock_ti = MockTi(
            user_question=args.user_question,
            keyword_list=getattr(args, "keyword_list", None)
        )
        
        result = retrieval_obj.retrieval(
            types=args.types,
            document_types=args.document_types,
            topk=args.topk,
            ti=mock_ti,
        )
        
        os.makedirs("/airflow/xcom", exist_ok=True)
        
        with open('/airflow/xcom/return.json', 'w') as f:
            json.dump(result, f)
            logging.info(f"Result saved to /airflow/xcom/return.json")
        return result
        
    except Exception as e:
        logging.error(f"Error during retrieval: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()