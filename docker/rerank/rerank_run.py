import os
import sys
import json
import logging
import argparse
from rerank import Reranker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main():
    """
    Main function to run the reranking tasks.
    
    args:
        --user-question (str): User question for reranking.
        --similarity-results (str): Results from similarity search.
        --keyword-results (str): Results from keyword search.
    """
    parser = argparse.ArgumentParser(description='Run reranking tasks')
    parser.add_argument('--user-question', type=str, default=None,
                      help='User question for reranking')
    parser.add_argument('--similarity-results', type=str, default=None,
                      help='Results from similarity search')
    parser.add_argument('--keyword-results', type=str, default=None,
                      help='Results from keyword search')
    parser.add_argument('--topk', type=int, default=5,
                      help='Number of top results to rerank')
    
    args = parser.parse_args()
    
    try:
        reranker_obj = Reranker()
        logging.info(f"Reranker object created with: {args}")
        
        class MockTi:
            def __init__(self, user_question: str, similarity_results: str = None, keyword_results: str = None):
                self.user_question = user_question
                self.similarity_results = similarity_results
                self.keyword_results = keyword_results

            def xcom_pull(self, task_ids: str, key: str=None):
                if task_ids == 'generate_query_task' and key == 'return_value':
                    return self.user_question or "What is the current number of electors currently in a Scottish Parliament constituency?"
                elif task_ids == 'similarity_retrieval_task' and key == 'return_value':
                    try:
                        if not self.similarity_results:
                            logging.info("No similarity results provided.")
                            return []
                        
                        import ast
                        results = ast.literal_eval(self.similarity_results) if self.similarity_results else []
                        logging.info(f"Parsed similarity results: {results}")
                        return results
                    except Exception as e:
                        logging.warning(f"Could not parse similarity results: {self.similarity_results}, error: {e}")
                        return []
                elif task_ids == 'keyword_retrieval_task' and key == 'return_value':
                    try:
                        if not self.keyword_results:
                            logging.info("No keyword results provided.")
                            return []
                        
                        import ast
                        results = ast.literal_eval(self.keyword_results) if self.keyword_results else []
                        logging.info(f"Parsed keyword results: {results}")
                        return results
                    except Exception as e:
                        logging.warning(f"Could not parse keyword results: {self.keyword_results}, error: {e}")
                        return []
                return None
        
        mock_ti = MockTi(
            user_question=args.user_question,
            similarity_results=args.similarity_results,
            keyword_results=args.keyword_results
        )
        
        result = reranker_obj.rerank(topk=args.topk, ti=mock_ti)
            
        os.makedirs("/airflow/xcom", exist_ok=True)
        
        with open('/airflow/xcom/return.json', 'w') as f:
            json.dump(result, f)
            logging.info(f"Result saved to /airflow/xcom/return.json")
        
        return result
        
    except Exception as e:
        logging.error(f"Error during reranking: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()