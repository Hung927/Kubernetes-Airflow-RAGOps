import os
import sys
import json
import logging
import argparse
from ragas_evaluator import Ragas

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description='Run RAGAS evaluation')
    parser.add_argument('--user-question', type=str, default=None,
                      help='User question for evaluation')
    parser.add_argument('--llm-answer', type=str, default=None,
                      help='LLM generated answer')
    parser.add_argument('--similarity-results', type=str, default=None,
                      help='Results from similarity search')
    parser.add_argument('--keyword-results', type=str, default=None,
                      help='Results from keyword search')
    parser.add_argument('--rerank-results', type=str, default=None,
                      help='Results from reranking')
    parser.add_argument('--use-similarity', type=str, default="False",
                      help='Flag to use similarity retrieval')
    parser.add_argument('--use-keyword', type=str, default="False",
                      help='Flag to use keyword retrieval')
    parser.add_argument('--use-rerank', type=str, default="False",
                      help='Flag to use rerank')
    
    args = parser.parse_args()
    
    try:
        ragas_obj = Ragas(qa_path="/app/data/qa_pairs.json")
        logging.info(f"Ragas object created with arguments: {args}")
        
        class MockTi:
            def __init__(self, user_question=None, llm_answer=None, 
                        similarity_results=None, keyword_results=None, rerank_results=None):
                self.user_question = user_question
                self.llm_answer = llm_answer
                self.similarity_results = similarity_results
                self.keyword_results = keyword_results
                self.rerank_results = rerank_results
            
            def xcom_pull(self, task_ids, key=None):
                if task_ids == 'random_question_task' and key == 'return_value':
                    return self.user_question
                elif task_ids == 'llm_task' and key == 'return_value':
                    return self.llm_answer
                elif task_ids == 'similarity_retrieval_task' and key == 'return_value':
                    try:
                        import ast
                        similarity_results = ast.literal_eval(self.similarity_results) if self.similarity_results else []
                        logging.info(f"Parsed search results: {similarity_results}")
                        return similarity_results if isinstance(similarity_results, list) else []
                    except Exception as e:
                        logging.warning(f"Could not parse similarity results: {e}")
                        return []
                elif task_ids == 'keyword_retrieval_task' and key == 'return_value':
                    try:
                        import ast
                        keyword_results = ast.literal_eval(self.keyword_results) if self.keyword_results else []
                        logging.info(f"Parsed keyword results: {keyword_results}")
                        return keyword_results if isinstance(keyword_results, list) else []
                    except Exception as e:
                        logging.warning(f"Could not parse keyword results: {e}")
                        return []
                elif task_ids == 'reranking_task' and key == 'return_value':
                    try:
                        import ast
                        rerank_results = ast.literal_eval(self.rerank_results) if self.rerank_results else []
                        logging.info(f"Parsed rerank results: {rerank_results}")
                        return rerank_results if isinstance(rerank_results, list) else []
                    except Exception as e:
                        logging.warning(f"Could not parse rerank results: {e}")
                        return []
                return None
        
        mock_ti = MockTi(
            user_question=args.user_question,
            llm_answer=args.llm_answer,
            similarity_results=args.similarity_results,
            keyword_results=args.keyword_results,
            rerank_results=args.rerank_results
        )
        
        use_similarity = args.use_similarity.lower() == "true"
        use_keyword = args.use_keyword.lower() == "true"
        use_rerank = args.use_rerank.lower() == "true"
        
        result = ragas_obj.ragas(
            ti=mock_ti,
            USE_SIMILARITY=use_similarity,
            USE_KEYWORD=use_keyword,
            USE_RERANK=use_rerank
        )
        
        # 確保 xcom 目錄存在
        os.makedirs("/airflow/xcom", exist_ok=True)
        
        # 將結果寫入 xcom
        with open('/airflow/xcom/return.json', 'w') as f:
            json.dump(result, f)
            logging.info(f"Result saved to /airflow/xcom/return.json")
        
        return result
        
    except Exception as e:
        logging.error(f"Error during RAGAS evaluation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()