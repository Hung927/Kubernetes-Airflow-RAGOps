import os
import sys
import json
import logging
import argparse
from llm import LLM

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main():
    """
    Main function to run the LLM tasks.
    
    args:
        --types (str): Type of LLM task to run. Options are 'keyword', 'general', 'rag', 'validation', 'summary'.
        --model (str): Name of the LLM model to use.
        --temperature (float): Temperature setting for the model.
        --keep-alive (str): Keep alive setting for the model.
        --num-ctx (int): Number of context tokens for the model.
        --user-question (str): User question for LLM processing.
        --search-results (str): Search results for LLM processing.
    """
    
    parser = argparse.ArgumentParser(description='Run LLM tasks')
    parser.add_argument('--types', type=str, required=True, 
                      choices=['keyword', 'general', 'rag', 'validation', 'summary'],
                      help='LLM task type (keyword, general, rag, validation, summary)')
    parser.add_argument('--model', type=str, default='gemma2:9b',
                      help='LLM model name')
    parser.add_argument('--temperature', type=float, default=0.0,
                      help='Temperature for LLM processing')
    parser.add_argument('--keep-alive', type=str, default="0s",
                      help='Keep alive setting for LLM processing')
    parser.add_argument('--num-ctx', type=int, default=8192,
                      help='Number of context tokens for LLM processing')
    parser.add_argument('--user-question', type=str, default=None,
                      help='User question for LLM processing')
    parser.add_argument('--search-results-types', type=str, default=None,
                        help='Types of search results for LLM processing')
    parser.add_argument('--search-results', type=str, default=None,
                      help='Search results for LLM processing')
    
    args = parser.parse_args()
    
    try:
        llm_obj = LLM(
            model=args.model,
            temperature=args.temperature,
            keep_alive=args.keep_alive,
            num_ctx=args.num_ctx,
        )
        logging.info(f"LLM object created with: {args}")
        
        class MockTi:
            def __init__(self, user_question: str, search_results_types: str=None, search_results: str = None):
                self.user_question = user_question
                self.search_results_types = search_results_types
                self.search_results = search_results
            
            def xcom_pull(self, task_ids, key=None):
                if task_ids == 'generate_query_task' and key == 'return_value':
                    return args.user_question or "What is the current number of electors currently in a Scottish Parliament constituency?"
                elif task_ids == self.search_results_types and key == 'return_value':
                    try:
                        import ast
                        search_results = ast.literal_eval(self.search_results)
                        logging.info(f"Parsed search results: {search_results}")
                        return search_results
                    except:
                        logging.warning(f"Could not parse search results: {self.search_results}")
                        return self.search_results
                return None
        
        mock_ti = MockTi(
            user_question=args.user_question,
            search_results_types=getattr(args, "search_results_types", None),
            search_results=getattr(args, "search_results", None)
        )
        
        result = llm_obj.llm(
            types=args.types,
            ti=mock_ti,
        )
        
        os.makedirs("/airflow/xcom", exist_ok=True)
        
        with open('/airflow/xcom/return.json', 'w') as f:
            json.dump(result, f)
            logging.info(f"Result saved to /airflow/xcom/return.json")
        
        return result
        
    except Exception as e:
        logging.error(f"Error during LLM processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()