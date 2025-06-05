import socket
import logging
from typing import Optional
import requests

class APIConfig:
    def __init__(self, llm_model: str, embed_model: str, document_types: str):
        self.llm_model = llm_model
        self.embed_model = embed_model
        self.document_types = document_types
        
    def call_retrieval_api(
        self, 
        api_host: str, 
        api_port: int = 8000, 
        types: str = "similarity", 
        topk: int = 10,
    ):
        def _call_api(**context):
            ti = context['ti']
            user_question = ti.xcom_pull(task_ids='generate_query_task', key='return_value')
            keyword_list = str(ti.xcom_pull(task_ids='keyword_extraction_task', key='return_value')) if types=="keyword" else "[]"
            api_url = f"http://{api_host}:{api_port}/retrieve"
            payload = {
                "types": types,
                "document_types": self.document_types,
                "topk": topk,
                "embed_model": self.embed_model,
                "user_question": user_question,
                "keyword_list": keyword_list
            }
            
            try:
                socket.gethostbyname(api_host)
                logging.info(f"Sending request to {api_url} with payload: {payload}")
                response = requests.post(api_url, json=payload, timeout=300)
                response.raise_for_status()
                result = response.json()
                return result["result"]
            except Exception as e:
                logging.error(f"Error calling retrieval API: {e}")
                raise
            
        return _call_api
    
    def call_rerank_api(
        self, 
        api_host: str, 
        api_port: int = 8001, 
        topk: int = 5,
    ):
        def _call_api(**context):
            ti = context['ti']
            user_question = ti.xcom_pull(task_ids='generate_query_task', key='return_value')
            similarity_results = str(ti.xcom_pull(task_ids='similarity_retrieval_task', key='return_value')) if ti.xcom_pull(task_ids='similarity_retrieval_task', key='return_value') else "[]"
            keyword_results = str(ti.xcom_pull(task_ids='keyword_retrieval_task', key='return_value')) if ti.xcom_pull(task_ids='keyword_retrieval_task', key='return_value') else "[]"
            api_url = f"http://{api_host}:{api_port}/rerank"
            payload = {
                "topk": topk,
                "user_question": user_question,
                "similarity_results": similarity_results,
                "keyword_results": keyword_results,
            }
            
            try:
                socket.gethostbyname(api_host)
                logging.info(f"Sending request to {api_url} with payload: {payload}")
                response = requests.post(api_url, json=payload, timeout=30)
                response.raise_for_status()
                result = response.json()
                return result["result"]
            except Exception as e:
                logging.error(f"Error calling rerank API: {e}")
                raise
            
        return _call_api
    
    def call_llm_api(
        self, 
        api_host: str, 
        api_port: int = 8002, 
        types: str = "rag",
        temperature: float = 0.0,
        keep_alive: str = "0s",
        num_ctx: int = 8192,
        search_results_types: Optional[str] = None,
    ):
        def _call_api(**context):
            ti = context['ti']
            user_question = ti.xcom_pull(task_ids='generate_query_task', key='return_value')
            search_results = str(ti.xcom_pull(task_ids=search_results_types, key='return_value')) if search_results_types else None
            api_url = f"http://{api_host}:{api_port}/llm"
            payload = {
                "user_question": user_question,
                "types": types,
                "model": self.llm_model,
                "temperature": temperature,
                "keep_alive": keep_alive,
                "num_ctx": num_ctx,
                "search_results_types": search_results_types,
                "search_results": search_results,
            }
            logging.info(f"Payload for LLM API: {payload}")
            try:
                socket.gethostbyname(api_host)
                logging.info(f"Sending request to {api_url} with payload: {payload}")
                response = requests.post(api_url, json=payload, timeout=300)
                response.raise_for_status()
                result = response.json()
                return result["result"]
            except Exception as e:
                logging.error(f"Error calling llm API: {e}")
                raise
            
        return _call_api
    
    def call_ragas_api(
        self, 
        api_host: str, 
        api_port: int = 8003, 
        use_similarity: bool = False,
        use_keyword: bool = False,
        use_rerank: bool = False,
    ):
        def _call_api(**context):
            ti = context['ti']
            user_question = ti.xcom_pull(task_ids='generate_query_task', key='return_value')
            llm_answer = ti.xcom_pull(task_ids='llm_task', key='return_value')
            similarity_results = str(ti.xcom_pull(task_ids='similarity_retrieval_task', key='return_value')) if use_similarity else None
            keyword_results = str(ti.xcom_pull(task_ids='keyword_retrieval_task', key='return_value')) if use_keyword else None
            rerank_results = str(ti.xcom_pull(task_ids='reranking_task', key='return_value')) if use_rerank else None
            api_url = f"http://{api_host}:{api_port}/ragas"
            payload = {
                "user_question": user_question,
                "llm_answer": llm_answer,
                "similarity_results": similarity_results,
                "keyword_results": keyword_results,
                "rerank_results": rerank_results,
                "use_similarity": use_similarity,
                "use_keyword": use_keyword,
                "use_rerank": use_rerank,
            }
            logging.info(f"Payload for RAGAS API: {payload}")
            try:
                socket.gethostbyname(api_host)
                logging.info(f"Sending request to {api_url} with payload: {payload}")
                response = requests.post(api_url, json=payload, timeout=300)
                response.raise_for_status()
                result = response.json()
                return result["result"]
            except Exception as e:
                logging.error(f"Error calling ragas API: {e}")
                raise
            
        return _call_api