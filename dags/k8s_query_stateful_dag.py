from dotenv import load_dotenv
load_dotenv(dotenv_path="dags/.env")

import os
import sys
import json
import random
import logging
sys.path.append(os.getcwd())
from airflow import DAG
from pendulum import duration
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes import config, client
from kubernetes.client.models import V1EnvVar, V1PodDNSConfig, V1PodDNSConfigOption

from utils.api_calling import APIConfig
from utils.expert_branch import ExpertBranch

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')
try:
    with open(CONFIG_PATH, "r") as f:
        config_data = json.load(f)
    rag_config = config_data.get("rag_pipeline_config")
    logging.info(f"Loaded RAG pipeline config: {rag_config}")
except (FileNotFoundError, json.JSONDecodeError) as e:
    logging.warning(f"Could not load or parse {CONFIG_PATH}. Using default RAG config. Error: {e}")
    rag_config = {
        "use_expert_retrieval": False,
        "use_similarity_retrieval": False,
        "use_keyword_retrieval": False,
        "use_rerank": False,
        "use_ragas": False
    }
USE_EXPERT = rag_config.get("use_expert_retrieval", False)
USE_SIMILARITY = rag_config.get("use_similarity_retrieval", False)
USE_KEYWORD = rag_config.get("use_keyword_retrieval", False)
USE_RAGAS = rag_config.get("use_ragas", False)
if USE_SIMILARITY and USE_KEYWORD:
    USE_RERANK = True
elif (USE_SIMILARITY or USE_KEYWORD):
    USE_RERANK = rag_config.get("use_rerank", False)
else:
    USE_RERANK = False
logging.info(f"RAG pipeline config: USE_EXPERT={USE_EXPERT}, USE_SIMILARITY={USE_SIMILARITY}, USE_KEYWORD={USE_KEYWORD}, USE_RERANK={USE_RERANK}, USE_RAGAS={USE_RAGAS}")    

try:
    config.load_kube_config(config_file="~/.kube/config")
    v1 = client.CoreV1Api()
    pods = v1.list_namespaced_pod(namespace="default")
    logging.info(f"Pods in default namespace: {[pod.metadata.name for pod in pods.items]}")
except Exception as e:
    logging.error(f"Error loading Kubernetes config: {e}")

    
default_args = {
  "owner": "HUNG",
  "start_date": datetime(2025, 4, 24),
  "retries": 3,
  "retry_delay": duration(minutes=5),
}

def get_user_question(user_question:str = None) -> str:
    """Get a random user question from the qa_pairs.json file."""
    if not user_question:
    #     return "The immune systems of bacteria have enzymes that protect against infection by what kind of cells?"
        return "Who proposed that innate intertial is the natural state of objects?"
        # return "Besides the North Sea and the Irish Channel, what else was lowered in the last cold phase?"
        # return "What organization is devoted to Jihad against Israel?"
        # return "What is the total make up of fish species living in the Amazon?"
        # return "What are the stages in a compound engine called?"
        # return "When was the cabinet-level Energy Department created?"
        # return "US is concerned about confrontation of the Middle East with which other country?"
        # return "How did user of Tymnet connect"
        # return "What did Stiglitz present in 2009 regarding global inequality?"

        # return "Where did the Normans and Byzantines sign the peace treaty?"
    # qa_path = os.path.join(os.path.dirname(__file__), "data/qa_pairs.json")
    # try:
    #     with open(qa_path, "r") as f:
    #         qa_data = json.load(f)
    #         user_question = random.choice(list(qa_data.keys()))
    # except Exception as e:
    #     logging.error(f"Error loading or parsing {qa_path}: {e}")
    #     user_question = "What is the current number of electors currently in a Scottish Parliament constituency? "
    
    # return user_question

with DAG("RAG_Query_DAG", default_args=default_args, schedule=timedelta(hours=12), catchup=False, max_active_runs=3, max_active_tasks=6) as dag:
    """RAG pipeline DAG for retrieval-augmented generation (RAG) using Ollama and Qdrant."""
    
    ollama_url = os.getenv("OLLAMA_HOST", "10.20.1.95:11433")
    qdrant_url = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
    retrieval_api_host = os.getenv("RETRIEVAL_API_HOST")
    retrieval_api_port = os.getenv("RETRIEVAL_API_PORT", 8000)
    rerank_api_host = os.getenv("RERANK_API_HOST")
    rerank_api_port = os.getenv("RERANK_API_PORT", 8001)
    llm_api_host = os.getenv("LLM_API_HOST")
    llm_api_port = os.getenv("LLM_API_PORT", 8002)
    ragas_api_host = os.getenv("RAGAS_API_HOST")
    ragas_api_port = os.getenv("RAGAS_API_PORT", 8003)
    
    llm_model = config_data.get("llm_model", "gemma2:9b")
    embed_model = config_data.get("embed_model", "imac/zpoint_large_embedding_zh")
    document_types = config_data.get("document_types", "squad")
    
    APIConfig = APIConfig(
        llm_model=llm_model,
        embed_model=embed_model,
        document_types=document_types
    )
    
    user_question = config_data.get("user_question", None)   
    generate_query_task = PythonOperator(
        task_id="generate_query_task",
        python_callable=get_user_question,
        op_kwargs={
            "user_question": user_question
        },
    )
    
    if USE_EXPERT:
        logging.info("Using expert retrieval")
        expert_retrieval_task = PythonOperator(
            task_id="expert_retrieval_task",
            python_callable=APIConfig.call_retrieval_api(
                api_host=retrieval_api_host, 
                api_port=retrieval_api_port, 
                types="expert",
                topk=5
            )
        )
        expert_validation_task = PythonOperator(
            task_id="expert_validation_task",
            python_callable=APIConfig.call_llm_api(
                api_host=llm_api_host, 
                api_port=llm_api_port, 
                types="validation",
            )
        )
        ExpertBranch = ExpertBranch()
        expert_branching_task = BranchPythonOperator(
            task_id="expert_branching_task",
            python_callable=ExpertBranch.branch_logic,
            op_kwargs={
                "USE_SIMILARITY": USE_SIMILARITY,
                "USE_KEYWORD": USE_KEYWORD
            },
        )        
    else:
        logging.info("Not using expert retrieval")
        expert_retrieval_task = None
        expert_validation_task = None
        expert_branching_task = None
    
    if USE_SIMILARITY:
        logging.info("Using similarity retrieval")
        similarity_retrieval_task = PythonOperator(
            task_id="similarity_retrieval_task",
            python_callable=APIConfig.call_retrieval_api(
                api_host=retrieval_api_host, 
                api_port=retrieval_api_port, 
                types="similarity",
                topk=10
            )
        )
    else:
        logging.info("Not using similarity retrieval")
        similarity_retrieval_task = None
    
    if USE_KEYWORD:
        logging.info("Using keyword retrieval")
        keyword_extraction_task = PythonOperator(
            task_id="keyword_extraction_task",
            python_callable=APIConfig.call_llm_api(
                api_host=llm_api_host, 
                api_port=llm_api_port, 
                types="keyword",
            )
        )
        keyword_retrieval_task = PythonOperator(
            task_id="keyword_retrieval_task",
            python_callable=APIConfig.call_retrieval_api(
                api_host=retrieval_api_host, 
                api_port=retrieval_api_port, 
                types="keyword",
                topk=10,
            )
        )
    else:
        logging.info("Not using keyword retrieval")
        keyword_extraction_task = None
        keyword_retrieval_task = None
    
    
    if USE_RERANK:
        logging.info("Using rerank")
        reranking_task = PythonOperator(
            task_id="reranking_task",
            python_callable=APIConfig.call_rerank_api(
                api_host=rerank_api_host, 
                api_port=rerank_api_port, 
                topk=5,
            )
        )
    else:
        logging.info("Not using rerank")
        reranking_task = None
    
    if reranking_task:
        search_results_types = "reranking_task"
    elif similarity_retrieval_task:
        search_results_types = "similarity_retrieval_task"
    elif keyword_retrieval_task:
        search_results_types = "keyword_retrieval_task"
    else:
        search_results_types = None
    
    llm_task = PythonOperator(
        task_id="llm_task",
        python_callable=APIConfig.call_llm_api(
            api_host=llm_api_host, 
            api_port=llm_api_port, 
            types="rag" if (similarity_retrieval_task or keyword_retrieval_task) else "general",
            search_results_types=search_results_types,
        )
    )
    
    if USE_RAGAS:
        logging.info("Using RAGAS evaluation")
        ragas_evaluation_task = PythonOperator(
            task_id="ragas_evaluation_task",
            python_callable=APIConfig.call_ragas_api(
                api_host=ragas_api_host,
                api_port=ragas_api_port,
                use_similarity=USE_SIMILARITY,
                use_keyword=USE_KEYWORD,
                # use_similarity=False,
                # use_keyword=False,
                use_rerank=USE_RERANK,
            )
        )
    else:
        logging.info("Not using RAGAS evaluation")
        ragas_evaluation_task = None
    
        
    retrieval_tasks = []
    temp_tasks = generate_query_task
    
    if USE_EXPERT:
        generate_query_task >> expert_retrieval_task >> expert_validation_task >> expert_branching_task
        temp_tasks = expert_branching_task  

    if USE_SIMILARITY:
        temp_tasks >> similarity_retrieval_task
        retrieval_tasks.append(similarity_retrieval_task)

    if USE_KEYWORD:
        temp_tasks >> keyword_extraction_task >> keyword_retrieval_task
        retrieval_tasks.append(keyword_retrieval_task)

    if USE_RERANK and retrieval_tasks:
        for task in retrieval_tasks:
            task >> reranking_task
        reranking_task >> llm_task
    elif len(retrieval_tasks) == 1:
        retrieval_tasks[0] >> llm_task
    else:
        temp_tasks >> llm_task
    
    if USE_RAGAS and any(retrieval_tasks):
        llm_task >> ragas_evaluation_task     