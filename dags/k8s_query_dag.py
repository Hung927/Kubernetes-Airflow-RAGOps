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

def get_user_question():
    """Get a random user question from the qa_pairs.json file."""
    
    # return "What did Stiglitz present in 2009 regarding global inequality?"
    qa_path = os.path.join(os.path.dirname(__file__), "data/qa_pairs.json")
    try:
        with open(qa_path, "r") as f:
            qa_data = json.load(f)
            user_question = random.choice(list(qa_data.keys()))
    except Exception as e:
        logging.error(f"Error loading or parsing {qa_path}: {e}")
        user_question = "What is the current number of electors currently in a Scottish Parliament constituency? "
    
    return user_question
    # return "Where did the Normans and Byzantines sign the peace treaty?"

with DAG("RAG_Query_DAG2", default_args=default_args, schedule=timedelta(hours=12), catchup=False) as dag:
    """RAG pipeline DAG for retrieval-augmented generation (RAG) using Ollama and Qdrant."""
    
    retrieval_image = "shaohung/airflow-retrieval:v1.1"
    rerank_image = "shaohung/airflow-rerank:v1.0"
    llm_image = "shaohung/airflow-llm:v1.0"
    ragas_image = "shaohung/airflow-ragas:v1.0"
    
    
    ollama_url = os.getenv("OLLAMA_HOST", "10.20.1.95:11433")
    qdrant_url = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
    
    llm_model = config_data.get("llm_model", "gemma2:9b")
    embed_model = config_data.get("embed_model", "imac/zpoint_large_embedding_zh")
    
    generate_query_task = PythonOperator(
        task_id="generate_query_task",
        python_callable=get_user_question,
    )
    
    if USE_EXPERT:
        logging.info("Using expert retrieval")
        expert_retrieval_task = KubernetesPodOperator(
            task_id="expert_retrieval_task",
            name="expert-retrieval",
            namespace="default",
            image=retrieval_image,
            cmds=["python", "retrieval_run.py"],
            arguments=[
                "--document-type", config_data.get("document_types"),
                "--types", "expert",
                "--topk", "5",
                "--embed-model", embed_model,
                "--user-question", "{{ ti.xcom_pull(task_ids='generate_query_task', key='return_value') }}"
            ],
            env_vars=[
                V1EnvVar(name="OLLAMA_HOST", value=ollama_url), 
                V1EnvVar(name="QDRANT_URL", value=qdrant_url)
            ],
            config_file="~/.kube/config",
            in_cluster=False,
            is_delete_operator_pod=True,
            get_logs=True,
            do_xcom_push=True,
        )
        expert_validation_task = KubernetesPodOperator(
            task_id="expert_validation_task",
            name="expert-validation",
            namespace="default",
            image=llm_image,
            arguments=[
                "--types", "validation",
                "--model", llm_model,
                "--user-question", "{{ ti.xcom_pull(task_ids='generate_query_task', key='return_value') }}"
            ],
            env_vars=[
                V1EnvVar(name="OLLAMA_HOST", value=ollama_url)
            ],
            config_file="~/.kube/config",
            in_cluster=False,
            is_delete_operator_pod=True,
            get_logs=True,
            do_xcom_push=True,
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
        similarity_retrieval_task = KubernetesPodOperator(
            task_id="similarity_retrieval_task",
            name="similarity-retrieval",
            namespace="default",
            image=retrieval_image,
            cmds=["python", "retrieval_run.py"],
            arguments=[
                "--document-type", config_data.get("document_types"),
                "--types", "similarity",
                "--topk", "10",
                "--embed-model", embed_model,
                "--user-question", "{{ ti.xcom_pull(task_ids='generate_query_task', key='return_value') }}"
            ],
            env_vars=[
                V1EnvVar(name="OLLAMA_HOST", value=ollama_url), 
                V1EnvVar(name="QDRANT_URL", value=qdrant_url)
            ],
            config_file="~/.kube/config",
            in_cluster=False,
            is_delete_operator_pod=True,
            get_logs=True,
            do_xcom_push=True,
        )
    else:
        logging.info("Not using similarity retrieval")
        similarity_retrieval_task = None
    
    if USE_KEYWORD:
        logging.info("Using keyword retrieval")
        keyword_extraction_task = KubernetesPodOperator(
            task_id="keyword_extraction_task",
            name="keyword-extraction",
            namespace="default",
            image=llm_image,
            arguments=[
                "--types", "keyword",
                "--model", llm_model,
                "--user-question", "{{ ti.xcom_pull(task_ids='generate_query_task', key='return_value') }}"
            ],
            env_vars=[
                V1EnvVar(name="OLLAMA_HOST", value=ollama_url)
            ],
            config_file="~/.kube/config",
            in_cluster=False,
            is_delete_operator_pod=True,
            get_logs=True,
            do_xcom_push=True,
        )
        keyword_retrieval_task = KubernetesPodOperator(
            task_id="keyword_retrieval_task",
            name="keyword-retrieval",
            namespace="default",
            image=retrieval_image,
            cmds=["python", "retrieval_run.py"],
            arguments=[
                "--document-type", config_data.get("document_types"),
                "--types", "keyword",
                "--topk", "10",
                "--embed-model", embed_model,
                "--user-question", "{{ ti.xcom_pull(task_ids='generate_query_task', key='return_value') }}",
                "--keyword-list", "{{ ti.xcom_pull(task_ids='keyword_extraction_task', key='return_value') }}"
            ],
            env_vars=[
                V1EnvVar(name="OLLAMA_HOST", value=ollama_url), 
                V1EnvVar(name="QDRANT_URL", value=qdrant_url)
            ],
            config_file="~/.kube/config",
            in_cluster=False,
            is_delete_operator_pod=True,
            get_logs=True,
            do_xcom_push=True,
        )
    else:
        logging.info("Not using keyword retrieval")
        keyword_extraction_task = None
        keyword_retrieval_task = None
    
    
    if USE_RERANK:
        logging.info("Using rerank")
        reranking_task = KubernetesPodOperator(
            task_id="reranking_task",
            name="reranking",
            namespace="default",
            image=rerank_image,
            cmds=["python", "rerank_run.py"],
            arguments=[
                "--user-question", "{{ ti.xcom_pull(task_ids='generate_query_task', key='return_value') }}",
                "--similarity-results", "{{ ti.xcom_pull(task_ids='similarity_retrieval_task', key='return_value') }}",
                "--keyword-results", "{{ ti.xcom_pull(task_ids='keyword_retrieval_task', key='return_value') }}",
                "--topk", "5"
            ],
            config_file="~/.kube/config",
            in_cluster=False,
            is_delete_operator_pod=True,
            get_logs=True,
            do_xcom_push=True,
        )
    else:
        logging.info("Not using rerank")
        reranking_task = None
    
    llm_task_op_kwargs = ["--types", "rag" if (similarity_retrieval_task or keyword_retrieval_task) else "general"]
    llm_task_op_kwargs.extend(["--model", llm_model])
    llm_task_op_kwargs.extend(["--user-question", "{{ ti.xcom_pull(task_ids='generate_query_task', key='return_value') }}"])
    if reranking_task:
        llm_task_op_kwargs.extend(["--search-results-types", "reranking_task"])
        llm_task_op_kwargs.extend(["--search-results", "{{ ti.xcom_pull(task_ids='reranking_task', key='return_value') }}" ])
    elif similarity_retrieval_task:
        llm_task_op_kwargs.extend(["--search-results-types", "similarity_retrieval_task"])
        llm_task_op_kwargs.extend(["--search-results", "{{ ti.xcom_pull(task_ids='similarity_retrieval_task', key='return_value') }}"])
    elif keyword_retrieval_task:
        llm_task_op_kwargs.extend(["--search-results-types", "keyword_retrieval_task"])
        llm_task_op_kwargs.extend(["--search-results", "{{ ti.xcom_pull(task_ids='keyword_retrieval_task', key='return_value') }}"])
    llm_task = KubernetesPodOperator(
        task_id="llm_task",
        name="llm",
        namespace="default",
        image=llm_image,
        arguments=llm_task_op_kwargs,
        env_vars=[
            V1EnvVar(name="OLLAMA_HOST", value=ollama_url)
        ],
        config_file="~/.kube/config",
        in_cluster=False,
        is_delete_operator_pod=True,
        get_logs=True,
        do_xcom_push=True,
    )
    
    if USE_RAGAS:
        dns_config = V1PodDNSConfig(
            nameservers=["8.8.8.8", "8.8.4.4"],
            options=[V1PodDNSConfigOption(name="ndots", value="5")]
        )
        
        logging.info("Using RAGAS evaluation")
        ragas_evaluation_task = KubernetesPodOperator(
            task_id="ragas_evaluation_task",
            name="ragas-evaluation",
            namespace="default",
            image=ragas_image,
            arguments=[
                "--user-question", "{{ ti.xcom_pull(task_ids='generate_query_task', key='return_value') }}",
                "--llm-answer", "{{ ti.xcom_pull(task_ids='llm_task', key='return_value') }}",
                "--similarity-results", "{{ ti.xcom_pull(task_ids='similarity_retrieval_task', key='return_value') }}",
                "--keyword-results", "{{ ti.xcom_pull(task_ids='keyword_retrieval_task', key='return_value') }}",
                "--rerank-results", "{{ ti.xcom_pull(task_ids='reranking_task', key='return_value') }}",
                "--use-similarity", str(USE_SIMILARITY),
                "--use-keyword", str(USE_KEYWORD),
                "--use-rerank", str(USE_RERANK)
            ],
            env_vars=[
                V1EnvVar(name="OPENAI_API_KEY", value=os.getenv("OPENAI_API_KEY"))
            ],
            config_file="~/.kube/config",
            in_cluster=False,
            is_delete_operator_pod=True,
            get_logs=True,
            do_xcom_push=True,
            dns_config=dns_config,
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