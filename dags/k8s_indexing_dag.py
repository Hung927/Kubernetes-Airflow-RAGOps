from dotenv import load_dotenv
load_dotenv(dotenv_path="dags/.env")

import os
import sys
import json
sys.path.append(os.getcwd())
from airflow import DAG
from pendulum import duration
from datetime import datetime
from kubernetes.client.models import (
    V1EnvVar,
    V1Volume, 
    V1VolumeMount, 
    V1HostPathVolumeSource,
    V1PodDNSConfig, 
    V1PodDNSConfigOption
)
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator

from plugins import json_update_sensor


default_args = {
  "owner": "HUNG",
  "start_date": datetime(2025, 4, 28),
  "retries": 3,
  "retry_delay": duration(minutes=5),
}


with DAG("K8S_Indexing_DAG", default_args=default_args, schedule="@once", catchup=False) as dag:
    """Config file upload check and data processing pipeline DAG."""
    
    config_data = json.load(open("dags/config.json", "r"))
    ollama_url = os.getenv("OLLAMA_HOST", "10.20.1.95:11433")
    qdrant_url = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
    data_processing_image = "shaohung/airflow-data-processing:v1.0"
    data_embedding_image = "shaohung/airflow-data-embedding:v1.0"
    
    config_volume = V1Volume(
        name='airflow-config',
        host_path=V1HostPathVolumeSource(
            path='/home/ubuntu/hung/Kubernetes-Airflow-RAGOps/dags',
            type='Directory'
        )
    )

    config_volume_mount = V1VolumeMount(
        name='airflow-config',
        mount_path='/app/dags',
        read_only=False
    )
    
    dns_config = V1PodDNSConfig(
        nameservers=["8.8.8.8", "8.8.4.4"],
        options=[V1PodDNSConfigOption(name="ndots", value="5")]
    )

        
    uploaded_files_check_task = json_update_sensor.JsonUpdateSensor(
        task_id="uploaded_files_check_task",
        filepath="dags/config.json",
        key="uploaded_files",
        expected_value=config_data["uploaded_files"],
        poke_interval=3,
        timeout=60
    )
    
    data_processing_task = KubernetesPodOperator(
        task_id="data_processing_task",
        name="data-processing",
        namespace="default",
        image=data_processing_image,
        cmds=["python", "data_processing_run.py"],
        arguments=[
            "--config-path", "/app/dags/config.json",
            "--data-context-path", "/app/dags/data/data_context.json"
        ],
        volumes=[config_volume],
        volume_mounts=[config_volume_mount],
        config_file="~/.kube/config",
        in_cluster=False,
        is_delete_operator_pod=True,
        get_logs=True,
        do_xcom_push=True,
        dns_config=dns_config,
    )
    
    file_list_update_check_task = json_update_sensor.JsonUpdateSensor(
        task_id="file_list_update_check_task",
        filepath="dags/config.json",
        key="file_list",
        expected_value=config_data["file_list"],
        poke_interval=3,
        timeout=600
    )
    
    data_embedding_task = KubernetesPodOperator(
        task_id="data_embedding_task",
        name="data-embedding",
        namespace="default",
        image=data_embedding_image,
        cmds=["python", "data_embedding_run.py"],
        arguments=[
            "--data-context-path", "/app/dags/data/data_context.json",
            "--embed-model", config_data.get("embed_model", "imac/zpoint_large_embedding_zh")
        ],
        volumes=[config_volume],
        volume_mounts=[config_volume_mount],
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
    
    uploaded_files_check_task >> data_processing_task
    file_list_update_check_task >> data_embedding_task
