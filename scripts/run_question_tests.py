import os
import sys
import json
import time
import logging
import requests
import argparse
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'rag_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger('rag-test')

DEFAULT_QUESTIONS = [
    "The immune systems of bacteria have enzymes that protect against infection by what kind of cells?",
    "Who proposed that innate inertial is the natural state of objects?",
    "Besides the North Sea and the Irish Channel, what else was lowered in the last cold phase?",
    "What organization is devoted to Jihad against Israel?",
    "What is the total make up of fish species living in the Amazon?",
    "What are the stages in a compound engine called?",
    "When was the cabinet-level Energy Department created?",
    "US is concerned about confrontation of the Middle East with which other country?",
    "How did user of Tymnet connect?",
    "What did Stiglitz present in 2009 regarding global inequality?"
]

class RAGTester:
    def __init__(self, airflow_url, config_path, auth=None, results_dir=None):
        """初始化 RAG 測試器"""
        self.airflow_url = airflow_url
        self.config_path = config_path
        self.auth = auth
        
        # 創建結果目錄
        self.results_dir = results_dir or f"rag_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_dir, exist_ok=True)
        logger.info(f"結果將保存在: {self.results_dir}")
        
        # 讀取原始配置
        with open(self.config_path, 'r') as f:
            self.original_config = json.load(f)
        logger.info(f"已讀取配置文件: {self.config_path}")
    
    def _write_config(self, config):
        """寫入配置文件"""
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=4)
    
    def update_question(self, question):
        """更新配置文件中的問題"""
        config = self.original_config.copy()
        config["user_question"] = question
        self._write_config(config)
        logger.info(f"已更新問題: {question}")
    
    def trigger_dag(self, dag_id="K8S_API_Query_DAG"):
        """觸發 DAG 執行"""
        endpoint = f"{self.airflow_url.rstrip('/')}/api/v1/dags/{dag_id}/dagRuns"
        
        run_id = f"manual__{datetime.now().strftime('%Y%m%d%H%M%S')}"
        payload = {
            "dag_run_id": run_id,
            "conf": {}
        }
        
        headers = {
            "Content-Type": "application/json",
        }
        
        try:
            logger.info(f"觸發 DAG: {dag_id} (run_id: {run_id})")
            username = "admin"
            password = "WRrxhXsHpwWn57a6"
            
            if self.auth:
                auth_tuple = self.auth
            else:
                auth_tuple = (username, password)
                
            logger.info(f"使用認證: {auth_tuple[0]}:{'*' * len(auth_tuple[1])}")  # 記錄用戶名但隱藏密碼
            
            response = requests.post(
                endpoint, 
                json=payload, 
                auth=auth_tuple, 
                headers=headers
            )
            response.raise_for_status()
            logger.info(f"DAG 觸發成功: {response.json()}")
            return run_id
        except requests.exceptions.RequestException as e:
            logger.error(f"觸發 DAG 失敗: {e}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"回應狀態碼: {e.response.status_code}")
                logger.error(f"回應內容: {e.response.text}")
            return None
    
    def wait_for_dag_completion(self, dag_id, run_id, timeout=3600, check_interval=30):
        """等待 DAG 完成執行"""
        endpoint = f"{self.airflow_url.rstrip('/')}/api/v1/dags/{dag_id}/dagRuns/{run_id}"
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(endpoint, auth=self.auth)
                response.raise_for_status()
                data = response.json()
                state = data.get('state')
                
                if state == 'success':
                    logger.info(f"DAG 執行成功: {run_id}")
                    return True
                elif state == 'failed':
                    logger.error(f"DAG 執行失敗: {run_id}")
                    return False
                
                logger.info(f"DAG 執行狀態: {state}, 等待...")
                time.sleep(check_interval)
            except Exception as e:
                logger.error(f"檢查 DAG 狀態時出錯: {e}")
                time.sleep(check_interval)
        
        logger.error(f"DAG 執行超時 (超過 {timeout} 秒)")
        return False
    
    def get_xcom_value(self, dag_id, run_id, task_id):
        """從 XCom 獲取任務結果"""
        endpoint = f"{self.airflow_url.rstrip('/')}/api/v1/dags/{dag_id}/dagRuns/{run_id}/taskInstances/{task_id}/xcomEntries/return_value"
        
        try:
            response = requests.get(endpoint, auth=self.auth)
            response.raise_for_status()
            return response.json().get('value')
        except Exception as e:
            logger.error(f"獲取 XCom 值失敗 (task_id={task_id}): {e}")
            return None
    
    def collect_results(self, dag_id, run_id, question):
        """收集所有步驟的結果"""
        results = {
            "question": question,
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
        }
        
        # 收集各步驟結果
        task_ids = [
            "similarity_retrieval_task",  # 相似度檢索結果
            # "keyword_extraction_task",    # 關鍵詞抽取結果
            # "keyword_retrieval_task",     # 關鍵詞檢索結果
            # "reranking_task",             # 重排序結果
            "llm_task",                   # LLM生成答案
            "ragas_evaluation_task"       # RAGAS評估結果
        ]
        
        for task_id in task_ids:
            value = self.get_xcom_value(dag_id, run_id, task_id)
            if task_id == "ragas_evaluation_task" and isinstance(value, str):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    logger.warning(f"無法解析 RAGAS 結果為 JSON: {value[:100]}...")
        
            results[task_id] = value
        
        logger.info(f"已收集問題 '{question}' 的所有結果")
        return results
    
    def save_result(self, result, question_number):
        """保存單個問題的結果"""
        sanitized_question = result["question"][:30].replace(" ", "_").replace("?", "")
        filename = f"{question_number:02d}_{sanitized_question}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"已保存結果到: {filepath}")
    
    def save_summary(self, all_results):
        """保存所有問題結果的總結"""
        summary = {
            "test_time": datetime.now().isoformat(),
            "total_questions": len(all_results),
            "questions": [r["question"] for r in all_results],
            "run_ids": [r["run_id"] for r in all_results],
            "llm_answers": [r.get("llm_task") for r in all_results],
            "ragas_scores": [r.get("ragas_evaluation_task") for r in all_results]
        }
        
        filepath = os.path.join(self.results_dir, "summary.json")
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"已保存摘要到: {filepath}")
        
        # 創建一個簡單的可讀報告
        report_path = os.path.join(self.results_dir, "report.md")
        with open(report_path, 'w') as f:
            f.write("# RAG 測試報告\n\n")
            f.write(f"測試時間: {summary['test_time']}\n\n")
            f.write(f"總問題數: {summary['total_questions']}\n\n")
            
            for i, (question, answer) in enumerate(zip(summary['questions'], summary['llm_answers'] or [])):
                f.write(f"## 問題 {i+1}: {question}\n\n")
                f.write(f"回答: {answer}\n\n")
                
                # 如果有 RAGAS 評分
                ragas = summary['ragas_scores'][i] if summary['ragas_scores'] else None
                if ragas:
                    f.write("### RAGAS 評分\n\n")
                    for metric, score in ragas.items():
                        f.write(f"- {metric}: {score}\n")
                f.write("\n---\n\n")
                
        logger.info(f"已生成報告: {report_path}")
    
    def run_test(self, questions):
        """執行測試流程"""
        all_results = []
        
        for i, question in enumerate(questions):
            logger.info(f"[{i+1}/{len(questions)}] 處理問題: {question}")
            
            # 更新問題
            self.update_question(question)
            
            # 觸發 DAG
            run_id = self.trigger_dag()
            if not run_id:
                logger.error(f"問題 {i+1} 觸發失敗，跳過")
                continue
            
            # 等待 DAG 完成
            success = self.wait_for_dag_completion("K8S_API_Query_DAG", run_id)
            if not success:
                logger.error(f"問題 {i+1} 執行失敗，跳過結果收集")
                continue
            
            # 收集結果
            results = self.collect_results("K8S_API_Query_DAG", run_id, question)
            all_results.append(results)
            
            # 保存此問題的結果
            self.save_result(results, i+1)
            
            # 等待一段時間再繼續下一個問題
            if i < len(questions) - 1:
                wait_time = 1
                logger.info(f"等待 {wait_time} 秒後繼續下一個問題...")
                time.sleep(wait_time)
        
        # 保存總結
        self.save_summary(all_results)
        
        return all_results

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='自動測試RAG流程並收集結果')
    parser.add_argument('--airflow-url', default='http://localhost:8080', help='Airflow 網頁伺服器 URL')
    parser.add_argument('--config-path', default='/home/ubuntu/hung/Kubernetes-Airflow-RAGOps/dags/config.json', 
                        help='DAG配置檔案路徑')
    parser.add_argument('--questions-file', help='包含問題的JSON檔案路徑')
    parser.add_argument('--username', help='Airflow 使用者名稱', default='admin')
    parser.add_argument('--password', help='Airflow 密碼', default='WRrxhXsHpwWn57a6')
    parser.add_argument('--results-dir', help='結果儲存目錄', default='/home/ubuntu/hung/Kubernetes-Airflow-RAGOps/result/')
    parser.add_argument('--indices', help='要使用的問題索引, 用逗號分隔 (例如: 0,2,4)')
    
    args = parser.parse_args()
    
    # 設置認證
    auth = None
    if args.username and args.password:
        auth = (args.username, args.password)
    
    # 載入問題
    questions = DEFAULT_QUESTIONS
    if args.questions_file:
        try:
            with open(args.questions_file, 'r') as f:
                questions = json.load(f)
        except Exception as e:
            logger.error(f"載入問題檔案失敗: {e}, 使用預設問題")
    
    # 過濾問題
    if args.indices:
        try:
            indices = [int(i.strip()) for i in args.indices.split(',')]
            questions = [questions[i] for i in indices if 0 <= i < len(questions)]
        except Exception as e:
            logger.error(f"解析索引失敗: {e}, 使用全部問題")
    
    # 初始化測試器
    tester = RAGTester(
        airflow_url=args.airflow_url,
        config_path=args.config_path,
        auth=auth,
        results_dir=args.results_dir
    )
    
    # 執行測試
    logger.info(f"開始測試 {len(questions)} 個問題")
    tester.run_test(questions)
    logger.info("測試完成!")

if __name__ == "__main__":
    main()
