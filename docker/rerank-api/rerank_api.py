import os
import logging
import threading
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rerank import Reranker
import uvicorn
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("rerank-api")

app = FastAPI(title="Rerank API", description="API for rerank tasks")

last_used_time = time.time()
INACTIVITY_TIMEOUT = 300

class RerankRequest(BaseModel):
    topk: int = 5
    user_question: str
    similarity_results: Optional[str] = None
    keyword_results: Optional[str] = None

class MockTi:
    def __init__(self, user_question: str, similarity_results: str = None, keyword_results: str = None):
        self.user_question = user_question
        self.similarity_results = similarity_results
        self.keyword_results = keyword_results

    def xcom_pull(self, task_ids: str, key: str = None):
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

def update_last_used_time():
    """更新最後使用時間"""
    global last_used_time
    last_used_time = time.time()

def inactivity_monitor():
    """監控不活躍時間的函數"""
    global last_used_time
    while True:
        time.sleep(60)  # 每分鐘檢查一次
        if time.time() - last_used_time > INACTIVITY_TIMEOUT:
            logger.info(f"No activity for {INACTIVITY_TIMEOUT} seconds, shutting down...")
            os._exit(0)

@app.get("/")
def root():
    """健康檢查接口"""
    update_last_used_time()
    return {"status": "healthy", "service": "rerank-api"}

@app.post("/rerank")
def rerank(request: RerankRequest):
    """執行 Rerank 任務的接口"""
    update_last_used_time()
    
    try:
        logger.info(f"Rerank object created with: {request}")
        rerank_obj = Reranker()
        
        mock_ti = MockTi(
            user_question=request.user_question,
            similarity_results=request.similarity_results,
            keyword_results=request.keyword_results
        )
        
        result = rerank_obj.rerank(
            topk=request.topk,
            ti=mock_ti
        )
        
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Error during rerank: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
def startup_event():
    """啟動時的事件處理"""
    logger.info("Rerank API starting up...")
    monitor_thread = threading.Thread(target=inactivity_monitor, daemon=True)
    monitor_thread.start()

if __name__ == "__main__":
    uvicorn.run("rerank_api:app", host="0.0.0.0", port=8001, reload=False)