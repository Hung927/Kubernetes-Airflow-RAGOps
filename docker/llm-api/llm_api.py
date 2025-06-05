import os
import logging
import threading
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llm import LLM
import uvicorn
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("llm-api")

app = FastAPI(title="LLM API", description="API for llm tasks")

last_used_time = time.time()
INACTIVITY_TIMEOUT = 300

class LLMRequest(BaseModel):
    types: str = "rag"
    model: str = "gemma2:9b"
    temperature: float = 0.0
    keep_alive: str = "0s"
    num_ctx: int = 8192
    user_question: str
    search_results_types: Optional[str] = None
    search_results: Optional[str] = None

class MockTi:
    def __init__(self, user_question: str, search_results_types: str=None, search_results: str = None):
        self.user_question = user_question
        self.search_results_types = search_results_types
        self.search_results = search_results

    def xcom_pull(self, task_ids: str, key: str=None):
        if task_ids == 'generate_query_task' and key == 'return_value':
            return self.user_question or "What is the current number of electors currently in a Scottish Parliament constituency?"
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
    return {"status": "healthy", "service": "llm-api"}

@app.post("/llm")
def llm(request: LLMRequest):
    """執行 llm 任務的接口"""
    update_last_used_time()
    
    try:
        logger.info(f"LLM object created with: {request}")
        llm_obj = LLM(
            model=request.model,
            temperature=request.temperature,
            keep_alive=request.keep_alive,
            num_ctx=request.num_ctx,
        )
        
        mock_ti = MockTi(
            user_question=request.user_question,
            search_results_types=request.search_results_types,
            search_results=request.search_results
        )
        
        result = llm_obj.llm(
            types=request.types,
            ti=mock_ti
        )
        
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Error during llm: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
def startup_event():
    """啟動時的事件處理"""
    logger.info("LLM API starting up...")
    monitor_thread = threading.Thread(target=inactivity_monitor, daemon=True)
    monitor_thread.start()

if __name__ == "__main__":
    uvicorn.run("llm_api:app", host="0.0.0.0", port=8002, reload=False)