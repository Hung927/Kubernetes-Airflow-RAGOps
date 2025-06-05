import os
import logging
import threading
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from retrieval import Retrieval
import uvicorn
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("retrieval-api")

app = FastAPI(title="Retrieval API", description="API for retrieval tasks")

last_used_time = time.time()
INACTIVITY_TIMEOUT = 300

class RetrievalRequest(BaseModel):
    types: str = "similarity"
    document_types: str = "squad"
    topk: int = 10
    embed_model: str = "imac/zpoint_large_embedding_zh"
    user_question: str
    keyword_list: Optional[str] = None

class MockTi:
    def __init__(self, user_question: str, keyword_list: list = None):
        self.user_question = user_question
        self.keyword_list = keyword_list

    def xcom_pull(self, task_ids: str, key: str=None):
        if task_ids == 'generate_query_task' and key == 'return_value':
            return self.user_question or "What is the current number of electors currently in a Scottish Parliament constituency?"
        elif task_ids == 'keyword_extraction_task' and key == 'return_value':
            try:
                # 嘗試將字符串解析為 Python 列表
                import ast
                keywords = ast.literal_eval(self.keyword_list)
                logging.info(f"Parsed keyword list: {keywords}")
                return keywords
            except:
                # 如果解析失敗，返回原始字符串
                logging.warning(f"Could not parse keyword list: {self.keyword_list}")
                return self.keyword_list
        return None

# 初始化檢索對象
retrieval_instances = {}

def get_retrieval_instance(embed_model):
    """獲取或創建檢索實例"""
    global retrieval_instances
    if embed_model not in retrieval_instances:
        logger.info(f"Creating new retrieval instance for model: {embed_model}")
        retrieval_instances[embed_model] = Retrieval(embed_model=embed_model)
    return retrieval_instances[embed_model]

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
    return {"status": "healthy", "service": "retrieval-api"}

@app.post("/retrieve")
def retrieve(request: RetrievalRequest):
    """執行檢索任務的接口"""
    update_last_used_time()
    
    try:
        logger.info(f"Retrieval object created with: {request}")
        retrieval_obj = get_retrieval_instance(request.embed_model)
        
        # 創建模擬 ti 對象
        mock_ti = MockTi(
            user_question=request.user_question,
            keyword_list=request.keyword_list
        )
        
        # 執行檢索
        result = retrieval_obj.retrieval(
            types=request.types,
            document_types=request.document_types,
            topk=request.topk,
            ti=mock_ti
        )
        
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Error during retrieval: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
def startup_event():
    """啟動時的事件處理"""
    logger.info("Retrieval API starting up...")
    monitor_thread = threading.Thread(target=inactivity_monitor, daemon=True)
    monitor_thread.start()

if __name__ == "__main__":
    uvicorn.run("retrieval_api:app", host="0.0.0.0", port=8000, reload=False)