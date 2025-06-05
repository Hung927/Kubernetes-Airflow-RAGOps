import os
import logging
import threading
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ragas_evaluator import Ragas
import uvicorn
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ragas-api")

app = FastAPI(title="RAGAS API", description="API for ragas tasks")

last_used_time = time.time()
INACTIVITY_TIMEOUT = 300

class LLMRequest(BaseModel):
    user_question: str
    llm_answer: str
    similarity_results: Optional[str] = None
    keyword_results: Optional[str] = None
    rerank_results: Optional[str] = None
    use_similarity: bool = False
    use_keyword: bool = False
    use_rerank: bool = False

class MockTi:
    def __init__(self, user_question: str = None, llm_answer: str = None, 
                 similarity_results: str = None, keyword_results: str = None, rerank_results: str = None):
        self.user_question = user_question
        self.llm_answer = llm_answer
        self.similarity_results = similarity_results
        self.keyword_results = keyword_results
        self.rerank_results = rerank_results
    
    def xcom_pull(self, task_ids, key=None):
        if task_ids == 'generate_query_task' and key == 'return_value':
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
    return {"status": "healthy", "service": "ragas-api"}

@app.post("/ragas")
def ragas(request: LLMRequest):
    """執行 ragas 任務的接口"""
    update_last_used_time()
    
    try:
        logger.info(f"RAGAS object created with: {request}")
        ragas_obj = Ragas(
            qa_path="/app/data/qa_pairs.json"
        )
        
        mock_ti = MockTi(
            user_question=request.user_question,
            llm_answer=request.llm_answer,
            similarity_results=request.similarity_results,
            keyword_results=request.keyword_results,
            rerank_results=request.rerank_results
        )
        
        use_similarity = request.use_similarity
        use_keyword = request.use_keyword
        use_rerank = request.use_rerank
        
        result = ragas_obj.ragas(
            ti=mock_ti,
            USE_SIMILARITY=use_similarity,
            USE_KEYWORD=use_keyword,
            USE_RERANK=use_rerank
        )
        
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Error during ragas: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
def startup_event():
    """啟動時的事件處理"""
    logger.info("RAGAS API starting up...")
    monitor_thread = threading.Thread(target=inactivity_monitor, daemon=True)
    monitor_thread.start()

if __name__ == "__main__":
    uvicorn.run("ragas_api:app", host="0.0.0.0", port=8003, reload=False)