"""
A model worker executes the model.
"""
import json
import os
import threading
import time
import uuid

import chromadb
import requests
import uvicorn
from chromadb.utils import embedding_functions
from fastapi import FastAPI, Request
from sentence_transformers import CrossEncoder

from ..config import LOG_DIR, WORKER_HEART_BEAT_INTERVAL
from ..utils import build_logger, pretty_print_semaphore

GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
logger = build_logger("model_worker", f"model_worker_{worker_id}.log", LOG_DIR)
global_counter = 0

model_semaphore = None

CONTROLLER_ADDRESS = os.environ.get("CONTROLLER_ADDRESS")
WORKER_ADDRESS = os.environ.get("WORKER_ADDRESS")
WORKER_HOST = os.environ.get("WORKER_HOST")
QA_BANK_DB_PATH = os.environ.get("QA_BANK_DB_PATH")
WORKER_PORT = int(os.environ.get("WORKER_PORT", 40000))
MODEL_NAME = os.environ.get("MODEL_NAME", "rag")
NO_REGISTER = os.environ.get("NO_REGISTER", False)
LIMIT_MODEL_CONCURRENCY = int(os.environ.get("LIMIT_MODEL_CONCURRENCY", 5))
EMBEDDING_MODEL_PATH = os.environ.get("EMBEDDING_MODEL_PATH")
RERANKER_MODEL_PATH = os.environ.get("RERANKER_MODEL_PATH")
DEVICE = "cuda"

embedding_fn = embedding_functions.DefaultEmbeddingFunction() if EMBEDDING_MODEL_PATH == "default" \
    else embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_PATH)

def heart_beat_worker(controller):
    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()


class FeedbackWorker:
    def __init__(self):
        self.feedback_client = chromadb.PersistentClient(path=QA_BANK_DB_PATH)
        self.kg_collection = self.feedback_client.get_or_create_collection(name="KG_train", embedding_function=embedding_fn)
        self.ca_collection = self.feedback_client.get_or_create_collection(name="CA_train", embedding_function=embedding_fn)
        self.embedding_fn = embedding_fn
        self.reranker_model = CrossEncoder(RERANKER_MODEL_PATH, max_length=512)

        if not NO_REGISTER:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(
                target=heart_beat_worker, args=(self,))
            self.heart_beat_thread.start()

    def register_to_controller(self):
        logger.info("Register to controller")

        url = CONTROLLER_ADDRESS + "/register_worker"
        data = {
            "worker_name": WORKER_ADDRESS,
            "check_heart_beat": True,
            "worker_status": self.get_status()
        }
        r = requests.post(url, json=data)
        assert r.status_code == 200

    def send_heart_beat(self):
        logger.info(f"Send heart beat. Models: {[MODEL_NAME]}. "
                    f"Semaphore: {pretty_print_semaphore(model_semaphore)}. "
                    f"global_counter: {global_counter}")

        url = CONTROLLER_ADDRESS + "/receive_heart_beat"

        while True:
            try:
                ret = requests.post(url, json={
                    "worker_name": WORKER_ADDRESS,
                    "queue_length": self.get_queue_length()}, timeout=30)
                exist = ret.json()["exist"]
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"heart beat error: {e}")
            time.sleep(5)

        if not exist:
            self.register_to_controller()

    def get_queue_length(self):
        if model_semaphore is None:
            return 0
        else:
            return LIMIT_MODEL_CONCURRENCY - model_semaphore._value + (len(
                model_semaphore._waiters) if model_semaphore._waiters is not None else 0)

    def get_status(self):
        return {
            "model_names": [MODEL_NAME],
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    def initial_retrieval(self, query, collection):
        n_results = 50 if query else None
        res_chunks = list(collection.query(query_texts=query, n_results=n_results)['documents'][0])
        return res_chunks

    def get_sentence_pairs(self, query, chunks):
        sentence_pairs = [[query, chunks[i]] for i in range(len(chunks))]
        return sentence_pairs

    def reranker_top_k(self, sentence_pairs, top_k):
        # calculate scores of sentence pairs
        sort_scores = {}
        scores = self.reranker_model.predict(sentence_pairs)
        for i, score in enumerate(scores):
            sort_scores[i] = score
        scores_ = sorted(sort_scores.items(), key=lambda item: item[1], reverse=True)
        top_k_index = [scores_[indx][0] for indx in range(top_k)]
        return top_k_index

    def query_search(self, params):
        source = params.get('source')
        query = params.get('user_query')

        collections = [getattr(self, name, None) for name in params.get('collections')]
        if None in collections:
            raise NotImplementedError
        
        top_k = params.get('top_k')
        if isinstance(top_k, int):
            top_k = [top_k] * len(collections)
        assert len(collections)==len(top_k), "The number of top_k configs should be the same with collections"
        
        final_res = []
        for c in zip(collections, top_k):
            retr_chunks = self.initial_retrieval(query, c[0])
            sentence_pairs = self.get_sentence_pairs(query, retr_chunks)
            top_k_index = self.reranker_top_k(sentence_pairs, c[1])
            res = [retr_chunks[indx] for indx in top_k_index]
            final_res += res
        
        return final_res


app = FastAPI()


def release_model_semaphore(fn=None):
    model_semaphore.release()
    if fn is not None:
        fn()


@app.post("/worker_rag_query")
async def rag_query(request: Request):
    params = await request.json()
    return worker.query_search(params)


@app.post("/worker_get_status")
async def get_status(request: Request):
    return worker.get_status()


if __name__ == "__main__":
    worker = FeedbackWorker()
    uvicorn.run(app, host=WORKER_HOST, port=WORKER_PORT, log_level="info")
