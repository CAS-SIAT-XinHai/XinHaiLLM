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
PRO_KNOWLEDGE_DB_PATH = os.environ.get("PRO_KNOWLEDGE_DB_PATH")
SS_KNOWLEDGE_DB_PATH = os.environ.get("SS_KNOWLEDGE_DB_PATH")
WORKER_PORT = int(os.environ.get("WORKER_PORT", 40000))
MODEL_NAME = os.environ.get("MODEL_NAME", "rag")
NO_REGISTER = os.environ.get("NO_REGISTER", False)
LIMIT_MODEL_CONCURRENCY = int(os.environ.get("LIMIT_MODEL_CONCURRENCY", 5))
EMBEDDING_MODEL_PATH = os.environ.get("EMBEDDING_MODEL_PATH")
RERANKER_MODEL_PATH = os.environ.get("RERANKER_MODEL_PATH")
DEVICE = "cuda"

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_PATH)


def heart_beat_worker(controller):
    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()


class RAGWorker:
    def __init__(self):
        self.pro_client = chromadb.PersistentClient(path=PRO_KNOWLEDGE_DB_PATH)
        self.ss_client = chromadb.PersistentClient(path=SS_KNOWLEDGE_DB_PATH)
        ### 因为知识库是用langchain建立的
        self.pro_collection = self.pro_client.get_or_create_collection(name="langchain",
                                                                       embedding_function=embedding_fn)
        self.ss_collection = self.ss_client.get_or_create_collection(name="langchain", embedding_function=embedding_fn)
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

    def initial_retrieval(self, query):
        pro_chunks = []
        ss_chunks = []
        chunks_pro = self.pro_collection.query(query_texts=query, n_results=70)['documents'][0]
        chunks_ss = self.ss_collection.query(query_texts=query, n_results=70)['documents'][0]
        for i in range(70):
            pro_chunks.append(chunks_pro[i])
        for i in range(70):
            ss_chunks.append(chunks_ss[i])
        return pro_chunks, ss_chunks

    def initial_retrieval_with_meta(self, query, exclude):
        n_results = 30
        if exclude:
            where = {"id": {"$nin": exclude}}
            query_res = self.pro_collection.query(query_texts=query, n_results=n_results, where=where)
        else:
            query_res = self.pro_collection.query(query_texts=query, n_results=n_results)
        docs = query_res["documents"][0]
        metas = query_res["metadatas"][0]
        return docs, metas

    def get_sentence_pairs(self, query, chunks):
        sentence_pairs = [[query, chunks[i]] for i in range(len(chunks))]
        return sentence_pairs

    def reranker_top2(self, sentence_pairs):
        # calculate scores of sentence pairs
        sort_scores = {}
        scores = self.reranker_model.predict(sentence_pairs)
        for i, score in enumerate(scores):
            sort_scores[i] = score
        scores_ = sorted(sort_scores.items(), key=lambda item: item[1], reverse=True)
        index1, index2 = scores_[0][0], scores_[1][0]
        return index1, index2

    def reranker_topk(self, sentence_pairs, top_k):
        # calculate scores of sentence pairs
        sort_scores = {}
        scores = self.reranker_model.predict(sentence_pairs)
        for i, score in enumerate(scores):
            sort_scores[i] = score
        scores_ = sorted(sort_scores.items(), key=lambda item: item[1], reverse=True)
        # make sure top_k does not exceed the number of available scores
        top_k = min(top_k, len(scores_))
        topk_indx = [scores_[i][0] for i in range(top_k)]
        return topk_indx


    def rag(self, params):
        ### metadatas的格式：[{"source": "Human"}, {"source": "AI"}]
        ### documents格式: ['one', 'tow']
        query = params.get('user_query')
        pro_chunks, ss_chunks = self.initial_retrieval(query)
        pro_sentence_pairs = self.get_sentence_pairs(query, pro_chunks)
        ss_sentence_pairs = self.get_sentence_pairs(query, ss_chunks)
        pro_index1, pro_index2 = self.reranker_top2(pro_sentence_pairs)
        ss_index1, ss_index2 = self.reranker_top2(ss_sentence_pairs)

        return json.dumps({
            "rag_pro_knowledge_1": pro_chunks[pro_index1],
            "rag_pro_knowledge_2": pro_chunks[pro_index2],
            "rag_ss_knowledge_1": pro_chunks[ss_index1],
            "rag_ss_knowledge_2": pro_chunks[ss_index2]
        }, ensure_ascii=False)

    def rag_meta(self, params):
        query = params.get('user_query')
        top_k = params.get('top_k', 5)
        exclude = params.get('exclude', [])

        # rids = []
        # if no_repetition:
        #     cache_folder_path = "../../examples/PsyTraArena/cache/"
        #     cache_file_path = "reserved_ids.json"
        #     if not os.path.exists(cache_folder_path):
        #         os.makedirs(cache_folder_path)
        #     rids_path = os.path.join(cache_folder_path, cache_file_path)

        #     if not os.path.exists(rids_path):
        #         tmp_list = []
        #         with open(rids_path, 'w') as f:
        #             json.dump(tmp_list, f)
        #     else:
        #         with open(rids_path, 'r') as f:
        #             tmp_list = json.load(f)
        #         rids = sum(tmp_list, [])

        docs, metas = self.initial_retrieval_with_meta(query, exclude)
        pro_sentence_pairs = self.get_sentence_pairs(query, docs)
        topk_indx = self.reranker_topk(pro_sentence_pairs, top_k)
        topk_docs = [docs[indx] for indx in topk_indx]
        topk_metas = [metas[indx] for indx in topk_indx]

        # if no_repetition:
        #     new_rids = [c["id"] for c in topk_metas]
        #     tmp_list.append(new_rids)
        #     with open(rids_path, 'w') as f:
        #         json.dump(tmp_list, f)

        return json.dumps({
            "rag_pro_knowledge_docs": topk_docs,
            "rag_pro_knowledge_metas": topk_metas
        }, ensure_ascii=False)

    def rag_storage(self, params):
        query = params.get('user_query')
        top_k = params.get('top_k', 5)
        pro_collection = self.pro_client.get_or_create_collection(name="psyschool_4p_zh_20240908-0_summary",
                                                                  embedding_function=embedding_fn)
        pro_chunks = [i for i in pro_collection.query(query_texts=query, n_results=70)['documents'][0]]
        pro_sentence_pairs = self.get_sentence_pairs(query, pro_chunks)
        topk_indx = self.reranker_topk(pro_sentence_pairs, top_k)
        topk_chunks = [pro_chunks[indx] for indx in topk_indx]

        return json.dumps({
            "topk_chunks": topk_chunks
        }, ensure_ascii=False)


app = FastAPI()


def release_model_semaphore(fn=None):
    model_semaphore.release()
    if fn is not None:
        fn()


@app.post("/worker_rag_query")
async def rag_query(request: Request):
    params = await request.json()
    return worker.rag(params)


@app.post("/worker_rag_query_meta")
async def rag_query_meta(request: Request):
    params = await request.json()
    return worker.rag_meta(params)


@app.post("/worker_rag_storage")
async def rag_storage(request: Request):
    params = await request.json()
    return worker.rag_storage(params)


@app.post("/worker_get_status")
async def get_status(request: Request):
    return worker.get_status()


if __name__ == "__main__":
    worker = RAGWorker()
    uvicorn.run(app, host=WORKER_HOST, port=WORKER_PORT, log_level="info")
