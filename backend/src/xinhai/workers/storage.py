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
from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from xinhai.types.message import XinHaiChatMessage
from xinhai.types.storage import XinHaiFetchMessagesResponse, XinHaiFetchMessagesRequest, XinHaiStoreMessagesRequest
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
DB_PATH = os.environ.get("DB_PATH")
WORKER_PORT = int(os.environ.get("WORKER_PORT", 40000))
NO_REGISTER = os.environ.get("NO_REGISTER", False)
MODEL_NAME = os.environ.get("MODEL_NAME", "storage")
LIMIT_MODEL_CONCURRENCY = int(os.environ.get("LIMIT_MODEL_CONCURRENCY", 5))
EMBEDDING_MODEL_PATH = os.environ.get("EMBEDDING_MODEL_PATH")
DEVICE = "cuda"

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_PATH)


def heart_beat_worker(controller):
    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()


class StorageWorker:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=DB_PATH)
        self.embedding_fn = embedding_fn

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

    def insert_data(self, params):
        ### metadatas的格式：[{"source": "Human"}, {"source": "AI"}]
        ### documents格式: ['one', 'tow']
        user_id = params.get('user_id')
        documents = params.get('documents', [])
        metadatas = params.get('metadatas', [])
        short_memory = params.get("short_memory")

        if user_id is None or not documents or not metadatas:
            return json.dumps({
                "error_code": 1,
                "error_message": "Missing required parameters"
            })
        if short_memory:
            collection = self.client.get_or_create_collection(
                name="User_" + str(user_id),
                embedding_function=self.embedding_fn
            )
        else:
            collection = self.client.get_or_create_collection(
                name="User_" + str(user_id) + "_summary",
                embedding_function=self.embedding_fn
            )
        res = collection.count()
        ids = [f'{user_id}_{i + res}' for i in range(len(documents))]
        collection.add(documents=documents, ids=ids, metadatas=metadatas)
        logger.info(f'User_{user_id}\'s memory_storage adds a message')
        return json.dumps({
            "user_id": user_id,
            "new_document_count": len(documents),
            "error_code": 0
        })

    def get_data(self, params):
        ### 返回第k次及之后的对话
        user_id = params.get('user_id')

        if user_id is None:
            return json.dumps({
                "error_code": 1,
                "error_message": "Missing required parameters"
            })

        short_term_collection = self.client.get_or_create_collection(name="User_" + str(user_id),
                                                          embedding_function=self.embedding_fn)
        summary_collection = self.client.get_or_create_collection(name="User_" + str(user_id) + "_summary",
                                                          embedding_function=self.embedding_fn)
        short_term_dialogues = short_term_collection.get(include=['documents'])['documents']
        short_term_sources = short_term_collection.get(include=['metadatas'])['metadatas']
        summary_dialogues = summary_collection.get(include=['documents'])['documents']
        summary_sources = summary_collection.get(include=['metadatas'])['metadatas']
        # res = collection.count()
        # results = []
        # for i in range(k, res):
        #     results.append(sources[i]['source:  '] + dialogues[i])
        return json.dumps({
            "user_id": user_id,
            "short_term_documents": short_term_dialogues,
            "short_term_metadatas": short_term_sources,
            "summary_dialogues": summary_dialogues,
            "summary_sources": summary_sources,
            "error_code": 0
        })

    def search_similar(self, params):
        user_id = params.get('user_id')
        query = params.get('query')
        k = params.get('k', 4)

        if user_id is None or query is None:
            return json.dumps({
                "error_code": 1,
                "error_message": "Missing required parameters"
            })

        collection = self.client.get_or_create_collection(name="User_" + str(user_id),
                                                          embedding_function=self.embedding_fn)
        search = collection.query(query_texts=query, n_results=k)
        dialogues = search['documents'][0]
        sources = search['metadatas'][0]
        results = []
        for i in range(k):
            results.append(sources[i]['source'] +  ":" + dialogues[i])
        return json.dumps({
            "user_id": user_id,
            "query": query,
            "results": results,
            "error_code": 0
        })

    def delete(self, params):
        ## 删除用户的对话存储库
        user_id = params.get("user_id")

        if user_id is None:
            return json.dumps({
                "error_code": 1,
                "error_message": "Missing required parameters"
            })

        collection_name = "User_" + str(user_id)
        self.client.delete_collection(name=collection_name)
        logger.info(f"User_{user_id}'s memory_storage has been deleted!")

        return json.dumps({
            "user_id": user_id,
            "message": f" User_{user_id}'s memory storage has been deleted",
            "error_code": 0
        })

    def fetch_messages(self, request: XinHaiFetchMessagesRequest):
        room_id = request.room.roomId
        collection = self.client.get_or_create_collection(name="Room_" + str(room_id),
                                                          embedding_function=self.embedding_fn)
        messages = collection.get(include=['metadatas'])['metadatas']
        return XinHaiFetchMessagesResponse(messages=[XinHaiChatMessage.model_validate_json(m['message']) for m in messages])

    def store_messages(self, request: XinHaiStoreMessagesRequest):
        room_id = request.room.roomId
        collection = self.client.get_or_create_collection(name="Room_" + str(room_id),
                                                          embedding_function=self.embedding_fn)
        res = collection.count()
        ids = [message.indexId for message in request.messages]
        documents = [message.content for message in request.messages]
        collection.add(documents=documents, ids=ids,
                       metadatas=[{"message": m.model_dump_json()} for m in request.messages])

        logger.info(f'Room_{room_id}\'s memory_storage adds a message')
        return json.dumps({
            "status": 200
        })


app = FastAPI()


def release_model_semaphore(fn=None):
    model_semaphore.release()
    if fn is not None:
        fn()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder({"detail": exc.errors(), "body": exc.body}),
    )


@app.post("/worker_storage_insert")
async def insert_data(request: Request):
    params = await request.json()
    return worker.insert_data(params)


@app.post("/worker_storage_get")
async def get_data(request: Request):
    params = await request.json()
    return worker.get_data(params)


@app.post("/worker_storage_search")
async def search_similar(request: Request):
    params = await request.json()
    return worker.search_similar(params)


@app.post("/worker_storage_delete")
async def delete(request: Request):
    params = await request.json()
    return worker.delete(params)


@app.post("/worker_get_status")
async def get_status(request: Request):
    return worker.get_status()


@app.post("/worker_fetch_messages")
async def fetch_messages(request: XinHaiFetchMessagesRequest, response_model=XinHaiFetchMessagesResponse):
    # params = await request.json()
    return worker.fetch_messages(request)


@app.post("/worker_store_messages")
async def store_messages(request: XinHaiStoreMessagesRequest):
    # params = await request.json()
    return worker.store_messages(request)


if __name__ == "__main__":
    worker = StorageWorker()
    uvicorn.run(app, host=WORKER_HOST, port=WORKER_PORT, log_level="info")
