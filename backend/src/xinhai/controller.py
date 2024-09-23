"""
A controller manages distributed workers.
It sends worker addresses to clients.
"""
import argparse
import dataclasses
import json
import os
import re
import threading
import time
from enum import Enum, auto
from typing import List, Union

import aiofiles
import numpy as np
import requests
import uvicorn
from fastapi import FastAPI, Request, status
from fastapi import UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from more_itertools import sliced
from openai import OpenAI, OpenAIError
from sse_starlette import EventSourceResponse

from llamafactory.api.protocol import ChatCompletionResponse, ChatCompletionRequest
from .config import CONTROLLER_HEART_BEAT_EXPIRATION, LOG_DIR, STATIC_PATH
from .types.message import XinHaiChatCompletionRequest
from .utils import build_logger, server_error_msg

from xinhai.types.message import XinHaiMMRequest, XinHaiMMResponse, XinHaiMMResult

logger = build_logger("controller", "controller.log", LOG_DIR)


class DispatchMethod(Enum):
    LOTTERY = auto()
    SHORTEST_QUEUE = auto()

    @classmethod
    def from_str(cls, name):
        if name == "lottery":
            return cls.LOTTERY
        elif name == "shortest_queue":
            return cls.SHORTEST_QUEUE
        else:
            raise ValueError(f"Invalid dispatch method")


@dataclasses.dataclass
class WorkerInfo:
    model_names: List[str]
    speed: int
    queue_length: int
    check_heart_beat: bool
    last_heart_beat: str


def heart_beat_controller(controller):
    while True:
        time.sleep(CONTROLLER_HEART_BEAT_EXPIRATION)
        controller.remove_stable_workers_by_expiration()


class Controller:
    def __init__(self, dispatch_method: str):
        # Dict[str -> WorkerInfo]
        self.worker_info = {}
        self.dispatch_method = DispatchMethod.from_str(dispatch_method)

        self.heart_beat_thread = threading.Thread(
            target=heart_beat_controller, args=(self,))
        self.heart_beat_thread.start()

        logger.info("Init controller")

    def register_worker(self, worker_name: str, check_heart_beat: bool,
                        worker_status: dict):
        if worker_name not in self.worker_info:
            logger.info(f"Register a new worker: {worker_name}")
        else:
            logger.info(f"Register an existing worker: {worker_name}")

        if not worker_status:
            worker_status = self.get_worker_status(worker_name)
        if not worker_status:
            return False

        self.worker_info[worker_name] = WorkerInfo(
            worker_status["model_names"], worker_status["speed"], worker_status["queue_length"],
            check_heart_beat, time.time())

        logger.info(f"Register done: {worker_name}, {worker_status}")
        return True

    def get_worker_status(self, worker_name: str):
        try:
            r = requests.post(worker_name + "/worker_get_status", timeout=5)
        except requests.exceptions.RequestException as e:
            logger.error(f"Get status fails: {worker_name}, {e}")
            return None

        if r.status_code != 200:
            logger.error(f"Get status fails: {worker_name}, {r}")
            return None

        return r.json()

    def remove_worker(self, worker_name: str):
        del self.worker_info[worker_name]

    def refresh_all_workers(self):
        old_info = dict(self.worker_info)
        self.worker_info = {}

        for w_name, w_info in old_info.items():
            if not self.register_worker(w_name, w_info.check_heart_beat, None):
                logger.info(f"Remove stale worker: {w_name}")

    def list_models(self):
        model_names = set()

        for w_name, w_info in self.worker_info.items():
            model_names.update(w_info.model_names)

        return list(model_names)

    def get_worker_address(self, model_name: str):
        if self.dispatch_method == DispatchMethod.LOTTERY:
            worker_names = []
            worker_speeds = []
            for w_name, w_info in self.worker_info.items():
                if model_name in w_info.model_names:
                    worker_names.append(w_name)
                    worker_speeds.append(w_info.speed)
            worker_speeds = np.array(worker_speeds, dtype=np.float32)
            norm = np.sum(worker_speeds)
            if norm < 1e-4:
                return ""
            worker_speeds = worker_speeds / norm
            if True:  # Directly return address
                pt = np.random.choice(np.arange(len(worker_names)),
                                      p=worker_speeds)
                worker_name = worker_names[pt]
                return worker_name

            # Check status before returning
            while True:
                pt = np.random.choice(np.arange(len(worker_names)),
                                      p=worker_speeds)
                worker_name = worker_names[pt]

                if self.get_worker_status(worker_name):
                    break
                else:
                    self.remove_worker(worker_name)
                    worker_speeds[pt] = 0
                    norm = np.sum(worker_speeds)
                    if norm < 1e-4:
                        return ""
                    worker_speeds = worker_speeds / norm
                    continue
            return worker_name
        elif self.dispatch_method == DispatchMethod.SHORTEST_QUEUE:
            worker_names = []
            worker_qlen = []
            for w_name, w_info in self.worker_info.items():
                if model_name in w_info.model_names:
                    worker_names.append(w_name)
                    worker_qlen.append(w_info.queue_length / w_info.speed)
            if len(worker_names) == 0:
                return ""
            min_index = np.argmin(worker_qlen)
            w_name = worker_names[min_index]
            self.worker_info[w_name].queue_length += 1
            logger.info(f"names: {worker_names}, queue_lens: {worker_qlen}, ret: {w_name}")
            return w_name
        else:
            raise ValueError(f"Invalid dispatch method: {self.dispatch_method}")

    def receive_heart_beat(self, worker_name: str, queue_length: int):
        if worker_name not in self.worker_info:
            logger.info(f"Receive unknown heart beat. {worker_name}")
            return False

        self.worker_info[worker_name].queue_length = queue_length
        self.worker_info[worker_name].last_heart_beat = time.time()
        logger.info(f"Receive heart beat. {worker_name}")
        return True

    def remove_stable_workers_by_expiration(self):
        expire = time.time() - CONTROLLER_HEART_BEAT_EXPIRATION
        to_delete = []
        for worker_name, w_info in self.worker_info.items():
            if w_info.check_heart_beat and w_info.last_heart_beat < expire:
                to_delete.append(worker_name)

        for worker_name in to_delete:
            self.remove_worker(worker_name)

    def worker_api_chat_completion_streaming(self, request: Union[XinHaiChatCompletionRequest, ChatCompletionRequest]):
        worker_addr = self.get_worker_address(request.model)
        logger.info(f"Worker {request.model}: {worker_addr} , {request}")
        if not worker_addr:
            logger.info(f"no worker: {request.model}")
            ret = {
                "text": server_error_msg,
                "error_code": 2,            }
            yield json.dumps(ret).encode() + b"\0"

        if isinstance(request, XinHaiChatCompletionRequest):
            messages = request.to_chat(STATIC_PATH)
        else:
            messages = request.messages

        openai_api_key = "EMPTY"  # OPENAI_API_KEY
        openai_api_base = f"{worker_addr}/v1/"

        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

        logger.info(f"Sending messages: {messages}!")

        for response in client.chat.completions.create(
                model=request.model,
                messages=messages,
                stream=True
        ):
            yield response.to_json()

    def worker_api_chat_completion(self, request: Union[XinHaiChatCompletionRequest, ChatCompletionRequest]):
        worker_addr = self.get_worker_address(request.model)
        logger.info(f"Worker {request.model}: {worker_addr} , {request}")
        if not worker_addr:
            logger.info(f"no worker: {request.model}")
            ret = {
                "text": server_error_msg,
                "error_code": 2,
            }
            return ret

        if isinstance(request, XinHaiChatCompletionRequest):
            messages = request.to_chat(STATIC_PATH)
        else:
            messages = request.messages

        openai_api_key = "EMPTY"  # OPENAI_API_KEY
        openai_api_base = f"{worker_addr}/v1/"

        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

        logger.info(f"Sending messages: {messages}!")

        response = client.chat.completions.create(
            model=request.model,
            messages=messages
        )
        return response

    @staticmethod
    def chat_completion(client, model, messages):
        try:
            logger.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            logger.info(f"Sending messages to {model}: {messages}")
            chat_response = client.chat.completions.create(
                model=model,
                messages=messages
            )
            logger.info("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            content = chat_response.choices[0].message.content
            if content.strip():
                logger.info(f"Get response from {model}: {content}")
                return content.strip()
            else:
                usage = chat_response.usage
                logger.error(f"Error response from {model}: {usage}")
        except OpenAIError as e:
            # Handle all OpenAI API errors
            logger.warning("*****************************************")
            logger.warning(f"Error response from {model}: {e}")

    def worker_api_storage_chat(self, params):

        knowledge_worker_addr = self.get_worker_address(params["knowledge"])
        logger.info(f"Worker {params['knowledge']}: {knowledge_worker_addr}")
        if not knowledge_worker_addr:
            logger.info(f"no worker: {params['knowledge']}")
            ret = {
                "text": server_error_msg,
                "error_code": 2,
            }
            return ret
        worker_addr = "https://api.siliconflow.cn"
        # worker_addr = self.get_worker_address(params["model"])
        openai_api_key = ""  # OPENAI_API_KEY
        openai_api_base = f"{worker_addr}/v1/"
        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base
        )
        # content is the psychology test questions
        content = params["question"]
        try:
            r = requests.post(knowledge_worker_addr + "/worker_rag_storage",
                              json={
                                  "user_query": content,
                                  "top_k": 5
                              },
                              timeout=60)
        except requests.exceptions.RequestException as e:
            logger.error(f"Get status fails: {knowledge_worker_addr}, {e}")
            return None

        if r.status_code != 200:
            logger.error(f"Get status fails: {knowledge_worker_addr}, {r}")
            return None

        knowledge_data = r.json()
        if isinstance(knowledge_data, str):
            knowledge_data = json.loads(knowledge_data)

        logger.info(f"Get response from [knowledge]: {knowledge_data}")

        rewriting_prompt = ("##任务\n你是一位心理学领域的资深专家，你需要为给出的心理学测评题目选择最符合心理学原理的选项。\n\n"
                            #"##示例题目及回答\n如果以下给出的示例题目及回答对你的本次回答具有帮助，你可以将它们作为参考\n{history}\n\n"
                            "##可供参考的知识和经验\n如果以下给出的知识和经验对你的本次回答具有帮助，你可以将它们作为参考\n{ProEK}\n\n"
                            "##需要你回答的心理学题目\n{query}\n\n"
                            "##注意\n请以JSON格式返回你的答案，不要添加任何分析和其他内容。你的输出仅限于候选选项的字母，例如：{{'ans':'A'}}\n\n你的答案是：")
        # for llama use
        # rewriting_prompt =("## Task\nYou are a seasoned expert in the field of psychology. Your task is to select the option that best aligns with psychological principles for the given psychology assessment question.\n\n"
        #                    "## Sample Question and Answer\nIf the sample questions and answers provided below are helpful for your current response, you may refer to them as a reference.\n{history}\n\n"
        #                    # "## Knowledge and Experience for Reference\nIf the knowledge and experience provided below are helpful for your current response, you may refer to them as a reference.\n{ProEK}\n\n"
        #                    "## Psychology Question You Need to Answer\n{query}\n\n"
        #                    "## Note\nPlease return your answer in JSON format without adding any analysis or other content. Your output should be limited to the letter of the chosen option, for example: {{'ans':'A'}}\n\nYour answer is:")

        answer_form = '```json{"ans": "(options)"}```'

        history = []

        for m in params["messages"]:
            history.append(f"{m['role']}: {m['content']}")

        messages = [{
            "role": "user",
            "content": rewriting_prompt.format(
                history="\n".join(history),
                query=content,
                ProEK=knowledge_data["topk_chunks"],
            ) + answer_form,
        }]

        while True:
            logger.info(f"Sending messages to worker: {messages}")
            content = self.chat_completion(client, model=params['model'], messages=messages)
            if content:
                try:
                    json_string = re.search(r"\s*(\{.*?\})\s*", content, re.DOTALL).group(1)
                    rr = json.loads(json_string)
                    if rr["ans"]:
                        logger.info(f"Returning response from worker: {rr}")
                        return rr
                except Exception as e:
                    logger.error(f"Returning response error:{content}")

    def worker_api_rag_chat(self, params):

        knowledge_worker_addr = self.get_worker_address(params["knowledge"])
        logger.info(f"Worker {params['knowledge']}: {knowledge_worker_addr}")
        if not knowledge_worker_addr:
            logger.info(f"no worker: {params['knowledge']}")
            ret = {
                "text": server_error_msg,
                "error_code": 2,
            }
            return ret

        worker_addr = self.get_worker_address(params["model"])
        openai_api_key = "EMPTY"  # OPENAI_API_KEY
        openai_api_base = f"{worker_addr}/v1/"
        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        messages = []
        for m in params["messages"]:
            messages.append({
                "role": m['role'],
                "content": m['content'],
            })
        content = self.chat_completion(client, model=params['model'], messages=messages)

        try:
            r = requests.post(knowledge_worker_addr + "/worker_rag_query",
                              json={
                                  "user_query": content
                              },
                              timeout=60)
        except requests.exceptions.RequestException as e:
            logger.error(f"Get status fails: {knowledge_worker_addr}, {e}")
            return None

        if r.status_code != 200:
            logger.error(f"Get status fails: {knowledge_worker_addr}, {r}")
            return None

        knowledge_data = r.json()
        if isinstance(knowledge_data, str):
            knowledge_data = json.loads(knowledge_data)

        logger.info(f"Get response from [knowledge]: {knowledge_data}")

        rewriting_prompt = ("你是一名精通心理学知识的专家。请你基于下述用户的描述，对用户描述分析后，利用下面专业心理学知识和社科类知识对原回答进行回答改写，以提高回答的专业性和知识性，并增加情感上的支持。\n"
                            "1. 以表达善意或爱意或安慰的话开头，以提供情感上的支持，回答过程中应始终保持尊重、热情、真诚、共情、积极关注的态度。\n"
                            "2. 标记出并保留专业领域词汇，如心理学专业术语，实验、包括人名、书籍名、调查数据来源、引用来源，专业心理名词，专业词语解释等，优先使用这些词汇附近的知识。\n"
                            "3. 在补充知识过程中，确保知识和问题及描述的相关性，尽可能多地补充的相关知识，禁止出现与用户描述现象无关的内容，可适当参考原回答内容。\n"
                            "4. 请确保按照信息的先后顺序、逻辑关系和相关性组织文本，同时在回答中添加适当过渡句帮助读者更好理解内容之间的关系和转变。\n"
                            "5. 请你尽可能生成长文本,用中文返回,内容应该知识丰富完整且有深度。\n"
                            "6. 仅返回最终的回答，不要出现其他内容。\n\n"
                            "用户的描述: {history} \n\n"
                            "原回答：{query} \n\n"
                            "专业心理学知识：{ProEK} \n\n"
                            "社科类知识：{SSEK} \n\n"
                            "最终回答是：")
        answer_form = "The generated response should be enclosed by [Response] and [End of Response]."

        history = []
        for m in params["messages"]:
            history.append(f"{m['role']}: {m['content']}")

        messages = [{
            "role": "user",
            "content": rewriting_prompt.format(
                history="\n".join(history),
                query=content,
                ProEK=knowledge_data["rag_pro_knowledge_1"],
                SSEK=knowledge_data["rag_ss_knowledge_1"],
            ) + "\n\n" + answer_form,
        }]

        rag_response_pattern = re.compile(r"\[Response]([\s\S]+)\[End of Response]")

        while True:
            logger.info(f"Sending messages to worker: {messages}")
            content = self.chat_completion(client, model=params['model'], messages=messages)
            if content:
                rr = rag_response_pattern.findall(content)
                if rr:
                    logger.info(f"Returning response from worker: {rr[0]}")
                    return {
                        "text": rr[0]
                    }

    def worker_api_rag_streaming(self, params):

        knowledge_worker_addr = self.get_worker_address(params["knowledge"])
        logger.info(f"Worker {params['knowledge']}: {knowledge_worker_addr}")
        if not knowledge_worker_addr:
            logger.info(f"no worker: {params['knowledge']}")
            ret = {
                "text": server_error_msg,
                "error_code": 2,
            }
            yield json.dumps(ret).encode() + b"\0"

        worker_addr = self.get_worker_address(params["model"])
        openai_api_key = "EMPTY"  # OPENAI_API_KEY
        openai_api_base = f"{worker_addr}/v1/"
        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        messages = []
        for m in params["messages"]:
            messages.append({
                "role": m['role'],
                "content": m['content'],
            })
        # content = self.chat_completion(client, model=params['model'], messages=messages)
        logger.info(f"Sending messages: {messages}!")
        content = ''
        while not content:
            for response in client.chat.completions.create(
                    model=params["model"],
                    messages=messages,
                    stream=True
            ):
                if isinstance(response.choices[0].delta.content, str):
                    content += response.choices[0].delta.content
                    d_str = response.to_json()
                    d = json.loads(d_str)
                    d['role'] = 'before_rag'
                    yield json.dumps(d)

        try:
            r = requests.post(knowledge_worker_addr + "/worker_rag_query",
                              json={
                                  "user_query": content
                              },
                              timeout=60)
        except requests.exceptions.RequestException as e:
            logger.error(f"Get status fails: {knowledge_worker_addr}, {e}")
            return None

        if r.status_code != 200:
            logger.error(f"Get status fails: {knowledge_worker_addr}, {r}")
            return None

        knowledge_data = r.json()

        yield json.dumps({
            "id": "chatcmpl-1a7b99803eed477f8187c92029807f52",
            "choices": [
                {
                    "delta": {
                        "content": knowledge_data
                    },
                    "finish_reason": None,
                    "index": 0
                }
            ],
            "role": params["knowledge"],
            "model": params['model']
        })

        if isinstance(knowledge_data, str):
            knowledge_data = json.loads(knowledge_data)

        logger.info(f"Get response from [knowledge]: {knowledge_data}")

        rewriting_prompt = ("你是一名精通心理学知识的专家。请你基于下述用户的描述，对用户描述分析后，利用下面专业心理学知识和社科类知识对原回答进行回答改写，以提高回答的专业性和知识性，并增加情感上的支持。\n"
                            "1. 以表达善意或爱意或安慰的话开头，以提供情感上的支持，回答过程中应始终保持尊重、热情、真诚、共情、积极关注的态度。\n"
                            "2. 标记出并保留专业领域词汇，如心理学专业术语，实验、包括人名、书籍名、调查数据来源、引用来源，专业心理名词，专业词语解释等，优先使用这些词汇附近的知识。\n"
                            "3. 在补充知识过程中，确保知识和问题及描述的相关性，尽可能多地补充的相关知识，禁止出现与用户描述现象无关的内容，可适当参考原回答内容。\n"
                            "4. 请确保按照信息的先后顺序、逻辑关系和相关性组织文本，同时在回答中添加适当过渡句帮助读者更好理解内容之间的关系和转变。\n"
                            "5. 请你尽可能生成长文本,用中文返回,内容应该知识丰富完整且有深度。\n"
                            "6. 仅返回最终的回答，不要出现其他内容。\n\n"
                            "用户的描述: {history} \n\n"
                            "原回答：{query} \n\n"
                            "专业心理学知识：{ProEK} \n\n"
                            "社科类知识：{SSEK} \n\n"
                            "最终回答是：")
        # answer_form = "The generated response should be enclosed by [Response] and [End of Response]."

        history = []
        for m in params["messages"]:
            history.append(f"{m['role']}: {m['content']}")

        messages = [{
            "role": "user",
            "content": rewriting_prompt.format(
                history="\n".join(history),
                query=content,
                ProEK=knowledge_data["rag_pro_knowledge_1"],
                SSEK=knowledge_data["rag_ss_knowledge_1"],
            ),
        }]

        logger.info(f"Sending messages to worker: {messages}")
        for response in client.chat.completions.create(
                model=params["model"],
                messages=messages,
                stream=True
        ):
            if isinstance(response.choices[0].delta.content, str):
                content += response.choices[0].delta.content
                d_str = response.to_json()
                d = json.loads(d_str)
                d['role'] = 'assistant'
                yield json.dumps(d)

    def worker_api_generate_gists(self, params):
        worker_addr = self.get_worker_address(params["model"])
        logger.info(f"Worker {params['model']}: {worker_addr}")
        if not worker_addr:
            logger.info(f"no worker: {params['model']}")
            ret = {
                "text": server_error_msg,
                "error_code": 2,
            }
            yield json.dumps(ret).encode() + b"\0"

        openai_api_key = "EMPTY"  # OPENAI_API_KEY
        openai_api_base = f"{worker_addr}/v1/"

        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

        for content in sliced(params["content"], n=4096):
            messages = [
                {
                    "role": "user",
                    "content": params["prompt"].format(content=content),
                }
            ]

            logger.info(f"Sending messages: {messages}!")

            for response in client.chat.completions.create(
                    model=params["model"],
                    messages=messages,
                    stream=True
            ):
                yield response.to_json()

    def worker_api_audit_gists(self, params):
        worker_addr = self.get_worker_address(params["model"])
        logger.info(f"Worker {params['model']}: {worker_addr}")
        if not worker_addr:
            logger.info(f"no worker: {params['model']}")
            ret = {
                "text": server_error_msg,
                "error_code": 2,
            }
            yield json.dumps(ret).encode() + b"\0"

        openai_api_key = "EMPTY"  # OPENAI_API_KEY
        openai_api_base = f"{worker_addr}/v1/"

        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

        for i, gist in enumerate(params["gists"]):
            messages = [
                {
                    "role": "user",
                    "content": params["prompt"].format(gist=gist["title"] + gist["description"],
                                                       invoice_content=params['invoice']),
                }
            ]

            logger.info(f"Sending messages: {messages}!")

            for response in client.chat.completions.create(
                    model=params["model"],
                    messages=messages,
                    stream=True
            ):
                d_str = response.to_json()
                d = json.loads(d_str)
                d['gist_id'] = i
                yield json.dumps(d)

    def worker_api_audit_attachments(self, params):
        worker_addr = self.get_worker_address(params["model"])
        logger.info(f"Worker {params['model']}: {worker_addr}")
        if not worker_addr:
            logger.info(f"no worker: {params['model']}")
            ret = {
                "text": server_error_msg,
                "error_code": 2,
            }
            yield json.dumps(ret).encode() + b"\0"

        openai_api_key = "EMPTY"  # OPENAI_API_KEY
        openai_api_base = f"{worker_addr}/v1/"

        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

        attachments = "\n".join(
            [f"[START OF ATTACHMENT]\n{gist['title']}\n{gist['description']}\n[END OF ATTACHMENT]\n" for i, gist in
             enumerate(params["attachments"])])

        for content in sliced(attachments, n=2500 - len(params['invoice'])):
            messages = [
                {
                    "role": "user",
                    "content": params["prompt"].format(attachments=content, invoice_content=params['invoice']),
                }
            ]

            logger.info(f"Sending messages: {messages}!")

            for response in client.chat.completions.create(
                    model=params["model"],
                    messages=messages,
                    stream=True
            ):
                yield response.to_json()

    def worker_api_fetch_messages(self, worker, params):
        storage_worker_addr = self.get_worker_address(worker)
        logger.info(f"Worker {worker}: {storage_worker_addr}")
        if not storage_worker_addr:
            logger.info(f"no worker: {worker}")
            ret = {
                "text": server_error_msg,
                "error_code": 2,
            }
            return ret

        try:
            r = requests.post(storage_worker_addr + "/worker_fetch_messages",
                              json=params,
                              timeout=60)
        except requests.exceptions.RequestException as e:
            logger.error(f"Get status fails: {storage_worker_addr}, {e}")
            return None

        if r.status_code != 200:
            logger.error(f"Get status fails: {storage_worker_addr}, {r}")
            return None

        return r.json()

    def worker_api_store_messages(self, worker, params):
        storage_worker_addr = self.get_worker_address(worker)
        logger.info(f"Worker {worker}: {storage_worker_addr}")
        if not storage_worker_addr:
            logger.info(f"no worker: {worker}")
            ret = {
                "text": server_error_msg,
                "error_code": 2,
            }
            return ret

        try:
            r = requests.post(storage_worker_addr + "/worker_store_messages",
                              json=params,
                              timeout=60)
        except requests.exceptions.RequestException as e:
            logger.error(f"Get status fails: {storage_worker_addr}, {e}")
            return None

        if r.status_code != 200:
            logger.error(f"Get status fails: {storage_worker_addr}, {r}")
            return None

        return r.json()

    def worker_api_storage_store_memory(self, worker, params):
        storage_worker_addr = self.get_worker_address(worker)
        logger.info(f"Worker {worker}: {storage_worker_addr}")
        if not storage_worker_addr:
            logger.info(f"no worker: {worker}")
            ret = {
                "text": server_error_msg,
                "error_code": 2,
            }
            return ret

        try:
            r = requests.post(storage_worker_addr + "/worker_store_memory",
                              json=params,
                              timeout=60)
        except requests.exceptions.RequestException as e:
            logger.error(f"Get status fails: {storage_worker_addr}, {e}")
            return None

        if r.status_code != 200:
            logger.error(f"Get status fails: {storage_worker_addr}, {r}")
            return None

        information_data = r.json()
        if isinstance(information_data, str):
            information_data = json.loads(information_data)

        logger.info(f"Get response from [storage]: {information_data}")

        return information_data

    def worker_api_storage_fetch_memory(self, worker, params):
        storage_worker_addr = self.get_worker_address(worker)
        logger.info(f"Worker {worker}: {storage_worker_addr}")
        if not storage_worker_addr:
            logger.info(f"no worker: {worker}")
            ret = {
                "text": server_error_msg,
                "error_code": 2,
            }
            return ret

        try:
            r = requests.post(storage_worker_addr + "/worker_fetch_memory",
                              json=params,
                              timeout=60)
            if r.status_code != 200:
                logger.error(f"Get status fails: {storage_worker_addr}, {r}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Get status fails: {storage_worker_addr}, {e}")

        return r.json()

    def worker_api_search_chat(self, worker, params):
        storage_worker_addr = self.get_worker_address(worker)
        logger.info(f"Worker {worker}: {storage_worker_addr}")
        if not storage_worker_addr:
            logger.info(f"no worker: {worker}")
            ret = {
                "text": server_error_msg,
                "error_code": 2,
            }
            return ret
        try:
            r = requests.post(storage_worker_addr + "/worker_storage_search",
                              json=params,
                              timeout=60)
        except requests.exceptions.RequestException as e:
            logger.error(f"Get status fails: {storage_worker_addr}, {e}")
            return None

        if r.status_code != 200:
            logger.error(f"Get status fails: {storage_worker_addr}, {r}")
            return None

        search_data = r.json()
        if isinstance(search_data, str):
            search_data = json.loads(search_data)

        logger.info(f"Get response from [storage]: {search_data}")

        return search_data

    def worker_api_ocr_image(self, params):
        worker_addr = self.get_worker_address(params["model"])
        try:
            r = requests.post(worker_addr + "/worker_ocr_image", json=params, timeout=60)
        except requests.exceptions.RequestException as e:
            logger.error(f"Get status fails: {worker_addr}, {e}")
            return None

        if r.status_code != 200:
            logger.error(f"Get status fails: {worker_addr}, {r}")
            return None

        return r.json()

    def worker_api_parse_file(self, params):
        worker_addr = self.get_worker_address(params["model"])
        try:
            r = requests.post(worker_addr + "/worker_parse_file", json=params, timeout=60)
        except requests.exceptions.RequestException as e:
            logger.error(f"Get status fails: {worker_addr}, {e}")
            return None

        if r.status_code != 200:
            logger.error(f"Get status fails: {worker_addr}, {r}")
            return None

        return r.json()

    # Let the controller act as a worker to achieve hierarchical
    # management. This can be used to connect isolated sub networks.
    def worker_api_get_status(self):
        model_names = set()
        speed = 0
        queue_length = 0

        for w_name in self.worker_info:
            worker_status = self.get_worker_status(w_name)
            if worker_status is not None:
                model_names.update(worker_status["model_names"])
                speed += worker_status["speed"]
                queue_length += worker_status["queue_length"]

        return {
            "model_names": list(model_names),
            "speed": speed,
            "queue_length": queue_length,
        }

    def worker_api_mllm_chat(self, params):
        worker_addr = self.get_worker_address(params["model"])
        try:
            r = requests.post(worker_addr + "/worker_mllm_chat", json=params, timeout=60)
        except requests.exceptions.RequestException as e:
            logger.error(f"Get status fails: {worker_addr}, {e}")
            return None

        if r.status_code != 200:
            logger.error(f"Get status fails: {worker_addr}, {r}")
            return None

        return r.json()

    def worker_api_query_search(self, worker, params):
        worker_addr = self.get_worker_address(worker)
        logger.info(f"Worker {worker}: {worker_addr}")
        if not worker_addr:
            logger.info(f"no worker: {worker}")
            ret = {
                "text": server_error_msg,
                "error_code": 2,
            }
            return ret

        try:
            r = requests.post(worker_addr + "/worker_rag_query",
                              json=params,
                              timeout=60)
        except requests.exceptions.RequestException as e:
            logger.error(f"Get status fails: {worker_addr}, {e}")
            return None

        if r.status_code != 200:
            logger.error(f"Get status fails: {worker_addr}, {r}")
            return None

        knowledge_data = r.json()
        if isinstance(knowledge_data, str):
            knowledge_data = json.loads(knowledge_data)

        logger.info(f"Get response from [{worker}]: {knowledge_data}")
        return knowledge_data

    def worker_api_query_search_meta(self, worker, params):
        worker_addr = self.get_worker_address(worker)
        logger.info(f"Worker {worker}: {worker_addr}")
        if not worker_addr:
            logger.info(f"no worker: {worker}")
            ret = {
                "text": server_error_msg,
                "error_code": 2,
            }
            return ret

        try:
            r = requests.post(worker_addr + "/worker_rag_query_meta",
                              json=params,
                              timeout=60)
        except requests.exceptions.RequestException as e:
            logger.error(f"Get status fails: {worker_addr}, {e}")
            return None

        if r.status_code != 200:
            logger.error(f"Get status fails: {worker_addr}, {r}")
            return None

        knowledge_data = r.json()
        if isinstance(knowledge_data, str):
            knowledge_data = json.loads(knowledge_data)

        logger.info(f"Get response from [{worker}]: {knowledge_data}")
        return knowledge_data

    def worker_api_storage_recall_memory(self, worker, params):
        storage_worker_addr = self.get_worker_address(worker)
        logger.info(f"Worker {worker}: {storage_worker_addr}")
        if not storage_worker_addr:
            logger.info(f"no worker: {worker}")
            ret = {
                "text": server_error_msg,
                "error_code": 2,
            }
            return ret

        try:
            r = requests.post(storage_worker_addr + "/worker_recall_memory",
                              json=params,
                              timeout=60)
            if r.status_code != 200:
                logger.error(f"Get status fails: {storage_worker_addr}, {r}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Get status fails: {storage_worker_addr}, {e}")

        return r.json()

    def worker_api_storage_delete_memory(self, worker, params):
        storage_worker_addr = self.get_worker_address(worker)
        logger.info(f"Worker {worker}: {storage_worker_addr}")
        if not storage_worker_addr:
            logger.info(f"no worker: {worker}")
            ret = {
                "text": server_error_msg,
                "error_code": 2,
            }
            return ret

        try:
            r = requests.post(storage_worker_addr + "/worker_delete_memory",
                              json=params,
                              timeout=60)
            if r.status_code != 200:
                logger.error(f"Get status fails: {storage_worker_addr}, {r}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Get status fails: {storage_worker_addr}, {e}")

        logger.error(r.text)
        return r.json()
    
    def mock_worker_api_MM_OCR(self, request:XinHaiMMRequest) -> XinHaiMMResponse:

        prompts = request.prompts
        print(prompts)
        default_results = [XinHaiMMResult(
            name=pr.name,
            value='default'
        ) for pr in prompts]

        response_data = XinHaiMMResponse(
            id=request.id,
            type=request.type,
            result=default_results,
            version=request.version,
            model=request.model,
        )

        return response_data

        


app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_PATH), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder({"detail": exc.errors(), "body": exc.body}),
    )


@app.post("/register_worker")
async def register_worker(request: Request):
    data = await request.json()
    controller.register_worker(
        data["worker_name"], data["check_heart_beat"],
        data.get("worker_status", None))


@app.post("/api/refresh_all_workers")
async def refresh_all_workers():
    models = controller.refresh_all_workers()


@app.post("/api/list_models")
async def list_models():
    models = controller.list_models()
    return {"models": models}


@app.post("/get_worker_address")
async def get_worker_address(request: Request):
    data = await request.json()
    addr = controller.get_worker_address(data["model"])
    return {"address": addr}


@app.post("/receive_heart_beat")
async def receive_heart_beat(request: Request):
    data = await request.json()
    exist = controller.receive_heart_beat(
        data["worker_name"], data["queue_length"])
    return {"exist": exist}


@app.post("/worker_get_status")
async def worker_api_get_status(request: Request):
    return controller.worker_api_get_status()


ALLOWED_FILETYPES = ["image/png", "image/jpg", "image/jpeg"]


@app.post("/api/ocr-image")
async def worker_api_ocr_image(request: Request):
    params = await request.json()
    return controller.worker_api_ocr_image(params)


@app.post("/api/mllm-chat")
async def worker_api_mllm_chat(request: Request):
    params = await request.json()
    return controller.worker_api_mllm_chat(params)


@app.post("/api/upload-image")
@app.post("/api/upload-file")
async def worker_api_upload_file(file: UploadFile):
    out_file_path = os.path.join(STATIC_PATH, file.filename.split(os.path.sep)[-1])
    async with aiofiles.open(out_file_path, 'wb') as out_file:
        while content := await file.read(1024):  # async read chunk
            await out_file.write(content)  # async write chunk

    return {"Result": out_file_path}


@app.post("/api/parse-file")
async def worker_api_parse_file(request: Request):
    params = await request.json()
    return controller.worker_api_parse_file(params)


@app.post("/api/generate-gists")
async def worker_api_generate_gists(request: Request):
    params = await request.json()
    generator = controller.worker_api_generate_gists(params)
    return StreamingResponse(generator)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder({"detail": exc.errors(), "body": exc.body}),
    )


@app.post("/api/chat-completion")
async def worker_api_chat_completion(request: XinHaiChatCompletionRequest):
    # params = await request.json()
    generator = controller.worker_api_chat_completion_streaming(request)
    return StreamingResponse(generator)


@app.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    status_code=status.HTTP_200_OK,
)
async def create_chat_completion(request: ChatCompletionRequest):
    if request.stream:
        generator = controller.worker_api_chat_completion_streaming(request)
        return EventSourceResponse(generator, media_type="text/event-stream")
    else:
        return controller.worker_api_chat_completion(request)


@app.post("/api/rag-chat-completion")
async def worker_api_rag_chat(request: Request):
    params = await request.json()
    return controller.worker_api_rag_chat(params)


@app.post("/api/rag-chat-streaming")
async def worker_api_rag_streaming(request: Request):
    params = await request.json()
    generator = controller.worker_api_rag_streaming(params)
    return StreamingResponse(generator)


@app.post("/api/storage-chat-completion")
async def worker_api_storage_chat(request: Request):
    params = await request.json()
    return controller.worker_api_storage_chat(params)


@app.post("/api/audit-gists")
async def worker_api_audit_gists(request: Request):
    params = await request.json()
    generator = controller.worker_api_audit_gists(params)
    return StreamingResponse(generator)


@app.post("/api/audit-attachments")
async def worker_api_audit_gists(request: Request):
    params = await request.json()
    generator = controller.worker_api_audit_attachments(params)
    return StreamingResponse(generator)


@app.post("/api/{worker}/store-memory")
async def worker_api_storage_store_memory(worker: str, request: Request):
    params = await request.json()
    return controller.worker_api_storage_store_memory(worker, params)


@app.post("/api/{worker}/fetch-memory")
async def worker_api_storage_fetch_memory(worker: str, request: Request):
    params = await request.json()
    return controller.worker_api_storage_fetch_memory(worker, params)


@app.post("/api/{worker}/recall-memory")
async def worker_api_storage_recall_memory(worker: str, request: Request):
    params = await request.json()
    return controller.worker_api_storage_recall_memory(worker, params)


@app.post("/api/{worker}/delete-memory")
async def worker_api_storage_delete_memory(worker: str, request: Request):
    params = await request.json()
    return controller.worker_api_storage_delete_memory(worker, params)


@app.post("/api/{worker}/store-messages")
async def worker_api_store_messages(worker: str, request: Request):
    params = await request.json()
    return controller.worker_api_store_messages(worker, params)


@app.post("/api/{worker}/fetch-messages")
async def worker_api_fetch_messages(worker: str, request: Request):
    params = await request.json()
    return controller.worker_api_fetch_messages(worker, params)


@app.post("/api/{worker}/query-search")
async def worker_api_query_search(worker: str, request: Request):
    params = await request.json()
    return controller.worker_api_query_search(worker, params)


@app.post("/api/{worker}/query-search-meta")
async def worker_api_query_search_meta(worker: str, request: Request):
    params = await request.json()
    return controller.worker_api_query_search_meta(worker, params)


@app.post("/api/{worker}/chat-search")
async def worker_api_search_chat(worker: str, request: Request):
    params = await request.json()
    return controller.worker_api_search_chat(worker, params)


@app.post("/api/MM_OCR")
async def worker_api_MM_OCR(request: XinHaiMMRequest):
    return controller.mock_worker_api_MM_OCR(request)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21001)
    parser.add_argument("--dispatch-method", type=str, choices=[
        "lottery", "shortest_queue"], default="shortest_queue")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    controller = Controller(args.dispatch_method)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
