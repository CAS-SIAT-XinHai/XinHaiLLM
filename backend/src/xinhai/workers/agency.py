"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Vimos Tan
"""
import json
import logging
import os
import threading
import time
import uuid
from typing import Optional, Annotated, Tuple, List, Dict

import requests
import uvicorn
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sse_starlette import EventSourceResponse
from starlette import status

from llamafactory.api.chat import ROLE_MAPPING
from llamafactory.api.common import dictify
from llamafactory.api.protocol import ChatCompletionResponse, ChatCompletionRequest, Function, FunctionCall, \
    ChatCompletionMessage, Finish, Role, ChatCompletionResponseChoice, ChatCompletionResponseUsage
from xinhai.arena.simulation import Simulation
from ..config import LOG_DIR, WORKER_HEART_BEAT_INTERVAL
from ..utils import build_logger, pretty_print_semaphore

GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
logger = logging.getLogger(__name__)
# logger = build_logger("model_worker", f"model_worker_{worker_id}.log", LOG_DIR)
global_counter = 0

model_semaphore = None

CONTROLLER_ADDRESS = os.environ.get("CONTROLLER_ADDRESS")
WORKER_ADDRESS = os.environ.get("WORKER_ADDRESS")
WORKER_HOST = os.environ.get("WORKER_HOST")
WORKER_PORT = int(os.environ.get("WORKER_PORT", 40000))
AGENCY_CONFIG_PATH = os.environ.get("AGENCY_CONFIG_PATH")
MODEL_NAME = os.environ.get("MODEL_NAME", "agency")
NO_REGISTER = os.environ.get("NO_REGISTER", False)
DEVICE = "cuda"
LIMIT_MODEL_CONCURRENCY = int(os.environ.get("LIMIT_MODEL_CONCURRENCY", 5))

api_key = os.environ.get("API_KEY", None)
security = HTTPBearer(auto_error=False)

logging.basicConfig(level=logging.DEBUG)
# logger.setLevel(logging.DEBUG)


def heart_beat_worker(controller):
    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()


class AgencyWorker:
    def __init__(self):
        self.simulator = Simulation.from_config(AGENCY_CONFIG_PATH)
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

    async def interact(self, request: "ChatCompletionRequest") -> Tuple[List[Dict[str, str]], str, str]:
        if len(request.messages) == 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid length")

        logging.debug(request)

        if request.messages[0].role == Role.SYSTEM:
            system = request.messages.pop(0).content
        else:
            system = ""

        if len(request.messages) % 2 == 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only supports u/a/u/a/u...")

        input_messages = []
        for i, message in enumerate(request.messages):
            if i % 2 == 0 and message.role not in [Role.USER, Role.TOOL]:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid role")
            elif i % 2 == 1 and message.role not in [Role.ASSISTANT, Role.FUNCTION]:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid role")

            if message.role == Role.ASSISTANT and isinstance(message.tool_calls, list) and len(message.tool_calls):
                name = message.tool_calls[0].function.name
                arguments = message.tool_calls[0].function.arguments
                content = json.dumps({"name": name, "argument": arguments}, ensure_ascii=False)
                input_messages.append({"role": ROLE_MAPPING[Role.FUNCTION], "content": content})
            else:
                input_messages.append({"role": ROLE_MAPPING[message.role], "content": message.content})

        tool_list = request.tools
        if isinstance(tool_list, list) and len(tool_list):
            try:
                tools = json.dumps([dictify(tool.function) for tool in tool_list], ensure_ascii=False)
            except Exception:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid tools")
        else:
            tools = ""

        responses = await self.simulator.environment.step(
            input_messages,
            system,
            tools,
            do_sample=request.do_sample,
            temperature=request.temperature,
            top_p=request.top_p,
            max_new_tokens=request.max_tokens,
            num_return_sequences=request.n,
        )

        logger.debug("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        logger.debug(f"Responses from current step is {responses}")

        return responses


async def create_chat_completion_response(
        request: "ChatCompletionRequest",
):
    completion_id = "chatcmpl-{}".format(uuid.uuid4().hex)
    tools, responses = worker.interact(request)

    prompt_length, response_length = 0, 0
    choices = []
    for i, response in enumerate(responses):
        # if tools:
        #     result = chat_model.engine.template.format_tools.extract(response.response_text)
        # else:
        #     result = response.response_text
        result = response.response_text

        if isinstance(result, tuple):
            name, arguments = result
            function = Function(name=name, arguments=arguments)
            tool_call = FunctionCall(id="call_{}".format(uuid.uuid4().hex), function=function)
            response_message = ChatCompletionMessage(role=Role.ASSISTANT, tool_calls=[tool_call])
            finish_reason = Finish.TOOL
        else:
            response_message = ChatCompletionMessage(role=Role.ASSISTANT, content=result)
            finish_reason = Finish.STOP if response.finish_reason == "stop" else Finish.LENGTH

        choices.append(ChatCompletionResponseChoice(index=i, message=response_message, finish_reason=finish_reason))
        prompt_length = response.prompt_length
        response_length += response.response_length

    usage = ChatCompletionResponseUsage(
        prompt_tokens=prompt_length,
        completion_tokens=response_length,
        total_tokens=prompt_length + response_length,
    )
    return ChatCompletionResponse(id=completion_id, model=request.model, choices=choices, usage=usage)


async def create_stream_chat_completion_response(
        request: "ChatCompletionRequest",
):
    completion_id = "chatcmpl-{}".format(uuid.uuid4().hex)
    response = await worker.interact(request)

    prompt_length, response_length = 0, 0
    choices = []
    # for i, response in enumerate(responses):
    #     # if tools:
    #     #     result = chat_model.engine.template.format_tools.extract(response.response_text)
    #     # else:
    #     #     result = response.response_text
    #     result = response.response_text
    #
    #     if isinstance(result, tuple):
    #         name, arguments = result
    #         function = Function(name=name, arguments=arguments)
    #         tool_call = FunctionCall(id="call_{}".format(uuid.uuid4().hex), function=function)
    #         response_message = ChatCompletionMessage(role=Role.ASSISTANT, tool_calls=[tool_call])
    #         finish_reason = Finish.TOOL
    #     else:
    #         response_message = ChatCompletionMessage(role=Role.ASSISTANT, content=result)
    #         finish_reason = Finish.STOP if response.finish_reason == "stop" else Finish.LENGTH
    #
    #     choices.append(ChatCompletionResponseChoice(index=i, message=response_message, finish_reason=finish_reason))
    #     prompt_length = response.prompt_length
    #     response_length += response.response_length

    yield json.dumps({
        "id": completion_id,
        "choices": [
            {
                "delta": {
                    "content": response.content,
                },
                "finish_reason": None,
                "index": 0
            }
        ],
        "role": Role.ASSISTANT,
        "model": Role.ASSISTANT
    })


app = FastAPI()


def release_model_semaphore(fn=None):
    model_semaphore.release()
    if fn is not None:
        fn()


async def verify_api_key(auth: Annotated[Optional[HTTPAuthorizationCredentials], Depends(security)]):
    if api_key and (auth is None or auth.credentials != api_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key.")


@app.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(verify_api_key)],
)
async def create_chat_completion(request: ChatCompletionRequest):
    # if not worker.engine.can_generate:
    #     raise HTTPException(status_code=status.HTTP_405_METHOD_NOT_ALLOWED, detail="Not allowed")

    if request.stream:
        generate = create_stream_chat_completion_response(request)
        return EventSourceResponse(generate, media_type="text/event-stream")
        # raise NotImplementedError
    else:
        return await create_chat_completion_response(request)


if __name__ == "__main__":
    worker = AgencyWorker()
    uvicorn.run(app, host=WORKER_HOST, port=WORKER_PORT, log_level="info")
