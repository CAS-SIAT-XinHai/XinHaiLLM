"""
A model worker executes the model.
"""
import asyncio
import json
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from threading import Thread
from typing import Sequence, Dict, Optional, List, Generator, AsyncGenerator, Any, Annotated, Tuple

import requests
import uvicorn
from fastapi import FastAPI, HTTPException, status, Depends
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from llamafactory.api.protocol import Role, ModelList, ModelCard, ChatCompletionResponse, ChatCompletionRequest, Function, \
    ChatCompletionMessage, FunctionCall, Finish, ChatCompletionResponseChoice, ChatCompletionResponseUsage, \
    ChatCompletionStreamResponse, ScoreEvaluationResponse, ScoreEvaluationRequest, ChatCompletionStreamResponseChoice
from llamafactory.chat.hf_engine import HuggingfaceEngine
from llamafactory.chat.vllm_engine import VllmEngine
from llamafactory.data import Role as DataRole
from llamafactory.extras.misc import torch_gc
from llamafactory.hparams import get_infer_args
from sse_starlette import EventSourceResponse

from ..config import LOG_DIR, WORKER_HEART_BEAT_INTERVAL
from ..utils import build_logger, pretty_print_semaphore

GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
logger = build_logger("model_worker", f"model_worker_{worker_id}.log", LOG_DIR)
global_counter = 0

CONTROLLER_ADDRESS = os.environ.get("CONTROLLER_ADDRESS")
WORKER_ADDRESS = os.environ.get("WORKER_ADDRESS")
WORKER_HOST = os.environ.get("WORKER_HOST")
WORKER_PORT = int(os.environ.get("WORKER_PORT"))
NO_REGISTER = os.environ.get("NO_REGISTER", False)
MODEL_NAME = os.environ.get("MODEL_NAME", "paddleocr")
LIMIT_MODEL_CONCURRENCY = int(os.environ.get("LIMIT_MODEL_CONCURRENCY", 5))
DEVICE = "cuda"

model_semaphore = None

api_key = os.environ.get("API_KEY", None)
security = HTTPBearer(auto_error=False)


@asynccontextmanager
async def lifespan(app: "FastAPI"):  # collects GPU memory
    yield
    torch_gc()


def dictify(data: "BaseModel") -> Dict[str, Any]:
    try:  # pydantic v2
        return data.model_dump(exclude_unset=True)
    except AttributeError:  # pydantic v1
        return data.dict(exclude_unset=True)


def jsonify(data: "BaseModel") -> str:
    try:  # pydantic v2
        return json.dumps(data.model_dump(exclude_unset=True), ensure_ascii=False)
    except AttributeError:  # pydantic v1
        return data.json(exclude_unset=True, ensure_ascii=False)


def heart_beat_worker(controller):
    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()


def _start_background_loop(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


class LLMWorker:
    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:
        model_args, data_args, finetuning_args, generating_args = get_infer_args(args)
        if model_args.infer_backend == "huggingface":
            self.engine: "BaseEngine" = HuggingfaceEngine(model_args, data_args, finetuning_args, generating_args)
        elif model_args.infer_backend == "vllm":
            self.engine: "BaseEngine" = VllmEngine(model_args, data_args, finetuning_args, generating_args)
        else:
            raise NotImplementedError("Unknown backend: {}".format(model_args.infer_backend))

        self._loop = asyncio.new_event_loop()
        self._thread = Thread(target=_start_background_loop, args=(self._loop,), daemon=True)
        self._thread.start()

        if not NO_REGISTER:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(
                target=heart_beat_worker, args=(self,))
            self.heart_beat_thread.start()

    def chat(
            self,
            messages: Sequence[Dict[str, str]],
            system: Optional[str] = None,
            tools: Optional[str] = None,
            **input_kwargs,
    ) -> List["Response"]:
        task = asyncio.run_coroutine_threadsafe(self.achat(messages, system, tools, **input_kwargs), self._loop)
        return task.result()

    async def achat(
            self,
            messages: Sequence[Dict[str, str]],
            system: Optional[str] = None,
            tools: Optional[str] = None,
            **input_kwargs,
    ) -> List["Response"]:
        return await self.engine.chat(messages, system, tools, **input_kwargs)

    def stream_chat(
            self,
            messages: Sequence[Dict[str, str]],
            system: Optional[str] = None,
            tools: Optional[str] = None,
            **input_kwargs,
    ) -> Generator[str, None, None]:
        generator = self.astream_chat(messages, system, tools, **input_kwargs)
        while True:
            try:
                task = asyncio.run_coroutine_threadsafe(generator.__anext__(), self._loop)
                yield task.result()
            except StopAsyncIteration:
                break

    async def astream_chat(
            self,
            messages: Sequence[Dict[str, str]],
            system: Optional[str] = None,
            tools: Optional[str] = None,
            **input_kwargs,
    ) -> AsyncGenerator[str, None]:
        async for new_token in self.engine.stream_chat(messages, system, tools, **input_kwargs):
            yield new_token

    def get_scores(
            self,
            batch_input: List[str],
            **input_kwargs,
    ) -> List[float]:
        task = asyncio.run_coroutine_threadsafe(self.aget_scores(batch_input, **input_kwargs), self._loop)
        return task.result()

    async def aget_scores(
            self,
            batch_input: List[str],
            **input_kwargs,
    ) -> List[float]:
        return await self.engine.get_scores(batch_input, **input_kwargs)

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

    def add_content(self, prompt, new_content):
        if '[INST]' in prompt:
            split_index = prompt.rfind(' [/INST]')
        elif '<|im_end|>' in prompt:
            split_index = prompt.rfind('<|im_end|>')
        else:
            split_index = prompt.rfind('###Assistant:')
        left_prompt = prompt[:split_index]
        right_prompt = prompt[split_index:]
        prompt = left_prompt + new_content + right_prompt
        return prompt


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROLE_MAPPING = {
    Role.USER: DataRole.USER.value,
    Role.ASSISTANT: DataRole.ASSISTANT.value,
    Role.SYSTEM: DataRole.SYSTEM.value,
    Role.FUNCTION: DataRole.FUNCTION.value,
    Role.TOOL: DataRole.OBSERVATION.value,
}


def _process_request(request: "ChatCompletionRequest") -> Tuple[List[Dict[str, str]], str, str]:
    if len(request.messages) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid length")

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

    return input_messages, system, tools


def _create_stream_chat_completion_chunk(
        completion_id: str,
        model: str,
        delta: "ChatCompletionMessage",
        index: Optional[int] = 0,
        finish_reason: Optional["Finish"] = None,
) -> str:
    choice_data = ChatCompletionStreamResponseChoice(index=index, delta=delta, finish_reason=finish_reason)
    chunk = ChatCompletionStreamResponse(id=completion_id, model=model, choices=[choice_data])
    return jsonify(chunk)


async def create_chat_completion_response(
        request: "ChatCompletionRequest", chat_model: "ChatModel"
) -> "ChatCompletionResponse":
    completion_id = "chatcmpl-{}".format(uuid.uuid4().hex)
    input_messages, system, tools = _process_request(request)
    responses = await chat_model.achat(
        input_messages,
        system,
        tools,
        do_sample=request.do_sample,
        temperature=request.temperature,
        top_p=request.top_p,
        max_new_tokens=request.max_tokens,
        num_return_sequences=request.n,
    )

    prompt_length, response_length = 0, 0
    choices = []
    for i, response in enumerate(responses):
        if tools:
            result = chat_model.engine.template.format_tools.extract(response.response_text)
        else:
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
        request: "ChatCompletionRequest", chat_model: "ChatModel"
) -> AsyncGenerator[str, None]:
    completion_id = "chatcmpl-{}".format(uuid.uuid4().hex)
    input_messages, system, tools = _process_request(request)
    if tools:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot stream function calls.")

    yield _create_stream_chat_completion_chunk(
        completion_id=completion_id, model=request.model, delta=ChatCompletionMessage(role=Role.ASSISTANT, content="")
    )
    async for new_token in chat_model.astream_chat(
            input_messages,
            system,
            tools,
            do_sample=request.do_sample,
            temperature=request.temperature,
            top_p=request.top_p,
            max_new_tokens=request.max_tokens,
    ):
        if len(new_token) != 0:
            yield _create_stream_chat_completion_chunk(
                completion_id=completion_id, model=request.model, delta=ChatCompletionMessage(content=new_token)
            )

    yield _create_stream_chat_completion_chunk(
        completion_id=completion_id, model=request.model, delta=ChatCompletionMessage(), finish_reason=Finish.STOP
    )
    yield "[DONE]"


async def create_score_evaluation_response(
        request: "ScoreEvaluationRequest", chat_model: "ChatModel"
) -> "ScoreEvaluationResponse":
    if len(request.messages) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid request")

    scores = await chat_model.aget_scores(request.messages, max_length=request.max_length)
    return ScoreEvaluationResponse(model=request.model, scores=scores)


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    model_card = ModelCard(id="gpt-3.5-turbo")
    return ModelList(data=[model_card])


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
    if not worker.engine.can_generate:
        raise HTTPException(status_code=status.HTTP_405_METHOD_NOT_ALLOWED, detail="Not allowed")

    if request.stream:
        generate = create_stream_chat_completion_response(request, worker)
        return EventSourceResponse(generate, media_type="text/event-stream")
    else:
        return await create_chat_completion_response(request, worker)


async def stream_chat_completion(
        messages: Sequence[Dict[str, str]], system: str, tools: str, request: ChatCompletionRequest
):
    choice_data = ChatCompletionStreamResponseChoice(
        index=0, delta=ChatCompletionMessage(role=Role.ASSISTANT, content=""), finish_reason=None
    )
    chunk = ChatCompletionStreamResponse(model=request.model, choices=[choice_data])
    yield jsonify(chunk)

    async for new_token in worker.astream_chat(
            messages,
            system,
            tools,
            do_sample=request.do_sample,
            temperature=request.temperature,
            top_p=request.top_p,
            max_new_tokens=request.max_tokens,
    ):
        if len(new_token) == 0:
            continue

        choice_data = ChatCompletionStreamResponseChoice(
            index=0, delta=ChatCompletionMessage(content=new_token), finish_reason=None
        )
        chunk = ChatCompletionStreamResponse(model=request.model, choices=[choice_data])
        yield jsonify(chunk)

    choice_data = ChatCompletionStreamResponseChoice(
        index=0, delta=ChatCompletionMessage(), finish_reason=Finish.STOP
    )
    chunk = ChatCompletionStreamResponse(model=request.model, choices=[choice_data])
    yield jsonify(chunk)
    yield "[DONE]"


def release_model_semaphore(fn=None):
    model_semaphore.release()
    if fn is not None:
        fn()


@app.post("/worker_get_status")
async def get_status(request: Request):
    return worker.get_status()


if __name__ == "__main__":
    worker = LLMWorker()
    uvicorn.run(app, host=WORKER_HOST, port=WORKER_PORT, log_level="info")
