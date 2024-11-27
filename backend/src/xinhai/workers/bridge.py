"""
A model worker executes the model.
"""
import asyncio
import base64
import io
import json
import os
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from threading import Thread
from typing import Sequence, Dict, Optional, List, Generator, AsyncGenerator, Any, Tuple

import requests
import uvicorn
from PIL import Image
from fastapi import FastAPI, HTTPException, status
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from openai import OpenAI
from sse_starlette import EventSourceResponse

try:
    from urllib.parse import urlparse, ParseResult
except ImportError:
    from urlparse import urlparse, ParseResult

from xinhai.types.message import Role, ROLE_MAPPING, ChatCompletionResponse, ChatCompletionRequest
from ..config import LOG_DIR, WORKER_HEART_BEAT_INTERVAL
from ..utils import build_logger, pretty_print_semaphore, torch_gc

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
API_KEY = os.environ.get("API_KEY")
API_BASE = os.environ.get("API_BASE")
MLLM_LIMIT_MM_PER_PROMPT = int(os.environ.get("MLLM_LIMIT_MM_PER_PROMPT"))
DEVICE = "cuda"

model_semaphore = None

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


class BridgeWorker:
    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:
        self.client = OpenAI(
            api_key=API_KEY,
            base_url=API_BASE,
        )

        self._loop = asyncio.new_event_loop()
        self._thread = Thread(target=_start_background_loop, args=(self._loop,), daemon=True)
        self._thread.start()

        if not NO_REGISTER:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(
                target=heart_beat_worker, args=(self,))
            self.heart_beat_thread.start()

    def stream_chat(
            self,
            messages: Sequence[Dict[str, str]],
            system: Optional[str] = None,
            tools: Optional[str] = None,
            **input_kwargs,
    ) -> Generator[str, None, None]:
        for response in self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                stream=True
        ):
            yield response.to_json()

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


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _process_request(
        request: "ChatCompletionRequest",
) -> Tuple[List[Dict[str, str]], Optional[str], Optional[str], Optional[List["ImageInput"]]]:
    logger.info(f"==== request ====\n{json.dumps(dictify(request), indent=2, ensure_ascii=False)}")

    if len(request.messages) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid length")

    if request.messages[0].role == Role.SYSTEM:
        system = request.messages.pop(0).content
    else:
        system = None

    if len(request.messages) % 2 == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only supports u/a/u/a/u...")

    input_messages = []
    images = []
    for i, message in enumerate(request.messages):
        if i % 2 == 0 and message.role not in [Role.USER, Role.TOOL]:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid role")
        elif i % 2 == 1 and message.role not in [Role.ASSISTANT, Role.FUNCTION]:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid role")

        if message.role == Role.ASSISTANT and isinstance(message.tool_calls, list) and len(message.tool_calls):
            tool_calls = [
                {"name": tool_call.function.name, "arguments": tool_call.function.arguments}
                for tool_call in message.tool_calls
            ]
            content = json.dumps(tool_calls, ensure_ascii=False)
            input_messages.append({"role": ROLE_MAPPING[Role.FUNCTION], "content": content})
        elif isinstance(message.content, list):
            content = []
            for input_item in message.content:
                if input_item.type == "text":
                    content.append(input_item)
                else:
                    image_url = input_item.image_url.url
                    if re.match(r"^data:image\/(png|jpg|jpeg|gif|bmp);base64,(.+)$", image_url):  # base64 image
                        image_stream = io.BytesIO(base64.b64decode(image_url.split(",", maxsplit=1)[1]))
                    elif os.path.isfile(image_url):  # local file
                        image_stream = open(image_url, "rb")
                    else:  # web uri
                        if image_url.startswith("blob:"):
                            image_url = image_url[7:]
                        parsed_url = urlparse(image_url)
                        updated_url = f"{CONTROLLER_ADDRESS}/{parsed_url.path}"
                        image_stream = requests.get(updated_url, stream=True).raw

                    buf = io.BytesIO(image_stream.read())
                    images.append(Image.open(buf).convert("RGB"))
                    img_b64_str = base64.b64encode(buf.getvalue()).decode()
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64_str}"}})
            input_messages.append({"role": ROLE_MAPPING[message.role], "content": content})
        else:
            input_messages.append({"role": ROLE_MAPPING[message.role], "content": message.content})

    if len(images) > MLLM_LIMIT_MM_PER_PROMPT:
        start = 0
        count = 0
        for j, m in enumerate(input_messages):
            if isinstance(m['content'], list):
                count += 1
                if count == len(images) - MLLM_LIMIT_MM_PER_PROMPT:
                    start = j
        input_messages = input_messages[start + 2:]

    tool_list = request.tools
    if isinstance(tool_list, list) and len(tool_list):
        try:
            tools = json.dumps([dictify(tool.function) for tool in tool_list], ensure_ascii=False)
        except json.JSONDecodeError:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid tools")
    else:
        tools = None

    return input_messages


async def create_chat_completion_response(
        request: "ChatCompletionRequest", chat_model
) -> "ChatCompletionResponse":
    input_messages = _process_request(request)
    response = chat_model.client.chat.completions.create(
        model=MODEL_NAME,
        messages=input_messages,
        stream=False
    )
    logger.info(f"Getting response: {response}")
    return ChatCompletionResponse.model_validate_json(response.to_json())


async def create_stream_chat_completion_response(
        request: "ChatCompletionRequest", chat_model
) -> AsyncGenerator[str, None]:
    input_messages = _process_request(request)
    # if tools:
    #     raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot stream function calls.")

    for new_token in chat_model.stream_chat(
            input_messages,
            # system,
            # tools,
            do_sample=request.do_sample,
            temperature=request.temperature,
            top_p=request.top_p,
            max_new_tokens=request.max_tokens,
    ):
        yield new_token


# async def verify_api_key(auth: Annotated[Optional[HTTPAuthorizationCredentials], Depends(security)]):
#     if API_KEY and (auth is None or auth.credentials != API_KEY):
#         raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key.")


@app.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    status_code=status.HTTP_200_OK,
    # dependencies=[Depends(verify_api_key)],
)
async def create_chat_completion(request: ChatCompletionRequest):
    if request.stream:
        generate = create_stream_chat_completion_response(request, worker)
        return EventSourceResponse(generate, media_type="text/event-stream")
    else:
        return await create_chat_completion_response(request, worker)


def release_model_semaphore(fn=None):
    model_semaphore.release()
    if fn is not None:
        fn()


@app.post("/worker_get_status")
async def get_status(request: Request):
    return worker.get_status()


if __name__ == "__main__":
    worker = BridgeWorker()
    uvicorn.run(app, host=WORKER_HOST, port=WORKER_PORT, log_level="info")
