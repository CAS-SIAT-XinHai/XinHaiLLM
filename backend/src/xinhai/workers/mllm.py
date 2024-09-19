import asyncio
import base64
import io
import json
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from threading import Thread
from typing import Sequence, Dict, Optional, List, Generator, AsyncGenerator, Any, Annotated, Tuple, AsyncIterator
import torch
import requests
import uvicorn

from fastapi import FastAPI, HTTPException, status, Depends
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sse_starlette import EventSourceResponse
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from PIL import Image
from llamafactory.api.protocol import Role, ModelList, ModelCard, ChatCompletionResponse, ChatCompletionRequest, \
    Function, \
    ChatCompletionMessage, FunctionCall, Finish, ChatCompletionResponseChoice, ChatCompletionResponseUsage, \
    ChatCompletionStreamResponse, ChatCompletionStreamResponseChoice
from llamafactory.chat.base_engine import Response
from llamafactory.data import Role as DataRole
from llamafactory.extras.misc import torch_gc
from xinhai.config import LOG_DIR, WORKER_HEART_BEAT_INTERVAL
from xinhai.utils import build_logger, pretty_print_semaphore
from transformers import AutoProcessor, LlavaForConditionalGeneration

# Input image and question
image = ImageAsset("cherry_blossom").pil_image.convert("RGB")
question = "What is the content of this image?"


# LLaVA-1.5
def run_llava(model_name_or_path):
    llm = LlavaForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(0)
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    conversation = [
        {

            "role": "user",
            "content": [
                {"type": "text", "text": "What are these?"},
                {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    stop_token_ids = None
    return llm, prompt,processor, stop_token_ids

# LLaVA-1.6/LLaVA-NeXT
def run_llava_next(model_name_or_path):
    prompt = "[INST] <image>\n{question} [/INST]"
    llm = LLM(
        model=model_name_or_path,
        dtype="float16"
    )
    stop_token_ids = None
    return llm, prompt, stop_token_ids

# Fuyu
def run_fuyu(model_name_or_path):
    prompt = "{question}\n"
    llm = LLM(
        model=model_name_or_path,
        dtype="float16"
    )
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# Phi-3-Vision
def run_phi3v(model_name_or_path):
    prompt = "<|user|>\n<|image_1|>\n{question}<|end|>\n<|assistant|>\n"  # noqa: E501
    # Note: The default setting of max_num_seqs (256) and
    # max_model_len (128k) for this model may cause OOM.
    # You may lower either to run this example on lower-end GPUs.

    # In this example, we override max_num_seqs to 5 while
    # keeping the original context length of 128k.
    llm = LLM(
        model=model_name_or_path,
        dtype="float16",
        trust_remote_code=True,
        max_num_seqs=5,
    )
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# PaliGemma
def run_paligemma(model_name_or_path):
    # PaliGemma has special prompt format for VQA
    prompt = "caption en"
    llm = LLM(
        model=model_name_or_path,
        dtype="float16"
    )
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# Chameleon
def run_chameleon(model_name_or_path):
    prompt = "{question}<image>"
    llm = LLM(
        model=model_name_or_path,
        dtype="float16"
    )
    stop_token_ids = None
    return llm, prompt, stop_token_ids


# MiniCPM-V
def run_minicpmv(model_name_or_path):
    # 2.0
    # The official repo doesn't work yet, so we need to use a fork for now
    # For more details, please see: See: https://github.com/vllm-project/vllm/pull/4087#issuecomment-2250397630 # noqa
    # model_name = "HwwwH/MiniCPM-V-2"

    # 2.5
    # model_name = "openbmb/MiniCPM-Llama3-V-2_5"

    # 2.6
    model_name = model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    llm = LLM(
        model=model_name,
        dtype="float16",
        trust_remote_code=True,
        enable_prefix_caching=False
    )
    # NOTE The stop_token_ids are different for various versions of MiniCPM-V
    # 2.0
    # stop_token_ids = [tokenizer.eos_id]

    # 2.5
    stop_token_ids = [tokenizer.eos_id, tokenizer.eot_id]

    # 2.6
    #stop_tokens = ['<|im_end|>', '<|endoftext|>']
    #stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]

    messages = [{
        'role': 'user',
        'content': '(<image>./</image>)\n{question}'
    }]
    prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)
    return llm, prompt, stop_token_ids

# InternVL
def run_internvl(model_name_or_path):
    model_name = model_name_or_path
    llm = LLM(
        model=model_name,
        dtype="float16",
        trust_remote_code=True,
        max_num_seqs=5,
        max_model_len=61152
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True)
    messages = [{'role': 'user', 'content': "<image>\n{question}"}]
    prompt = tokenizer.apply_chat_template(messages,
                                           tokenize=False,
                                           add_generation_prompt=True)

    # Stop tokens for InternVL
    # models variants may have different stop tokens
    # please refer to the model card for the correct "stop words":
    # https://huggingface.co/OpenGVLab/InternVL2-2B#service
    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
    return llm, prompt, stop_token_ids


# BLIP-2
def run_blip2(model_name_or_path):
    # BLIP-2 prompt format is inaccurate on HuggingFace model repository.
    # See https://huggingface.co/Salesforce/blip2-opt-2.7b/discussions/15#64ff02f3f8cf9e4f5b038262 #noqa
    prompt = "Question: {question} Answer:"
    llm = LLM(
        model=model_name_or_path,
        dtype="float16"
    )
    stop_token_ids = None
    return llm, prompt, stop_token_ids

def run_qwen2VL(model_name_or_path):
    prompt = "Question: {question} Answer:"
    llm = LLM(
        model=model_name_or_path,
        dtype="float16"
    )
    stop_token_ids = None
    return llm, prompt, stop_token_ids



model_example_map = {
    "llava": run_llava,
    "llava-next": run_llava_next,
    "fuyu": run_fuyu,
    "phi3_v": run_phi3v,
    "paligemma": run_paligemma,
    "chameleon": run_chameleon,
    "minicpmv": run_minicpmv,
    "blip-2": run_blip2,
    "internvl_chat": run_internvl,
    "qwen2VL":run_qwen2VL
}

GB = 1 << 30
worker_id = str(uuid.uuid4())[:6]
global_counter = 0
CONTROLLER_ADDRESS = os.environ.get("CONTROLLER_ADDRESS", "http://localhost:5000")
WORKER_ADDRESS = os.environ.get("WORKER_ADDRESS", "http://localhost:40001")
WORKER_HOST = os.environ.get("WORKER_HOST", "localhost")
WORKER_PORT = int(os.environ.get("WORKER_PORT", 7861))
NO_REGISTER = os.environ.get("NO_REGISTER", False)
MODEL_NAME = os.environ.get("MODEL_NAME", "MiniCPMV")
LIMIT_MODEL_CONCURRENCY = int(os.environ.get("LIMIT_MODEL_CONCURRENCY", 5))
MODEL_PATH = os.environ.get("MODEL_PATH", "/data/pretrained_models/InternVL2-2B")
DEVICE = "cuda"
model_semaphore = None
api_key = os.environ.get("API_KEY", "EMPTY")
security = HTTPBearer(auto_error=False)
logger = build_logger("model_worker", f"model_worker_{worker_id}.log", LOG_DIR)


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

class MLLMWorker:
    def __init__(self) -> None:
        if MODEL_NAME not in model_example_map:
            raise ValueError(f"Model type {MODEL_NAME} is not supported.")

        self.llm, self.prompt, self.stop_token_ids = model_example_map[MODEL_NAME](MODEL_PATH)

        self._loop = asyncio.new_event_loop()
        self._thread = Thread(target=_start_background_loop, args=(self._loop,), daemon=True)
        self._thread.start()

        if not NO_REGISTER:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(
                target=heart_beat_worker, args=(self,))
            self.heart_beat_thread.start()

    async def _generate(
            self,
            messages: Sequence[Dict[str, str]],
            system: Optional[str] = None,
            tools: Optional[str] = None,
            image: Optional["NDArray"] = None,
            **input_kwargs,
    ) -> AsyncIterator["RequestOutput"]:
        request_id = "chatcmpl-{}".format(uuid.uuid4().hex)
        print(messages)
        print(image)

        sampling_params = SamplingParams(temperature=0.2,
                                         max_tokens=2048,
                                         stop_token_ids=self.stop_token_ids)

        num_prompts = len(messages)

        if num_prompts == 1:
            # Single inference
            if image:
                inputs = {
                    "prompt": self.prompt.format(question=messages[0]["content"]),
                    "multi_modal_data": {
                        "image": image
                    },
                }
            else:
                inputs = {
                    "prompt": self.prompt.format(question=messages[0]["content"]),
                }
        else:
            # Batch inference
            inputs = [{
                "prompt": self.prompt,
                "multi_modal_data": {
                    "image": image
                },
            } for _ in range(num_prompts)]

        result_generator = self.llm.generate(
            inputs,
            sampling_params=sampling_params,
            # request_id=request_id,
            # lora_request=self.lora_request,

        )
        #return result_generator
        for item in result_generator:
            yield item
    async def chat(
            self,
            messages: Sequence[Dict[str, str]],
            system: Optional[str] = None,
            tools: Optional[str] = None,
            image: Optional["NDArray"] = None,
            **input_kwargs,
    ) -> List["Response"]:
        final_output = None
        # generator = await self._generate(messages, system, tools, image, **input_kwargs)
        # for request_output in generator:
        #     final_output = request_output
        async for o in self._generate(messages, system, tools, image, **input_kwargs):
            final_output = o.outputs[0].text

        results = []
        context=""
        for output in final_output:
            context+=output
        logger.info("model回答："+context)
        results.append(
            Response(
                response_text=context,
                response_length=len(context),
                prompt_length=len(context),
                finish_reason="stop",
            )
        )

        return results

    async def stream_chat(
            self,
            messages: Sequence[Dict[str, str]],
            system: Optional[str] = None,
            tools: Optional[str] = None,
            image: Optional["NDArray"] = None,
            **input_kwargs,
    ) -> Generator[str, None, None]:
        generated_text = ""
        generator = await self._generate(messages, system, tools, image, **input_kwargs)
        async for result in generator:
            delta_text = result.outputs[0].text[len(generated_text):]
            generated_text = result.outputs[0].text
            yield delta_text

    async def achat(
            self,
            messages: Sequence[Dict[str, str]],
            system: Optional[str] = None,
            tools: Optional[str] = None,
            image: Optional["NDArray"] = None,
            **input_kwargs,
    ) -> List["Response"]:
        return await self.chat(messages, system, tools, image, **input_kwargs)

    async def astream_chat(
            self,
            messages: Sequence[Dict[str, str]],
            system: Optional[str] = None,
            tools: Optional[str] = None,
            image: Optional["NDArray"] = None,
            **input_kwargs,
    ) -> AsyncGenerator[str, None]:
        async for new_token in self.stream_chat(messages, system, tools, image, **input_kwargs):
            yield new_token

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


def _process_request(
        request: "ChatCompletionRequest",
) -> Tuple[List[Dict[str, str]], Optional[str], Optional[str], Optional["NDArray"]]:
    logger.info("==== request ====\n{}".format(json.dumps(dictify(request), indent=2, ensure_ascii=False)))

    if len(request.messages) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid length")

    if request.messages[0].role == Role.SYSTEM:
        system = request.messages.pop(0).content
    else:
        system = None

    if len(request.messages) % 2 == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only supports u/a/u/a/u...")

    input_messages = []
    image = None
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
            for input_item in message.content:
                if input_item.type == "text":
                    input_messages.append({"role": ROLE_MAPPING[message.role], "content": input_item.text})
                else:
                    image_url = input_item.image_url.url
                    if image_url.startswith("data:image"):  # base64 image
                        image_data = base64.b64decode(image_url.split(",", maxsplit=1)[1])
                        image_path = io.BytesIO(image_data)
                    elif os.path.isfile(image_url):  # local file
                        image_path = open(image_url, "rb")
                    else:  # web uri
                        image_path = requests.get(image_url, stream=True).raw

                    from io import BytesIO
                    image = Image.open(image_path).convert('RGB').resize((800,600))
        else:
            input_messages.append({"role": ROLE_MAPPING[message.role], "content": message.content})

    tool_list = request.tools
    if isinstance(tool_list, list) and len(tool_list):
        try:
            tools = json.dumps([dictify(tool.function) for tool in tool_list], ensure_ascii=False)
        except json.JSONDecodeError:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid tools")
    else:
        tools = None

    return input_messages, system, tools, image


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
    input_messages, system, tools, image = _process_request(request)
    responses = await chat_model.achat(
        input_messages,
        system,
        tools,
        image,
        do_sample=request.do_sample,
        temperature=request.temperature,
        top_p=request.top_p,
        max_new_tokens=request.max_tokens,
        num_return_sequences=request.n,
        stop=request.stop,
    )

    prompt_length, response_length = 0, 0
    choices = []
    for i, response in enumerate(responses):
        if tools:
            result = chat_model.engine.template.extract_tool(response.response_text)
        else:
            result = response.response_text

        if isinstance(result, list):
            tool_calls = []
            for tool in result:
                function = Function(name=tool[0], arguments=tool[1])
                tool_calls.append(FunctionCall(id="call_{}".format(uuid.uuid4().hex), function=function))

            response_message = ChatCompletionMessage(role=Role.ASSISTANT, tool_calls=tool_calls)
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
    input_messages, system, tools, image = _process_request(request)
    if tools:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot stream function calls.")

    if request.n > 1:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot stream multiple responses.")

    yield _create_stream_chat_completion_chunk(
        completion_id=completion_id, model=request.model, delta=ChatCompletionMessage(role=Role.ASSISTANT, content="")
    )
    async for new_token in chat_model.astream_chat(
            input_messages,
            system,
            tools,
            image,
            do_sample=request.do_sample,
            temperature=request.temperature,
            top_p=request.top_p,
            max_new_tokens=request.max_tokens,
            stop=request.stop,
    ):
        if len(new_token) != 0:
            yield _create_stream_chat_completion_chunk(
                completion_id=completion_id, model=request.model, delta=ChatCompletionMessage(content=new_token)
            )

    yield _create_stream_chat_completion_chunk(
        completion_id=completion_id, model=request.model, delta=ChatCompletionMessage(), finish_reason=Finish.STOP
    )
    yield "[DONE]"


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
    #dependencies=[Depends(verify_api_key)],
)
async def create_chat_completion(request: ChatCompletionRequest):
    return await create_chat_completion_response(request, worker)


def release_model_semaphore(fn=None):
    model_semaphore.release()
    if fn is not None:
        fn()


@app.post("/worker_get_status")
async def get_status(request: Request):
    return worker.get_status()


if __name__ == "__main__":
    worker = MLLMWorker()
    uvicorn.run(app, host=WORKER_HOST, port=WORKER_PORT, log_level="info")
