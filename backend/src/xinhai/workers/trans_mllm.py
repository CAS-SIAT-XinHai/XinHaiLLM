"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors:wuhaihong
"""

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
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
# LLaVA-1.5
def run_llava(model_name_or_path):
    llm = LlavaForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        load_in_4bit=True
    ).to(0)
    tokenizers=None
    return llm,tokenizers

# MiniCPM-V
def run_minicpmv(model_name_or_path):
    model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.float16)
    model = model.to(device='cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model.eval()
    return model,tokenizer

# InternVL
def run_internvl(model_name_or_path):
    model = AutoModel.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, use_fast=False)

    return model,tokenizer

def run_qwen2VL(model_name_or_path):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name_or_path, torch_dtype="auto", device_map="auto"
    )
    tokenizer=None
    return model,tokenizer


model_example_map = {
    "llava": run_llava,
    "minicpmv": run_minicpmv,
    "internvl_chat": run_internvl,
    "qwen2VL":run_qwen2VL
}

def ask_llava(llm,tokenizer,modelpath,question,image):
    processor = AutoProcessor.from_pretrained(modelpath)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "{}".format(question)},
                {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    raw_image = Image.open(image)
    inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)
    output = llm.generate(**inputs, max_new_tokens=200, do_sample=False)
    answer_str=processor.decode(output[0][2:], skip_special_tokens=True)
    return answer_str
def ask_minicpmv(llm,tokenizer,modelpath,question,image):
    #image = Image.open(image).convert('RGB')
    msgs = [{'role': 'user', 'content': question}]
    res = llm.chat(
        image=image,
        msgs=msgs,
        tokenizer=tokenizer,
        sampling=True,  # if sampling=False, beam_search will be used by default
        temperature=0.7,
        # system_prompt='' # pass system_prompt if needed
    )
    #print("模型回答:"+res)
    return res
def ask_internvl(llm,tokenizer,modelpath,question,image):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    import torchvision.transforms as T
    def build_transform(input_size):
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        from torchvision.transforms.functional import InterpolationMode
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform
    def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio
    def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images
    def load_image(image_file, input_size=448, max_num=12):
        image = Image.open(image_file).convert('RGB')
        transform = build_transform(input_size=input_size)
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    pixel_values = load_image(image, max_num=12).to(torch.bfloat16).cuda()
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    response, history = llm.chat(tokenizer, None, question, generation_config, history=None, return_history=True)
    return response
def ask_qwen2VL(llm,tokenizer,modelpath,question,image)->str:
    processor = AutoProcessor.from_pretrained(modelpath)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": question},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    # Inference: Generation of the output
    generated_ids = llm.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    if output_text and output_text[0]:
        return output_text[0]
    else:
        return ""

model_asker_map = {
    "llava": ask_llava,
    "minicpmv": ask_minicpmv,
    "internvl_chat": ask_internvl,
    "qwen2VL":ask_qwen2VL
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

        self.llm,self.tokenizer= model_example_map[MODEL_NAME](MODEL_PATH)

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
    ) :
        request_id = "chatcmpl-{}".format(uuid.uuid4().hex)
        print(messages)
        print(image)
        question = messages[0]["content"]

        model_answer=model_asker_map[MODEL_NAME](self.llm,self.tokenizer,MODEL_PATH,question,image)
        #print(model_answer)
        if not isinstance(model_answer, str):
            raise TypeError("The model_answer is not a string type.")
        return model_answer
    async def chat(
            self,
            messages: Sequence[Dict[str, str]],
            system: Optional[str] = None,
            tools: Optional[str] = None,
            image: Optional["NDArray"] = None,
            **input_kwargs,
    ) -> List["Response"]:
        final_output = None
        context=await self._generate(messages, system, tools, image, **input_kwargs)
        results = []
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

    #只配置非流输出形式
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
