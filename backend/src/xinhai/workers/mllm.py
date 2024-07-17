import asyncio
import json, math, os, threading, time, uuid, base64, io
from io import BytesIO
from contextlib import asynccontextmanager
from threading import Thread
from typing import Sequence, Dict, Optional, List, Generator, AsyncGenerator, Any, Annotated, Tuple
from transformers import AutoConfig, AutoTokenizer
import timm
import torch
from PIL import Image
from torchvision import transforms
from vllm import LLM, SamplingParams
from vllm.sequence import MultiModalData
import requests
import uvicorn
from fastapi import FastAPI, HTTPException, status, Depends
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from llamafactory.api.protocol import Role, ModelList, ModelCard, ChatCompletionResponse, ChatCompletionRequest, \
    Function, \
    ChatCompletionMessage, FunctionCall, Finish, ChatCompletionResponseChoice, ChatCompletionResponseUsage, \
    ChatCompletionStreamResponse, ScoreEvaluationResponse, ScoreEvaluationRequest, ChatCompletionStreamResponseChoice
from llamafactory.chat.hf_engine import HuggingfaceEngine
from llamafactory.chat.vllm_engine import VllmEngine
from llamafactory.data import Role as DataRole
from llamafactory.extras.misc import torch_gc
from llamafactory.hparams import get_infer_args
from sse_starlette import EventSourceResponse

from backend.src.xinhai.utils import build_logger, pretty_print_semaphore
from backend.src.xinhai.config import LOG_DIR, WORKER_HEART_BEAT_INTERVAL, STATIC_PATH

GB = 1 << 30
worker_id = str(uuid.uuid4())[:6]
global_counter = 0
CONTROLLER_ADDRESS = os.environ.get("CONTROLLER_ADDRESS", "http://localhost:5000")
WORKER_ADDRESS = os.environ.get("WORKER_ADDRESS", "http://localhost:40001")
WORKER_HOST = os.environ.get("WORKER_HOST", "localhost")
WORKER_PORT = int(os.environ.get("WORKER_PORT", 7861))
NO_REGISTER = os.environ.get("NO_REGISTER", False)
MODEL_NAME = os.environ.get("MODEL_NAME", "minicpmv")
LIMIT_MODEL_CONCURRENCY = int(os.environ.get("LIMIT_MODEL_CONCURRENCY", 5))
MODEL_PATH = os.environ.get("MODEL_PATH", "/data/xuancheng/MiniCPM-V-2")
DEVICE = "cuda"
model_semaphore = None
api_key = os.environ.get("API_KEY", "EMPTY")
security = HTTPBearer(auto_error=False)
logger = build_logger("model_worker", f"model_worker_{worker_id}.log", LOG_DIR)

def heart_beat_worker(controller):
    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()

@asynccontextmanager
async def lifespan(app: "FastAPI"):
    yield
    torch_gc()

app = FastAPI(lifespan=lifespan)

def get_grid_placeholder(grid, query_num):
    image_placeholder = query_num + 2
    cols = grid[0]
    rows = grid[1]
    slices = 0
    for i in range(rows):
        lines = 0
        for j in range(cols):
            lines += image_placeholder
        if i < rows - 1:
            slices += lines + 1
        else:
            slices += lines
    slice_placeholder = 2 + slices
    return slice_placeholder

def slice_image(image, max_slice_nums=9, scale_resolution=448, patch_size=14, never_split=False):
    original_size = image.size
    original_width, original_height = original_size
    log_ratio = math.log(original_width / original_height)
    ratio = original_width * original_height / (scale_resolution * scale_resolution)
    multiple = min(math.ceil(ratio), max_slice_nums)
    best_grid = None
    if multiple > 1 and not never_split:
        candidate_split_grids_nums = []
        for i in [multiple - 1, multiple, multiple + 1]:
            if i == 1 or i > max_slice_nums:
                continue
            candidate_split_grids_nums.append(i)
        # source image, down-sampling and ensure divided by patch_size
        candidate_grids = []
        # find best grid
        for split_grids_nums in candidate_split_grids_nums:
            m = 1
            while m <= split_grids_nums:
                if split_grids_nums % m == 0:
                    candidate_grids.append([m, split_grids_nums // m])
                m += 1
        best_grid = [1, 1]
        min_error = float("inf")
        for grid in candidate_grids:
            error = abs(log_ratio - math.log(grid[0] / grid[1]))
            if error < min_error:
                best_grid = grid
                min_error = error
    return best_grid

class MLLMWorker:
    def __init__(self) -> None:
        self.config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        self.llm = LLM(
            model=MODEL_PATH,
            image_input_type="pixel_values",
            image_token_id=101,
            image_input_shape="1,3,448,448",
            image_feature_size=64,
            gpu_memory_utilization=0.75,
            trust_remote_code=True,
        )

        if not NO_REGISTER:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(target=heart_beat_worker, args=(self,))
            self.heart_beat_thread.start()

    def get_slice_image_placeholder(self, image):
        image_placeholder = self.config.query_num + 2

        best_grid = slice_image(
            image,
            self.config.max_slice_nums,
            self.config.scale_resolution,
            self.config.patch_size,
        )
        final_placeholder = image_placeholder

        if best_grid is not None:
            final_placeholder += get_grid_placeholder(best_grid, self.config.query_num)

        return final_placeholder - 1

    def get_prompt_and_image(self, input):
        image_base64_list = input['images']
        messages = input['messages']

        image_trans = []
        image_label = ""
        unk_label = ""
        images_result = ""
        if len(image_base64_list) > 0:
            for image_base64 in image_base64_list:
                image = Image.open(BytesIO(base64.b64decode(image_base64))).convert('RGB')
                addtion_tokens = self.get_slice_image_placeholder(image)
                image_label += "<image></image>"
                unk_label += '<unk>' * addtion_tokens
                image_trans.append(transforms.Compose([transforms.ToTensor()])(img=image))
            images_result = torch.stack(image_trans)
        question = ""
        for index, message_dict in enumerate(messages):
            role = message_dict["role"]
            content = message_dict["content"]

            if len(messages) == 1:
                question = f"<{role}>{image_label}{content}<AI>{unk_label}"
                return question, images_result

            if index > 0 and index == len(messages)-1:
                question += f"<{role}>{content}<assistant>{unk_label}"
            else:
                question += f"<{role}>{content}"
        return question, images_result

    def chat(self, input:dict):
        sampling_params = SamplingParams(
            temperature=input.get("temperature", 0.7),
            top_p=input.get("top_p", 0.8),
            top_k=input.get("top_k", 100),
            seed=input.get("seed", 3472),
            max_tokens=input.get("max_tokens", 1024),
            min_tokens=input.get("min_tokens", 150),
        )
        prompt, images = self.get_prompt_and_image(input)
        outputs = self.llm.generate(prompt, multi_modal_data=MultiModalData(type=MultiModalData.Type.IMAGE, data=images), sampling_params=sampling_params)
        return outputs[0].outputs[0].text



    async def achat(self, input:dict):
        return self.chat(input)

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

    # 调用测试函数
    # def test_call_method():
    #     url = "http://localhost:7861/worker_mllm_chat"  # 你的FastAPI服务器地址
    #     file = open("./example.png", "rb").read()
    #     base64_data = base64.b64encode(file)
    #     base64_str = str(base64_data, 'utf-8')
    #
    #     datas = {
    #         "model": "minicpmv",
    #         "temperature": 0,
    #         "messages": [{"role": "user", "content": "请解释一下图片内容"}],
    #         "images": [base64_str],
    #         "max_tokens": 2048
    #     }
    #     response = requests.post(url, data=json.dumps(datas))
    #     print(response.text)


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    pass

@app.post("/v1/chat/completions")
async def create_chat_completion():
    pass

@app.post("/worker_mllm_chat")
async def worker_mllm_chat(request: Request):
    params = await request.json()
    return worker.chat(params)

@app.post("/worker_get_status")
async def get_status(request: Request):
    return worker.get_status()


if __name__ == "__main__":
    worker = MLLMWorker()
    uvicorn.run(app, host=WORKER_HOST, port=WORKER_PORT, log_level="info")
