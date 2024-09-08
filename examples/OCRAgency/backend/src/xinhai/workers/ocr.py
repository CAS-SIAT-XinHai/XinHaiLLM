"""
A model worker executes the model.
"""
import base64
import io
import json
import os
import threading
import time
import uuid
from typing import Sequence, Dict, Optional, List, Generator, AsyncGenerator, Any, Annotated, Tuple

import df2img
import numpy as np
import pandas as pd
import requests
import torch
import uvicorn
from PIL import Image
from fastapi import FastAPI, Request
from paddleocr import PaddleOCR, draw_ocr
from llamafactory.api.protocol import Role, ModelList, ModelCard, ChatCompletionResponse, ChatCompletionRequest, \
    Function, \
    ChatCompletionMessage, FunctionCall, Finish, ChatCompletionResponseChoice, ChatCompletionResponseUsage, \
    ChatCompletionStreamResponse, ChatCompletionStreamResponseChoice
from llamafactory.chat.base_engine import Response
from llamafactory.data import Role as DataRole
from llamafactory.extras.misc import torch_gc
from fastapi import FastAPI, HTTPException, status, Depends
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from xinhai.config import LOG_DIR, WORKER_HEART_BEAT_INTERVAL
from xinhai.utils import build_logger, pretty_print_semaphore


GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
logger = build_logger("model_worker", f"model_worker_{worker_id}.log", LOG_DIR)
global_counter = 0

model_semaphore = None

CONTROLLER_ADDRESS = os.environ.get("CONTROLLER_ADDRESS")
WORKER_ADDRESS = os.environ.get("WORKER_ADDRESS")
WORKER_HOST = os.environ.get("WORKER_HOST")
WORKER_PORT = int(os.environ.get("WORKER_PORT", 40004))
NO_REGISTER = os.environ.get("NO_REGISTER", False)
MODEL_NAME = os.environ.get("MODEL_NAME", "paddleocr")
LIMIT_MODEL_CONCURRENCY = int(os.environ.get("LIMIT_MODEL_CONCURRENCY", 5))
DEVICE = "cuda"
STATIC_PATH="/home/whh/project/Xinhai/static"

def heart_beat_worker(controller):
    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()


class OCRModelWorker:
    def __init__(self):
        # ocr model
        self.ocr_model = PaddleOCR(use_angle_cls=True, use_gpu=True, lang="ch")

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

    def get_ocred_result(self, image, result):
        str_in_image = ''
        img_b64_str = ""
        if result[0] is not None:
            boxes = [line[0] for line in result[0]]
            txts = [line[1][0] for line in result[0]]
            scores = [line[1][1] for line in result[0]]

            im_show = draw_ocr(image, boxes, txts, scores, font_path=os.path.join(STATIC_PATH, 'simfang.ttf'))
            im_show = Image.fromarray(im_show)
            buffered = io.BytesIO()
            im_show.save(buffered, format='JPEG')
            img_b64_str = base64.b64encode(buffered.getvalue()).decode()

            result = [res[1][0] for res in result[0] if res[1][1] > 0.1]
            if len(result) > 0:
                str_in_image += '\n'.join(result)
        return str_in_image, img_b64_str

    @torch.inference_mode()
    def ocr_image(self, params):
        # image_name = params.get("image", None)
        # image = Image.open(os.path.join(STATIC_PATH, image_name)).convert("RGB")
        img_b64_str=params.get("image",None)
        image_data = base64.b64decode(img_b64_str)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        result = self.ocr_model.ocr(np.asarray(image), cls=True)
        str_in_image, img_b64_str = self.get_ocred_result(image, result)
        return json.dumps({
            "title": params.get("title"),
            "description": str_in_image,
            "image": img_b64_str,
            "error_code": 0
        })


    @torch.inference_mode()
    def parse_file(self, params):
        filename = params.get("filename", None)
        texts = []
        if filename.endswith(".xlsx"):
            sheet_to_df_map = pd.read_excel(os.path.join(STATIC_PATH, filename),
                                            sheet_name=None,
                                            index_col=0)
            for k, df in sheet_to_df_map.items():
                if not df.empty:
                    df.columns = [c.replace('Unnamed:', "") for c in df.columns]
                    df.fillna("", inplace=True)
                    fig = df2img.plot_dataframe(df, show_fig=False,
                                                title=dict(
                                                    font_family="WenQuanYi Micro Hei"
                                                ),
                                                tbl_header=dict(
                                                    font_family="WenQuanYi Micro Hei"
                                                ),
                                                tbl_cells=dict(
                                                    font_family="WenQuanYi Micro Hei"
                                                ),
                                                fig_size=(1000, 280))
                    # fig.update_layout(font=os.path.join(STATIC_PATH, 'simfang.ttf'))
                    fig_bytes = fig.to_image(format="png")
                    buf = io.BytesIO(fig_bytes)
                    img_b64_str = base64.b64encode(buf.getvalue()).decode()

                    texts.append({
                        "title": k,
                        "image": img_b64_str,
                        "description": df.to_string(index=False, index_names=False),
                        "error_code": 0
                    })
        elif filename.endswith(".rar"):
            import rarfile

            with rarfile.RarFile(os.path.join(STATIC_PATH, filename)) as rf:
                for f in rf.infolist():
                    print(f.filename, f.file_size)
                    if f.filename == "README":
                        print(rf.read(f))
        elif filename.endswith(".7z"):
            from py7zr import SevenZipFile
            with SevenZipFile(os.path.join(STATIC_PATH, filename), 'r') as zip:
                for fname, bio in zip.readall().items():
                    if fname.endswith((".jpg", ".jpeg", ".png", ".gif")):
                        image = Image.open(bio).convert("RGB")
                        result = self.ocr_model.ocr(np.asarray(image), cls=True)
                        str_in_image, img_b64_str = self.get_ocred_result(image, result)
                        texts.append({
                            "title": filename,
                            "description": str_in_image,
                            "image": img_b64_str,
                            "error_code": 0
                        })

        return json.dumps({
            "texts": texts,
            "error_code": 0
        })

    async def chat(
            self,
            messages: Sequence[Dict[str, str]],
            system: Optional[str] = None,
            tools: Optional[str] = None,
            image: Optional["NDArray"] = None,
            **input_kwargs,
    ) -> List["Response"]:
        params=json.loads(messages[0]["content"])
        final_output = self.ocr_image(params)
        results = []
        results.append(
            Response(
                response_text=final_output,
                response_length=len(final_output),
                prompt_length=len(final_output),
                finish_reason="stop",
            )
        )

        return results
    async def achat(
            self,
            messages: Sequence[Dict[str, str]],
            system: Optional[str] = None,
            tools: Optional[str] = None,
            image: Optional["NDArray"] = None,
            **input_kwargs,
    ) -> List["Response"]:
        return await self.chat(messages, system, tools, image, **input_kwargs)


app = FastAPI()

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
def dictify(data: "BaseModel") -> Dict[str, Any]:
    try:  # pydantic v2
        return data.model_dump(exclude_unset=True)
    except AttributeError:  # pydantic v1
        return data.dict(exclude_unset=True)


def release_model_semaphore(fn=None):
    model_semaphore.release()
    if fn is not None:
        fn()


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

async def create_chat_completion_response(
        request: "ChatCompletionRequest", chat_model: "ChatModel"
) -> "ChatCompletionResponse":
    completion_id = "chatcmpl-{}".format(uuid.uuid4().hex)
    input_messages, system, tools, image= _process_request(request)
    responses = await chat_model.achat(
        input_messages,
        system,
        tools,
        image,
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

@app.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    status_code=status.HTTP_200_OK,
    #dependencies=[Depends(verify_api_key)],
)
async def create_chat_completion(request: ChatCompletionRequest):
        return await create_chat_completion_response(request, worker)

@app.post("/worker_ocr_image")
async def ocr_image(request: Request):
    params = await request.json()
    return worker.ocr_image(params)

@app.post("/worker_parse_file")
async def parse_file(request: Request):
    params = await request.json()
    return worker.parse_file(params)

@app.post("/worker_get_status")
async def get_status(request: Request):
    return worker.get_status()



if __name__ == "__main__":
    worker = OCRModelWorker()
    uvicorn.run(app, host=WORKER_HOST, port=WORKER_PORT, log_level="info")
