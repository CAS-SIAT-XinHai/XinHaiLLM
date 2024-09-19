import time
from enum import Enum, unique
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field
from typing_extensions import Literal

import base64, requests, json, os, io
# API的URL
url = "http://localhost:40001/v1/chat/completions"
from pydantic import BaseModel, Field
class ImageURL(BaseModel):
    url: str
class MultimodalInputItem(BaseModel):
    type: Literal["text", "image_url"]
    text: Optional[str] = None
    image_url: Optional[ImageURL] = None

def get_image_meta(file_path=""):
    # 获取文件的基本信息
    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    file_type = os.path.splitext(file_path)[1].lower()[1:]  # 获取文件扩展名并转换为小写
    # 生成 base64 编码的预览图
    file = open(file_path, "rb").read()
    base64_data = base64.b64encode(file)
    base64_str = str(base64_data, 'utf-8')
    # 构建元数据字典
    file_metadata = {
        "name": file_name,
        "size": file_size,
        "type": file_type,
        "audio": False,  # 因为这是一个图像文件
        "duration": 0,  # 图像文件没有持续时间
        "url": f"data:image/{file_type};base64,{base64_str}",  # 如果需要上传到某个位置，这里可以填充实际的 URL
        "preview": f"data:image/{file_type};base64,{base64_str}",
        "progress": 100,  # 假设文件已经完全读取
    }
    return file_metadata
def re_image(url1):
    url1=url1
    question="Please identify and describe the content of the image."
    txt = MultimodalInputItem(type="text", text=question)
    pie=MultimodalInputItem(type="image_url",image_url=ImageURL(url=url1))
    # 请求体
    payload = {
        "model": "minicpmv",
        "messages" :[{
            "role": "user",
            "content": [txt.dict(),pie.dict()]
        }]
    }

    headers = {'Content-Type': 'application/json'}
    # 发送POST请求
    response = requests.post(url, data=json.dumps(payload),headers=headers)
    response=response.json()
    answer=response['choices'][0]['message']['content']
    # 打印响应
    print("回答:", answer)
    filename = url1.split("/")[-1]
    model_name="InternVL2-4B"
    # data={"image_name":filename,"model":model_name,"question":question,"answer":answer}
    # with open("/home/wuhaihong/xinhai/test/answer.jsonl", "a") as json_file:
    #     json.dump(data, json_file, indent=4, ensure_ascii=False)
    #     json_file.write("\n")

from pathlib import Path
def find_images(folder_path):
    image_files = []

    # 使用 Path 对象查找所有 .jpg 和 .png 文件
    for image_path in Path(folder_path).rglob('*'):
        if image_path.suffix.lower() in ['.jpg', '.png']:
            # 获取绝对路径
            image_files.append(str(image_path.resolve()))

    return image_files

folder_path = r"/home/wuhaihong/xinhai/test_piture"
image_files = find_images(folder_path)

for image in image_files:
    re_image(image)