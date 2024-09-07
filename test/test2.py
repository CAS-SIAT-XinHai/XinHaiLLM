import time
from enum import Enum, unique
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field
from typing_extensions import Literal

import base64, requests, json, os, io
# API的URL
url = "http://localhost:40004/v1/chat/completions"
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

txt = MultimodalInputItem(type="text", text="你能看到我发的一张图片吗？")
pie=MultimodalInputItem(type="image_url",image_url=ImageURL(url="/home/wuhaihong/xinhai/static/663346d79889c668e627799e7ebf126.png"))
# 请求体
payload = {
    "model": "Qwen1.5-7B-Chat",
    "messages" :[{
        "role": "user",
        "content": [txt.dict(),pie.dict()]
    }]
}

headers = {'Content-Type': 'application/json'}
# 发送POST请求
response = requests.post(url, data=json.dumps(payload),headers=headers)

# 打印响应
print("状态码:", response.status_code)
print("响应体:", response.json())


