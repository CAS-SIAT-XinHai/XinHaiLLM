import time
from enum import Enum, unique
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field
from typing_extensions import Literal

import base64, requests, json, os, io
from openai import OpenAI, OpenAIError
# API的URL
url = "http://localhost:40001/v1"
from pydantic import BaseModel, Field
client = OpenAI(
            api_key="EMPTY",
            base_url=url,
        )
chat_response = client.chat.completions.create(
    model="Qwen1.5-7B-Chat",
    messages=[{
        "role": "user",
        "content": "咨询者，面临以下心理问题：“察觉到一种模式，我总喜欢自己检讨、批评自己？”，具体如下：“我和我妈妈讲话会陷入一种很恶心的模式。我只要说遇到的烦恼她是无动于衷的，但是我一旦开始批评自我检讨，她就面露喜色。我觉得我已经不好办了，回想发现见人就自我检讨，觉得自己不是这里不足就是那里不足，再假假的夸两句别人，已经成为我的一种模式了，但这不正常的，苦恼？"
    }],
    stream=True,
    temperature=0.98

)
content = ""
for chunk in chat_response:
    if chunk.choices[0].delta.content:
        content += chunk.choices[0].delta.content



# 打印响应
print("响应体:", content)


