"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Vimos Tan
         yangdi
Date: 2024-07-19 17:22:57
LastEditTime: 2024-07-19 17:28:20
"""
import logging
from datetime import datetime
from typing import List
import sys
if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self
from xinhai.arena.agents import register_agent
from xinhai.arena.agents.base import BaseAgent
from xinhai.arena.topology.base import BaseTopology
from xinhai.types.arena import XinHaiArenaAgentTypes
from xinhai.types.message import XinHaiChatMessage
from xinhai.types.routing import XinHaiRoutingMessage, XinHaiRoutingType
from openai import OpenAI, OpenAIError
logger = logging.getLogger(__name__)
import base64
import io
import json
import requests
from PIL import Image
import re
@register_agent(XinHaiArenaAgentTypes.OCR_AGENT)
class OCR_AGENT(BaseAgent):
    agent_type = XinHaiArenaAgentTypes.OCR_AGENT
    def __init__(self, name, agent_id, role_description, llm, api_key, api_base,
                 environment_id, controller_address, locale,
                 allowed_routing_types, Answer_Refactoring_Template, user_question, answer_template):
        routing_prompt_template = " "
        summary_prompt_template = ""
        prompt_template = ""
        super().__init__(name, agent_id, role_description, llm, api_key, api_base,
                         environment_id, controller_address, locale,
                         allowed_routing_types, routing_prompt_template, summary_prompt_template, prompt_template)
        self.Answer_Refactoring_Template = Answer_Refactoring_Template
        self.user_question = user_question
        self.answer_template = answer_template

    def reset(self) -> None:
        pass

    @staticmethod
    def chat_completion(client, model, agent_id, messages):
        try:
            logger.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            logger.info(f"Sending messages to Agent-{agent_id}: {messages}")
            chat_response = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False,
                temperature=0.98
            )
            logger.info("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            content = chat_response.choices[0].message.content
            if content.strip():
                logger.info(f"Get response from Agent-{agent_id}: {content}")
                return content.strip()
            else:
                usage = chat_response.usage
                logger.error(f"Error response from Agent-{agent_id}: {usage}")
        except OpenAIError as e:
            # Handle all OpenAI API errors
            logger.warning("*****************************************")
            logger.warning(f"Error response from Agent-{agent_id}: {e}")

    def complete_conversation(self, prompt, num_retries=5):

        messages = [{
            "role": "user",
            "content": prompt,
        }]

        while True:
            logger.debug(messages)
            chat_response = self.chat_completion(self.client, model=self.llm, agent_id=self.agent_id, messages=messages)

            if chat_response:
                chat_response = str(chat_response)
                #rr = format_pattern.search(chat_response)
                # 由于整个匹配的字符串并不会以子组形式返回，所以需要手动提取整个匹配部分
                json_pattern = re.compile(r'\{.*?\}', re.DOTALL)  # re.DOTALL 允许 . 匹配换行符
                rr = json_pattern.search(chat_response)
                if rr:
                    json_str = rr.group(0)  # 提取 JSON 字符串
                    try:
                        response_data = json.loads(json_str)  # 将 JSON 字符串转换为字典
                        #print(response_data)
                        break
                    except json.JSONDecodeError as e:
                        logger.info(f"JSON 解码失败: {e},重新跑")

        response_data = json.dumps(response_data,ensure_ascii=False,)
        return self.name, response_data

    def ocr_conversation(self,image_url=None,num_retries=5):
        #answer_form = "The generated response should be enclosed by [Response] and [End of Response]."
        with open(image_url, "rb") as image_file:
            # 将图像读取为二进制数据
            image_data = image_file.read()
            # 将二进制数据进行 Base64 编码
            img_b64_str = base64.b64encode(image_data).decode('utf-8')
        question = json.dumps({
            "title": "piture.png",
            "description": "给你一张图片",
            "image": img_b64_str,
            "error_code": 0,
        })
        payload = {
            "model": "paddleocr",
            "messages": [{
                "role": "user",
                "content": question
            }]
        }
        url = "http://localhost:40004/v1/chat/completions"
        headers = {'Content-Type': 'application/json'}
        # 发送POST请求
        response = requests.post(url, data=json.dumps(payload), headers=headers)
        chat_response = response.json()
        content = chat_response["choices"][0]["message"]["content"]
        content = json.loads(content)

        #print(content)
        img_b64_str = content["image"]
        image_data = base64.b64decode(img_b64_str)
        # 将二进制数据加载到 BytesIO 对象中，并从中打开图像
        image = Image.open(io.BytesIO(image_data))
        # 将图像转换为 RGB 模式（与第一种方法一致）
        image = image.convert("RGB")
        image.save("/home/whh/project/Xinhai/static/output_image.jpg")

        return self.name, content["description"]

    def step(self, **kwargs):
        image_url = kwargs.get("image_url", "")
        try:
            role, content = self.ocr_conversation(image_url)
        except Exception as e:
            role, content="",""
            print("ocr工具出错！")
        ocr_tool_answer = "OCR Tool对图片进行了OCR文字识别,识别出的文字内容如下:" + content
        prompt = self.Answer_Refactoring_Template.format(role_description=self.role_description,
                                                    user_question=self.user_question,
                                                    answer_template='{{' + self.answer_template[1:-1] + '}}',
                                                    ocr_tool_answer=ocr_tool_answer,
                                                    )

        role, content = self.complete_conversation(prompt)

        t = datetime.now()
        return XinHaiChatMessage(
            indexId='-1',
            content=content,
            senderId=self.name,
            username=self.name,
            role="user",
            date=t.strftime("%a %b %d %Y"),
            timestamp=t.strftime("%H:%M"),
        )
