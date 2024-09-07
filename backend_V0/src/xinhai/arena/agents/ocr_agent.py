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

@register_agent(XinHaiArenaAgentTypes.OCR_AGENT)
class OCR_AGENT(BaseAgent):
    agent_type = XinHaiArenaAgentTypes.OCR_AGENT

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
            content = ""
            if chat_response.choices[0].message.content:
                content += chat_response.choices[0].message.content
            if content.strip():
                logger.info(f"Get response from Agent-{agent_id}: {content}")
                return content.strip()
            else:
                usage = "chat_response.usage"
                logger.error(f"Error response from Agent-{agent_id}: {usage}")
        except OpenAIError as e:
            # Handle all OpenAI API errors
            logger.warning("*****************************************")
            logger.warning(f"Error response from Agent-{agent_id}: {e}")

    def complete_conversation(self, prompt, image_url=None,num_retries=5):
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

    def routing(self, candidate_agents: List[Self], **kwargs) -> XinHaiRoutingMessage:
        """Routing logic for agent"""
        targets = [a.agent_id for a in candidate_agents]
        # if len(self.allowed_routing_types) == 1:
        #     routing_message = XinHaiRoutingMessage(
        #         agent_id=self.agent_id,
        #         routing_type=self.allowed_routing_types[0],
        #         targets=targets,
        #         routing_prompt="Static Routing"
        #     )
        #     return routing_message
        # else:
        routing_prompt = self.get_routing_prompt(candidate_agents, **kwargs)
        routing_message = None
        while not routing_message:
            #这个路由没啥意义，就是只能返回给多模态
            data = {'method': '[Unicast]', 'target': [0]}
            logger.debug(data)
            try:
                targets = data["target"]
            except KeyError:
                continue

            if isinstance(data['target'], int):
                targets = [data['target']]

            routing_type = XinHaiRoutingType.from_str(data['method'])
            if self.agent_id not in targets and routing_type in self.allowed_routing_types:
                routing_message = XinHaiRoutingMessage(
                    agent_id=self.agent_id,
                    routing_type=routing_type,
                    targets=targets,
                    routing_prompt=routing_prompt
                )

        return routing_message


    def step(self, routing, agents,image_url=None, **kwargs):
        # chat_summary = self.get_summary()
        # chat_history = '\n'.join(self.get_history())
        # prompt = self.prompt_template.format(chat_history=chat_history,
        #                                      chat_summary=chat_summary,
        #                                      role_description=self.role_description,
        #                                      routing=routing,
        #                                      agents=agents)
        prompt=""
        role, content = self.complete_conversation(prompt,image_url)
        content="Tool_Master对图片进行了OCR文字识别,识别出的文字内容如下:"+content
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
