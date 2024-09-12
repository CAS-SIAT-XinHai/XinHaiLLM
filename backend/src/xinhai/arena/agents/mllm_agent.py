"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Vimos Tan
         yangdi
Date: 2024-07-19 17:22:57
LastEditTime: 2024-07-19 17:28:20
"""
import json
import logging
from datetime import datetime
from typing import List
import re
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
from pydantic import BaseModel, Field
from typing import Optional
from typing_extensions import Literal
class ImageURL(BaseModel):
    url: str
class MultimodalInputItem(BaseModel):
    type: Literal["text", "image_url"]
    text: Optional[str] = None
    image_url: Optional[ImageURL] = None


@register_agent(XinHaiArenaAgentTypes.MLLM_AGENT)
class MLLM_AGENT(BaseAgent):
    agent_type = XinHaiArenaAgentTypes.MLLM_AGENT

    def __init__(self, name, agent_id, role_description, llm, api_key, api_base,
                 environment_id, controller_address, locale,
                 allowed_routing_types, mllm_prompt_template, answer_template):
        routing_prompt_template = " "
        summary_prompt_template = ""
        prompt_template = ""
        super().__init__(name, agent_id, role_description, llm, api_key, api_base,
                 routing_prompt_template, summary_prompt_template, prompt_template,
                 environment_id, controller_address, locale,
                 allowed_routing_types)
        self.mllm_prompt_template = mllm_prompt_template

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

    def complete_conversation(self, prompt,image_url=None, num_retries=5):

        #采用多模态的message
        txt = MultimodalInputItem(type="text", text=prompt)
        pie = MultimodalInputItem(type="image_url",
                                  image_url=ImageURL(url=image_url))

        messages =[{
        "role": "user",
        "content": [txt.dict(),pie.dict()]
    }]
        format_pattern = re.compile(r'\{(?:\s*"(.*?)"\s*:\s*"(.*?)"\s*,?)*\}')

        while True:
            logger.debug(messages)
            chat_response = self.chat_completion(self.client, model=self.llm, agent_id=self.agent_id, messages=messages)
            if chat_response:
                rr = format_pattern.findall(chat_response)
                if rr:
                    break
        last_match = rr[-1]

        # last_match 是一个元组，包含多个键值对，如 ('对象1', '值1', '对象2', '值2', ...)
        # 需要将这些元素重新组合成所需的字符串格式

        formatted_parts = []
        for i in range(0, len(last_match), 2):
            key = last_match[i]
            value = last_match[i + 1]
            if key and value:  # 确保 key 和 value 都存在
                formatted_parts.append('"{}":"{}"'.format(key, value))

        # 将这些格式化后的部分组合成最终字符串
        formatted_string = '{' + ','.join(formatted_parts) + '}'

        return self.name, formatted_string
    def step(self,routing,agents,**kwargs):
        #chat_summary = self.get_summary()
        #chat_history = '\n'.join(self.get_history())
        user_question=kwargs.get("user_question","")
        image_url=kwargs.get("image_url", "")
        # 统一一个发送的格式。
        prompt = self.mllm_prompt_template.format(
                                             role_description=self.role_description,
                                             user_question=user_question,
                                             answer_template=self.answer_template,
                                             )

        role, content = self.complete_conversation(prompt,image_url)

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
