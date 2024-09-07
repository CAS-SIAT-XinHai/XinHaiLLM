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

        while True:
            logger.debug(messages)
            chat_response = self.chat_completion(self.client, model=self.llm, agent_id=self.agent_id, messages=messages)
            if chat_response:
                rr = self.response_pattern.findall(chat_response)
                rr=[chat_response]
                if rr:
                    break

        return self.name, rr[0]
    def step(self, routing, agents,image_url=None,**kwargs):
        chat_summary = self.get_summary()
        chat_history = '\n'.join(self.get_history())

        # 统一一个发送的格式。
        prompt = self.prompt_template.format(chat_history=chat_history,
                                             user_question=self.user_question,
                                             role_description=self.role_description,
                                             agents=agents)


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
