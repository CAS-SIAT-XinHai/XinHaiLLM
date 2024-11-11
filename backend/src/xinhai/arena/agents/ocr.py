"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors:wuhaihong
Date: 2024-07-19 17:22:57
LastEditTime: 2024-07-19 17:28:20
"""
import json
import logging
from datetime import datetime
from typing import List

import requests

from xinhai.arena.agents import register_agent, BaseAgent
from xinhai.arena.agents.simple import SimpleAgent
from xinhai.types.arena import XinHaiArenaAgentTypes
from xinhai.types.message import XinHaiChatMessage
from xinhai.types.routing import XinHaiRoutingMessage

logger = logging.getLogger(__name__)


@register_agent(XinHaiArenaAgentTypes.OCR)
class OCRAgent(SimpleAgent):
    agent_type = XinHaiArenaAgentTypes.OCR

    def reset(self) -> None:
        pass

    def ocr_conversation(self, image_url=None, num_retries=5):
        # answer_form = "The generated response should be enclosed by [Response] and [End of Response]."
        payload = {
            "model": "paddleocr",
            "image": image_url
        }
        url = f"{self.controller_address}/api/ocr-image"
        headers = {'Content-Type': 'application/json'}

        try:
            # 发送POST请求
            r = requests.post(url, data=json.dumps(payload), headers=headers)
            if r.status_code != 200:
                logger.error(f"Get status fails: {self.controller_address}, {r}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Get status fails: {self.controller_address}, {e}")

        logger.debug(r.json())

        return r.json()

    def step(
            self,
            routing_message_in: XinHaiRoutingMessage,
            routing_message_out: XinHaiRoutingMessage,
            target_agents: List[BaseAgent], **kwargs
    ):
        chat_summary = self.get_summary()
        chat_history, chat_files = self.get_history(target_agents)
        target_agent_names = ", ".join([f"Agent-{n.agent_id} {n.name}" for n in target_agents])

        try:
            ocr_ret = self.ocr_conversation(chat_files[0].url)
        except Exception as e:
            ocr_ret = {'description': ''}
            logger.warning("ocr工具出错！")

        prompt = self.prompt_template.format(chat_history='\n'.join(chat_history),
                                             chat_summary=chat_summary,
                                             role_description=self.role_description,
                                             routing_type=routing_message_out.routing_type.routing_name,
                                             target_agent_names=target_agent_names,
                                             ocr_tool_answer=ocr_ret['description'])
        role, content = self.complete_conversation(prompt)

        t = datetime.now()
        return XinHaiChatMessage(
            content=content,
            senderId=self.name,
            username=self.name,
            date=t.strftime("%a %b %d %Y"),
            timestamp=t.strftime("%H:%M"),
        )
