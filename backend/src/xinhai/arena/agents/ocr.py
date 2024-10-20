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

import requests

from xinhai.arena.agents import register_agent, BaseAgent
from xinhai.types.arena import XinHaiArenaAgentTypes
from xinhai.types.message import XinHaiChatMessage

logger = logging.getLogger(__name__)


@register_agent(XinHaiArenaAgentTypes.OCR_AGENT)
class OCRAgent(BaseAgent):
    agent_type = XinHaiArenaAgentTypes.OCR_AGENT

    def __init__(self, name, agent_id, role_description, llm, api_key, api_base,
                 prompt_template,
                 environment_id, controller_address, locale,
                 allowed_routing_types, answer_template):
        super().__init__(name, agent_id, role_description, llm, api_key, api_base,
                         routing_prompt_template='',
                         summary_prompt_template='',
                         prompt_template=prompt_template,
                         environment_id=environment_id,
                         controller_address=controller_address,
                         locale=locale,
                         allowed_routing_types=allowed_routing_types)
        self.answer_template = answer_template

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

    def step(self, routing, agents, **kwargs):
        image_url = kwargs.get("image_url", "")
        user_question = kwargs.get("user_question", "")
        try:
            ocr_ret = self.ocr_conversation(image_url)
        except Exception as e:
            ocr_ret = {'description': ''}
            logger.warning("ocr工具出错！")

        prompt = self.prompt_template.format(role_description=self.role_description,
                                             user_question=user_question,
                                             answer_template='{' + self.answer_template[1:-1] + '}',
                                             ocr_tool_answer=ocr_ret['description'])
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
