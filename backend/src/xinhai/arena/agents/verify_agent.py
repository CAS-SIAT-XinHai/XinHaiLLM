"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: wuhaihong
Date: 2024-07-19 17:22:57
LastEditTime: 2024-07-19 17:28:20
"""
import logging
from datetime import datetime

from xinhai.arena.agents import register_agent, BaseAgent
from xinhai.types.arena import XinHaiArenaAgentTypes
from xinhai.types.message import XinHaiChatMessage

logger = logging.getLogger(__name__)


@register_agent(XinHaiArenaAgentTypes.VERIFY_AGENT)
class VERIFY_AGENT(BaseAgent):
    agent_type = XinHaiArenaAgentTypes.VERIFY_AGENT

    def __init__(self, name, agent_id, role_description, llm, api_key, api_base,
                 environment_id, controller_address, locale,
                 allowed_routing_types, Verify_prompt_template, answer_template):
        routing_prompt_template = ""
        summary_prompt_template = ""
        prompt_template = ""
        super().__init__(name, agent_id, role_description, llm, api_key, api_base,
                         routing_prompt_template, summary_prompt_template, prompt_template,
                         environment_id, controller_address, locale,
                         allowed_routing_types)

        self.Verify_prompt_template = Verify_prompt_template

        self.answer_template = answer_template

    def reset(self) -> None:
        pass

    def step(self, routing, agents, **kwargs, ):
        ocr_agent_answer = kwargs.get("ocr_agent_answer", "")
        mllm_agent_answer = kwargs.get("mllm_agent_answer", "")
        user_question = kwargs.get("user_question", "")
        prompt = self.Verify_prompt_template.format(role_description=self.role_description,
                                                    user_question=user_question,
                                                    answer_template=self.answer_template,
                                                    ocr_agent_answer=ocr_agent_answer,
                                                    mllm_agent_answer=mllm_agent_answer)

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
