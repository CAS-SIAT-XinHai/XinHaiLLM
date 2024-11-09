"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors:wuhaihong
Date: 2024-07-19 17:22:57
LastEditTime: 2024-07-19 17:28:20
"""
import logging
import re
import uuid
from datetime import datetime
from typing import List

from openai import OpenAI

from xinhai.arena.agents import register_agent, BaseAgent
from xinhai.arena.agents.simple import SimpleAgent
from xinhai.types.arena import XinHaiArenaAgentTypes, XinHaiArenaLLMConfig
from xinhai.types.message import XinHaiChatMessage, MultimodalInputItem, ImageURL
from xinhai.types.routing import XinHaiRoutingMessage

logger = logging.getLogger(__name__)


@register_agent(XinHaiArenaAgentTypes.MLLM)
class MLLMAgent(SimpleAgent):
    agent_type = XinHaiArenaAgentTypes.MLLM

    def __init__(self, name, agent_id, role_description, llm, mllm,
                 routing_prompt_template, summary_prompt_template, prompt_template,
                 environment_id, controller_address, locale,
                 allowed_routing_types):
        super().__init__(name, agent_id, role_description, llm,
                         routing_prompt_template, summary_prompt_template, prompt_template,
                         environment_id, controller_address, locale,
                         allowed_routing_types)
        self.mllm: XinHaiArenaLLMConfig = XinHaiArenaLLMConfig.from_config(mllm, controller_address)
        self.mllm_client = OpenAI(
            api_key=self.mllm.api_key,
            base_url=self.mllm.api_base,
        )

    def reset(self) -> None:
        pass

    def complete_mllm_conversation(self, mllm_prompt, image_url, num_retries=5):
        assert image_url is not None

        # 采用多模态的message
        txt = MultimodalInputItem(type="text", text=mllm_prompt)
        pie = MultimodalInputItem(type="image_url", image_url=ImageURL(url=image_url))
        messages = [{
            "role": "user",
            "content": [
                txt.model_dump(),
                pie.model_dump()
            ]
        }]
        format_pattern = re.compile(r'\{(?:\s*"(.*?)"\s*:\s*"(.*?)"\s*,?)*\}')

        attempts = 0  # 初始化计数器
        max_attempts = 5  # 最大尝试次数
        rr = []
        chat_response = "no answer"
        while attempts < max_attempts:  # 设置循环条件为尝试次数不超过 5 次
            logger.debug(messages)
            chat_response = self.chat_completion(self.client, model=self.llm, agent_id=self.agent_id, messages=messages)
            if chat_response:
                rr = format_pattern.findall(chat_response)
                if rr:
                    break
            attempts += 1
        if rr == []:
            rr = ['({},{})'.format("key", chat_response)]

        last_match = rr[-1]
        formatted_parts = []
        for i in range(0, len(last_match), 2):
            try:
                key = last_match[i]
                value = last_match[i + 1]
                if key and value:  # 确保 key 和 value 都存在
                    formatted_parts.append('"{}":"{}"'.format(key, value))
            except Exception as e:
                formatted_parts.append('"{}":"{}"'.format("key", "value"))
        # 将这些格式化后的部分组合成最终字符串
        formatted_string = '{' + ','.join(formatted_parts) + '}'

        return formatted_string

    def step(
            self,
            routing_message_in: XinHaiRoutingMessage,
            routing_message_out: XinHaiRoutingMessage,
            target_agents: List[BaseAgent], **kwargs
    ):
        chat_summary = self.get_summary()
        chat_history, chat_files = self.get_history(target_agents)
        target_agent_names = ", ".join([f"Agent-{n.agent_id} {n.name}" for n in target_agents])

        prompt = self.prompt_template.format(chat_history='\n'.join(chat_history),
                                             chat_summary=chat_summary,
                                             role_description=self.role_description,
                                             routing_type=routing_message_out.routing_type.routing_name,
                                             target_agent_names=target_agent_names)
        role, mllm_prompt = self.complete_conversation(prompt)

        content = self.complete_mllm_conversation(mllm_prompt, image_url=chat_files[0].url)

        t = datetime.now()

        return XinHaiChatMessage(
            id=uuid.uuid4().hex,
            indexId='-1',
            content=content,
            senderId=str(self.agent_id),
            receiverIds=[str(a.agent_id) for a in target_agents],
            username=self.name,
            role="user",
            date=t.strftime("%a %b %d %Y"),
            timestamp=t.strftime("%H:%M"),
            files=chat_files
        )
