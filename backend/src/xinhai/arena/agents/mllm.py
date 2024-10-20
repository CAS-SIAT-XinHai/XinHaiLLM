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
from datetime import datetime

from xinhai.arena.agents import register_agent, BaseAgent
from xinhai.types.arena import XinHaiArenaAgentTypes
from xinhai.types.message import XinHaiChatMessage

logger = logging.getLogger(__name__)


@register_agent(XinHaiArenaAgentTypes.MLLM_AGENT)
class MLLMAgent(BaseAgent):
    agent_type = XinHaiArenaAgentTypes.MLLM_AGENT

    def __init__(self, name, agent_id, role_description, llm,
                 environment_id, controller_address, locale,
                 allowed_routing_types,
                 api_key="EMPTY",
                 api_base=None):
        routing_prompt_template = " "
        summary_prompt_template = ""
        prompt_template = ""
        super().__init__(name, agent_id, role_description, llm,
                         routing_prompt_template, summary_prompt_template, prompt_template,
                         environment_id, controller_address, locale,
                         allowed_routing_types,
                         api_key=api_key,
                         api_base=api_base)

    def reset(self) -> None:
        pass

    def complete_conversation(self, prompt, image_url=None, num_retries=5):

        # 采用多模态的message
        txt = MultimodalInputItem(type="text", text=prompt)
        pie = MultimodalInputItem(type="image_url",
                                  image_url=ImageURL(url=image_url))
        messages = [{
            "role": "user",
            "content": [txt.dict(), pie.dict()]
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

        return self.name, formatted_string

    def step(self, routing, agents, **kwargs):
        # chat_summary = self.get_summary()
        # chat_history = '\n'.join(self.get_history())
        user_question = kwargs.get("user_question", "")
        image_url = kwargs.get("image_url", "")
        # 统一一个发送的格式。
        prompt = self.mllm_prompt_template.format(
            role_description=self.role_description,
            user_question=user_question,
            answer_template=self.answer_template,
        )

        role, content = self.complete_conversation(prompt, image_url)

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
