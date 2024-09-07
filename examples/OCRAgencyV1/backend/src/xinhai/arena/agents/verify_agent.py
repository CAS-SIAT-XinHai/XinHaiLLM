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
import re
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


@register_agent(XinHaiArenaAgentTypes.VERIFY_AGENT)
class VERIFY_AGENT(BaseAgent):
    agent_type = XinHaiArenaAgentTypes.VERIFY_AGENT

    def __init__(self,name, agent_id, role_description, llm, api_key, api_base,
                 environment_id, controller_address, locale,
                 allowed_routing_types,Verify_prompt_template,user_question,answer_template):
        routing_prompt_template = " "
        summary_prompt_template = ""
        prompt_template = ""
        super().__init__(name, agent_id, role_description, llm, api_key, api_base,
                 environment_id, controller_address, locale,
                 allowed_routing_types, routing_prompt_template, summary_prompt_template, prompt_template)
        self.Verify_prompt_template=Verify_prompt_template
        self.user_question=user_question
        self.answer_template=answer_template
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
        answer_form = "The generated response should be enclosed by [Response] and [End of Response]."
        messages = [{
            "role": "user",
            "content": prompt+"\n\n" + answer_form,
        }]
        format_pattern=re.compile(r"\[?Response]?([\s\S]+)\[?End of Response]?")

        while True:
            logger.debug(messages)
            chat_response = self.chat_completion(self.client, model=self.llm, agent_id=self.agent_id, messages=messages)
            if chat_response:
                rr = format_pattern.findall(chat_response)
                #提取出json部分
                if rr:
                    response=rr[0].strip()
                    json_pattern = re.compile(r'\{.*?\}', re.DOTALL)  # re.DOTALL 允许 . 匹配换行符
                    response = json_pattern.search(response)
                    if response:
                        json_str = response.group(0)  # 提取 JSON 字符串
                        try:
                            response_data = json.loads(json_str)  # 将 JSON 字符串转换为字典
                            print(response_data)
                            break
                        except json.JSONDecodeError as e:
                            print(f"JSON 解码失败: {e}")

        response_data = json.dumps(response_data, ensure_ascii=False, )
        return self.name, response_data

    def step(self, **kwargs,):

        ocr_agent_answer=kwargs.get("ocr_agent_answer","")
        mllm_agent_answer=kwargs.get("mllm_agent_answer","")
        prompt = self.Verify_prompt_template.format(role_description=self.role_description,
                                                    user_question=self.user_question,
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
