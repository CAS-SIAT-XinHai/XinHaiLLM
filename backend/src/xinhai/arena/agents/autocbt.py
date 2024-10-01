import logging
from datetime import datetime
import random
from xinhai.arena.agents import register_agent
from xinhai.arena.agents.base import BaseAgent
from xinhai.types.memory import XinHaiMemory, XinHaiShortTermMemory, XinHaiLongTermMemory, XinHaiChatSummary
from xinhai.types.message import XinHaiChatMessage
from xinhai.types.routing import XinHaiRoutingMessage, XinHaiRoutingType
from xinhai.types.prompt import XinHaiPromptType
from xinhai.types.storage import XinHaiFetchMemoryResponse, XinHaiStoreMemoryRequest, XinHaiFetchMemoryRequest
from xinhai.types.i18n import XinHaiI18NLocales
import sys
import requests
import json
import time
from openai import OpenAI, OpenAIError
from typing import List
if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self
import time
logger = logging.getLogger(__name__)


@register_agent("autocbt")
class AutoCBTAgent(BaseAgent):
    def __init__(self, name, agent_id, role_description, llm, api_key, api_base, routing_prompt_template, summary_prompt_template, prompt_template, environment_id, controller_address, locale, allowed_routing_types, static_routing=False, id_template=None, max_retries=4, summary_chunk_size=16):
        super().__init__(name, agent_id, role_description, llm, api_key, api_base, routing_prompt_template, summary_prompt_template, prompt_template, environment_id, controller_address, locale, allowed_routing_types, static_routing, id_template, max_retries)
        self.summary_chunk_size = summary_chunk_size

    def reset(self) -> None:
        pass

    def step(self, routing, agents, **kwargs):
        chat_summary = self.get_summary()
        chat_history = '\n'.join(self.get_history())

        supervisor_key_word = "督导师" if self.locale is XinHaiI18NLocales.CHINESE else "supervisor"

        prompt = self.prompt_template.format(chat_history=chat_history,
                                             chat_summary=chat_summary,
                                             role_description=self.role_description,
                                             original_question_of_user=self.environment.agents[0].role_description,
                                             chat_question=self.get_history()[0].replace('\n', '') if len(self.get_history()) > 0 else "None",
                                             draft_response=self.get_history()[1].replace('\n', '') if len(self.get_history()) > 1 else "None",
                                             revise_of_draft='\n'.join(["【"+his.replace('\n', '')+"】" for his in self.get_history() if supervisor_key_word in his]) if len(self.get_history()) > 2 else "None",
                                             routing=routing,
                                             agents=agents,
                                             agent_name=self.name)
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

    def routing(self, candidate_agents: List[Self], **kwargs) -> XinHaiRoutingMessage:
        while True:
            routing_message = super().routing(candidate_agents=candidate_agents, **kwargs)
            targets = routing_message.targets
            # 咨询者和督导师同时出现在咨询师的路由里，咨询师需要重新路由
            if 0 in targets and len(targets) > 1:
                continue
            return routing_message

    def chat_completion(self, client, model, agent_id, messages):
        sleep_time = 60 + random.randint(0, 60)
        while True:
            content = super().chat_completion(client, model, agent_id, messages)
            if (content is not None) and len(content) > 0:
                return content
            print(f"sleep {sleep_time}s to avoid limit speed of api")
            time.sleep(sleep_time) # siliconflow有api限速

    def complete_conversation(self, prompt, num_retries=5):
        messages = [{
            "role": "user",
            "content": prompt + "\n\n" + self.format_prompt,
        }]

        repeate_time=3
        final_response=""
        while True and repeate_time > 0:
            logger.debug(messages)
            chat_response = self.chat_completion(self.client, model=self.llm, agent_id=self.agent_id, messages=messages)
            if chat_response:
                repeate_time -= 1
                final_response = chat_response
                rr = self.format_pattern.findall(chat_response)
                if rr is not None and len(rr) > 0 and len(rr[0].strip()) > 0:
                    final_response = rr[0].strip()
                    break
        return self.name, final_response