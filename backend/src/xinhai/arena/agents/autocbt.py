"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Ancheng Xu
         Vimos Tan
Date: 2024-07-19 17:22:57
LastEditTime: 2024-10-19 17:28:20
"""
import logging
import random
import time
import uuid
from datetime import datetime
from typing import List

from xinhai.arena.agents import register_agent, BaseAgent
from xinhai.types.i18n import XinHaiI18NLocales
from xinhai.types.message import XinHaiChatMessage
from xinhai.types.routing import XinHaiRoutingMessage

logger = logging.getLogger(__name__)


@register_agent("autocbt")
class AutoCBTAgent(BaseAgent):

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
                                             chat_question=self.get_history()[0].replace('\n', '') if len(
                                                 self.get_history()) > 0 else "None",
                                             draft_response=self.get_history()[1].replace('\n', '') if len(
                                                 self.get_history()) > 1 else "None",
                                             revise_of_draft='\n'.join(
                                                 ["【" + his.replace('\n', '') + "】" for his in self.get_history() if
                                                  supervisor_key_word in his]) if len(
                                                 self.get_history()) > 2 else "None",
                                             routing=routing,
                                             agents=agents,
                                             agent_name=self.name)
        role, content = self.complete_conversation(prompt)

        t = datetime.now()

        return XinHaiChatMessage(
            content=content,
            senderId=self.name,
            username=self.name,
            date=t.strftime("%a %b %d %Y"),
            timestamp=t.strftime("%H:%M"),
        )

    def routing(self, candidate_agents: List[BaseAgent], **kwargs) -> XinHaiRoutingMessage:
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
            time.sleep(sleep_time)  # siliconflow有api限速

    def complete_conversation(self, prompt, num_retries=5):
        messages = [{
            "role": "user",
            "content": prompt + "\n\n" + self.format_prompt,
        }]

        repeate_time = 3
        final_response = ""
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
