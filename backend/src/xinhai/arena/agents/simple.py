'''
Descripttion: 
version: 1.0.0
Author: yangdi
Date: 2024-07-19 17:22:57
LastEditTime: 2024-07-19 17:28:20
'''
"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Vimos Tan
"""

import logging

from xinhai.arena.agents import register_agent
from xinhai.arena.agents.base import BaseAgent

logger = logging.getLogger(__name__)


@register_agent("simple")
class SimpleAgent(BaseAgent):

    def reset(self) -> None:
        pass

    def get_history(self):
        memory = self.retrieve_memory()
        dialogue_context = []
        for i, (agent_name, response) in enumerate(zip(memory["short_term_metadatas"], memory["short_term_documents"])):
            dialogue_context.append(f"{agent_name['source']}: {response}")
        if len(dialogue_context) > 10:
            dialogue_context = memory["summary_dialogues"][-1] + dialogue_context[-10:]
        return dialogue_context

    def routing(self, agent_descriptions):
        chat_history = '\n'.join(self.get_history())
        routing_prompt = self.routing_prompt_template.format(agent_name=self.name,
                                                             role_description=self.role_description,
                                                             chat_history=chat_history,
                                                             agent_descriptions=agent_descriptions)
        return self.prompt_for_routing(routing_prompt)

    def step(self, routing, agents):
        chat_history = '\n'.join(self.get_history())
        prompt = self.prompt_template.format(chat_history=chat_history,
                                             role_description=self.role_description,
                                             routing=routing,
                                             agents=agents)
        return self.complete_conversation(prompt)