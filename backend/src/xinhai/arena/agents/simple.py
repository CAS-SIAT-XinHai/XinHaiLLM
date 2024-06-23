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
        dialogue_context = []
        for i, (agent_name, response) in enumerate(self.memory):
            dialogue_context.append(f"{agent_name}: {response}")
        return dialogue_context

    def step(self):
        chat_history = '\n'.join(self.get_history())
        prompt = self.prompt_template.format(chat_history=chat_history, role_description=self.role_description)
        return self.complete_conversation(prompt)