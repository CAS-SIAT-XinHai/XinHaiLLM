"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Vimos Tan
         yangdi
Date: 2024-07-19 17:22:57
LastEditTime: 2024-07-19 17:28:20
"""
import logging
from datetime import datetime

from xinhai.arena.agents import register_agent
from xinhai.arena.agents.base import BaseAgent
from xinhai.arena.topology.base import BaseTopology
from xinhai.types.arena import XinHaiArenaAgentTypes
from xinhai.types.message import XinHaiChatMessage
from xinhai.types.routing import XinHaiRoutingMessage, XinHaiRoutingType

logger = logging.getLogger(__name__)


@register_agent(XinHaiArenaAgentTypes.SIMPLE)
class SimpleAgent(BaseAgent):
    agent_type = XinHaiArenaAgentTypes.SIMPLE

    def reset(self) -> None:
        pass

    def step(self, routing, agents, **kwargs):
        chat_summary = self.get_summary()
        chat_history = '\n'.join(self.get_history())
        prompt = self.prompt_template.format(chat_history=chat_history,
                                             chat_summary=chat_summary,
                                             role_description=self.role_description,
                                             routing=routing,
                                             agents=agents)
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
