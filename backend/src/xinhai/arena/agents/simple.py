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
from typing import List

from xinhai.arena.agents import register_agent, BaseAgent
from xinhai.types.arena import XinHaiArenaAgentTypes
from xinhai.types.message import XinHaiChatMessage
from xinhai.types.routing import XinHaiRoutingMessage

logger = logging.getLogger(__name__)


@register_agent(XinHaiArenaAgentTypes.SIMPLE)
class SimpleAgent(BaseAgent):
    agent_type = XinHaiArenaAgentTypes.SIMPLE

    def reset(self) -> None:
        pass

    def step(
            self,
            routing_message_in: XinHaiRoutingMessage,
            routing_message_out: XinHaiRoutingMessage,
            target_agents: List[BaseAgent], **kwargs
    ):
        chat_summary = self.get_summary()
        chat_history = '\n'.join(self.get_history())
        target_agent_names = ", ".join([f"Agent-{n.agent_id} {n.name}" for n in target_agents])

        prompt = self.prompt_template.format(chat_history=chat_history,
                                             chat_summary=chat_summary,
                                             role_description=self.role_description,
                                             routing_type=routing_message_out.routing_type.routing_name,
                                             target_agent_names=target_agent_names)
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
