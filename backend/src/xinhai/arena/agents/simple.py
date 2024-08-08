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
from xinhai.types.arena import XinHaiArenaAgentTypes
from xinhai.types.message import XinHaiChatMessage
from xinhai.types.routing import XinHaiRoutingMessage, XinHaiRoutingType

logger = logging.getLogger(__name__)


@register_agent(XinHaiArenaAgentTypes.SIMPLE)
class SimpleAgent(BaseAgent):
    agent_type = XinHaiArenaAgentTypes.SIMPLE

    def reset(self) -> None:
        pass

    def routing(self, agent_descriptions) -> XinHaiRoutingMessage:
        chat_summary = self.get_summary()
        chat_history = '\n'.join(self.get_history())
        # logger.debug("~~~~~~~~~~~~~~~~~~~~~~~~chat_history~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # logger.debug(f"{self.agent_id}: {chat_history}")
        # logger.debug("~~~~~~~~~~~~~~~~~~~~~~~~chat_history~~~~~~~~~~~~~~~~~~~~~~~~~~")
        routing_prompt = self.routing_prompt_template.format(agent_name=self.name,
                                                             role_description=self.role_description,
                                                             chat_summary=chat_summary,
                                                             chat_history=chat_history,
                                                             agent_descriptions=agent_descriptions,
                                                             routing_descriptions=XinHaiRoutingType.to_description(
                                                                 locale=self.locale
                                                             ))
        while True:
            data = self.prompt_for_routing(routing_prompt)
            logger.debug(data)
            targets = data["target"]
            if isinstance(data['target'], int):
                targets = [data['target']]

            if self.agent_id not in targets:
                break

        return XinHaiRoutingMessage(
            agent_id=self.agent_id,
            routing_type=XinHaiRoutingType.from_str(data['method']),
            targets=targets,
            routing_prompt=routing_prompt
        )

    def step(self, routing, agents):
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
