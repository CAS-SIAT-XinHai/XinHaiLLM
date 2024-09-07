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

logger = logging.getLogger(__name__)


@register_agent(XinHaiArenaAgentTypes.LLM_AGENT)
class LLM_Agent(BaseAgent):
    agent_type = XinHaiArenaAgentTypes.LLM_AGENT

    def reset(self) -> None:
        pass
    def routing(self, candidate_agents: List[Self], **kwargs) -> XinHaiRoutingMessage:
        """Routing logic for agent"""
        targets = [a.agent_id for a in candidate_agents]
        # if len(self.allowed_routing_types) == 1:
        #     routing_message = XinHaiRoutingMessage(
        #         agent_id=self.agent_id,
        #         routing_type=self.allowed_routing_types[0],
        #         targets=targets,
        #         routing_prompt="Static Routing"
        #     )
        #     return routing_message
        # else:
        routing_prompt = self.get_routing_prompt(candidate_agents, **kwargs)
        routing_message = None
        while not routing_message:
            #这个路由没啥意义，就是只能返回给多模态
            data = {'method': '[Unicast]', 'target': [0]}
            logger.debug(data)
            try:
                targets = data["target"]
            except KeyError:
                continue

            if isinstance(data['target'], int):
                targets = [data['target']]

            routing_type = XinHaiRoutingType.from_str(data['method'])
            if self.agent_id not in targets and routing_type in self.allowed_routing_types:
                routing_message = XinHaiRoutingMessage(
                    agent_id=self.agent_id,
                    routing_type=routing_type,
                    targets=targets,
                    routing_prompt=routing_prompt
                )

        return routing_message
    def step(self, routing, agents, **kwargs):
        chat_summary = self.get_summary()
        chat_history = '\n'.join(self.get_history())
        prompt = self.prompt_template.format(chat_history=chat_history,
                                             user_question=self.user_question,
                                             role_description=self.role_description,
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
