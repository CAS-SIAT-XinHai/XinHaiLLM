"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Vimos Tan
"""
import logging
from typing import List

from xinhai.arena.agents import register_agent
from xinhai.arena.agents.base import BaseAgent
from xinhai.types.arena import XinHaiArenaAgentTypes
from xinhai.types.message import XinHaiChatMessage

logger = logging.getLogger(__name__)


@register_agent(XinHaiArenaAgentTypes.PROXY)
class ProxyAgent(BaseAgent):
    agent_type = XinHaiArenaAgentTypes.PROXY

    def reset(self) -> None:
        pass

    def routing(self, agent_descriptions):
        """
        For proxy agent, it passes user input directly to next agent.
        """
        pass

    def step(self, routing, agents):
        pass

    def update_memory(self, messages: List[XinHaiChatMessage]):
        pass
