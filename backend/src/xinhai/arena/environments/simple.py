"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Vimos Tan
"""
import asyncio
import logging

import networkx as nx

from xinhai.arena.environments import register_environment
from xinhai.arena.environments.base import BaseEnvironment

logger = logging.getLogger(__name__)


@register_environment("simple")
class SimpleEnvironment(BaseEnvironment):
    """
    A basic environment implementing the logic of conversation.

    Args:
        agents: List of agents
        llm:
        max_turns: Maximum number of turns
        cnt_turn: Current turn number
    """

    async def step(self):
        """Run one step of the environment"""
        for agent in self.agents:
            memories = [agent.step()]
            [agent.update_memory(memories) for agent in self.agents]

        # for tail, head in nx.edge_dfs(self.topology.digraph, self.topology.nodes):
        #     messages = self.agents[tail].messages[head]
        #     m = self.agents[tail].chat_completion(head, messages=messages)
        #     self.agents[tail].update_memory(head, m)

        # All agent update memory based on current observation
        # await asyncio.gather(*[agent.astep("") for agent in self.agents])

        # for node in self.topology.nodes:
        #     self.agents[node].step()

        self.cnt_turn += 1

    def reset(self) -> None:
        """Reset the environment"""
        self.cnt_turn = 0
        for agent in self.agents:
            agent.reset()

    def is_done(self) -> bool:
        """Check if the environment is done"""
        return self.cnt_turn >= self.max_turns
