'''
Descripttion: 
version: 1.0.0
Author: yangdi
Date: 2024-07-24 10:04:36
LastEditTime: 2024-07-29 14:30:17
'''
"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Vimos Tan
"""
import logging

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
        agent_queue = [self.agents[0]]
        while agent_queue:
            agent = agent_queue.pop(0)

            agent_descriptions = "\n".join(
                [f"{n}: {self.agents[n].role_description}" for n in self.topology.digraph.neighbors(agent.agent_id)])
            while True:
                data = agent.routing(agent_descriptions)
                logger.debug(data)

                targets = data["target"]
                if isinstance(data['target'], int):
                    targets = [data['target']]
                targets = [self.agents[n] for n in targets if self.topology.digraph.has_edge(n, agent.agent_id) and n != agent.agent_id]
                
                if targets:
                    break

            if targets:
                agent_queue.extend(targets)

                targets_descriptions = "\n".join(
                    [f"{n.agent_id}: {n.role_description}" for n in targets])
                message = agent.step(routing=data["method"], agents=targets_descriptions)
                agent.update_memory([message])

                for a in targets:
                    a.update_memory([message])

        self.cnt_turn += 1

    def reset(self) -> None:
        """Reset the environment"""
        self.cnt_turn = 0
        for agent in self.agents:
            agent.reset()

    def is_done(self) -> bool:
        """Check if the environment is done"""
        return self.cnt_turn >= self.max_turns
