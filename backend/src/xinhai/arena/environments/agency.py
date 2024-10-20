"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Vimos Tan
"""
import logging

from xinhai.arena.environments import register_environment, BaseEnvironment
from xinhai.types.arena import XinHaiArenaEnvironmentTypes, XinHaiArenaAgentTypes

logger = logging.getLogger(__name__)


@register_environment(XinHaiArenaEnvironmentTypes.AGENCY)
class AgencyEnvironment(BaseEnvironment):
    """
    A basic environment implementing the logic of conversation.

    Args:
        agents: List of agents
        llm:
        max_turns: Maximum number of turns
        cnt_turn: Current turn number
    """

    async def step(self, input_messages, *args, **kwargs):
        """Run one step of the environment"""
        for topology in self.topologies:
            for turn_id, (targets, message) in enumerate(topology(self.agents, input_messages)):
                if turn_id > topology.max_turns:
                    break
                for a in targets:
                    a.update_memory([message])
                    if a.agent_type == XinHaiArenaAgentTypes.PROXY:
                        return message

    def reset(self) -> None:
        """Reset the environment"""
        self.cnt_turn = 0
        for agent in self.agents:
            agent.reset()

    def is_done(self) -> bool:
        """Check if the environment is done"""
        return self.cnt_turn >= self.max_turns
