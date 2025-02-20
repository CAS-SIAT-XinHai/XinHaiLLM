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
            chat_files = [f for i, message in enumerate(self.messages[::-1]) for f in message.files]
            for turn_id, (targets, message) in enumerate(topology(self.agents, input_messages, chat_files=chat_files)):
                for n in [message.senderId] + message.receiverIds:
                    a = self.agents[int(n)]
                    a.update_memory([message])
                    self.messages.append(message)

                for a in targets:
                    if a.agent_type == XinHaiArenaAgentTypes.PROXY:
                        return message

                if turn_id > topology.max_turns:
                    logger.warning(f"Reached maximum number of {topology.max_turns} turns!")
                    chat_files = [f for i, message in enumerate(self.messages[::-1]) for f in message.files]
                    message = topology.fast_response(self.agents, input_messages, chat_files=chat_files)
                    self.messages.append(message)
                    return message

    def reset(self) -> None:
        """Reset the environment"""
        self.cnt_turn = 0
        for agent in self.agents:
            agent.reset()

    def is_done(self) -> bool:
        """Check if the environment is done"""
        return self.cnt_turn >= self.max_turns
