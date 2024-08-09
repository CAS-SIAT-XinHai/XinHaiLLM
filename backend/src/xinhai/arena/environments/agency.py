"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Vimos Tan
"""
import logging

from llamafactory.api.protocol import Role
from xinhai.arena.environments import register_environment
from xinhai.arena.environments.base import BaseEnvironment
from xinhai.types.arena import XinHaiArenaAgentTypes, XinHaiArenaEnvironmentTypes
from xinhai.types.message import XinHaiChatMessage

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

    async def step(self,
                   input_messages,
                   system,
                   tools,
                   do_sample,
                   temperature,
                   top_p,
                   max_new_tokens,
                   num_return_sequences):
        """Run one step of the environment"""
        logger.debug(input_messages)
        print(input_messages)
        role_mapping = {
            Role.USER: str(self.agents[0].agent_id),
            Role.ASSISTANT: str(self.agents[1].agent_id),
        }
        self.agents[0].memory.short_term_memory.messages = XinHaiChatMessage.from_chat(input_messages, role_mapping)
        self.agents[1].memory.short_term_memory.messages = XinHaiChatMessage.from_chat(input_messages, role_mapping)

        agent_queue = [self.agents[1]]
        while agent_queue:
            agent = agent_queue.pop(0)

            agent_descriptions = "\n".join(
                [f"{n}: {self.agents[n].role_description}" for n in self.topology.digraph.neighbors(agent.agent_id)])

            data = agent.routing(agent_descriptions)
            logger.debug(data)

            targets = data["target"]
            if isinstance(data['target'], int):
                targets = [data['target']]
            targets = [self.agents[n] for n in targets if self.topology.digraph.has_edge(n, agent.agent_id)]

            if targets:
                agent_queue.extend(targets)

                targets_descriptions = "\n".join(
                    [f"{n.agent_id}: {n.role_description}" for n in targets])
                message = agent.step(routing=data["method"], agents=targets_descriptions)
                agent.update_memory([message])

                for a in targets:
                    if a.agent_type == XinHaiArenaAgentTypes.PROXY:
                        agent_queue = []
                        break
                    a.update_memory([message])

        self.cnt_turn += 1
        return message

    def reset(self) -> None:
        """Reset the environment"""
        self.cnt_turn = 0
        for agent in self.agents:
            agent.reset()

    def is_done(self) -> bool:
        """Check if the environment is done"""
        return self.cnt_turn >= self.max_turns
