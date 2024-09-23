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

        proxy_agents = [a for a in self.agents if a.agent_type == XinHaiArenaAgentTypes.PROXY]
        assert len(proxy_agents) == 1
        proxy_agent = proxy_agents[0]

        proxy_neighbors = [self.agents[n] for n in self.topology.digraph.neighbors(proxy_agent.agent_id)]
        assert len(proxy_neighbors) == 1
        start_agent = proxy_neighbors[0]

        role_mapping = {
            Role.USER: str(proxy_agent.agent_id),
            Role.ASSISTANT: str(start_agent.agent_id),
        }
        proxy_agent.memory.short_term_memory.messages = XinHaiChatMessage.from_chat(input_messages, role_mapping)
        start_agent.memory.short_term_memory.messages = XinHaiChatMessage.from_chat(input_messages, role_mapping)

        agent_queue = [start_agent]
        while agent_queue:
            agent = agent_queue.pop(0)
            candidate_agents = [self.agents[n] for n in self.topology.digraph.neighbors(agent.agent_id)]
            routing_message = agent.routing(candidate_agents)
            logger.debug(routing_message)
            targets = [self.agents[n] for n in routing_message.targets if
                       self.topology.digraph.has_edge(agent.agent_id, n)]
            if targets:
                agent_queue.extend(targets)
                targets_descriptions = "\n".join(
                    [f"{n.agent_id}: {n.role_description}" for n in targets])
                message = agent.step(
                    routing=routing_message.routing_type.routing_name,
                    agents=targets_descriptions
                )
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
