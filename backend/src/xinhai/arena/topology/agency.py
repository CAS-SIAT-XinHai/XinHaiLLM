"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Vimos Tan
"""
import logging

import networkx as nx

from llamafactory.api.protocol import Role
from xinhai.arena.topology import register_topology, BaseTopology
from xinhai.types.arena import XinHaiArenaAgentTypes
from xinhai.types.message import XinHaiChatMessage
from xinhai.types.routing import XinHaiRoutingType

logger = logging.getLogger(__name__)


@register_topology("agency")
class AgencyTopology(BaseTopology):
    def __init__(self, name, graph: nx.DiGraph, nodes=None, start=0, max_turns=0):
        super().__init__(name, graph, nodes=nodes, start=start, max_turns=max_turns)

    def __call__(self, agents, input_messages, *args, **kwargs):
        """Run one step of the environment"""
        logger.info(input_messages)

        proxy_agents = [a for a in agents if a.agent_type == XinHaiArenaAgentTypes.PROXY]
        assert len(proxy_agents) == 1
        proxy_agent = proxy_agents[0]

        proxy_neighbors = [agents[n] for n in self.digraph.neighbors(proxy_agent.agent_id)]
        assert len(proxy_neighbors) == 1
        start_agent = proxy_neighbors[self.start]

        role_mapping = {
            Role.USER: str(proxy_agent.agent_id),
            Role.ASSISTANT: str(start_agent.agent_id),
        }
        proxy_agent.memory.short_term_memory.messages = XinHaiChatMessage.from_chat(input_messages, role_mapping)
        start_agent.memory.short_term_memory.messages = XinHaiChatMessage.from_chat(input_messages, role_mapping)

        agent_queue = [start_agent]
        end_casted = set()
        while agent_queue:
            agent = agent_queue.pop(0)
            candidate_agents = [agents[n] for n in self.digraph.neighbors(agent.agent_id)]
            routing_message = agent.routing(candidate_agents)
            logger.debug(routing_message)
            if routing_message.routing_type == XinHaiRoutingType.END_CAST:
                end_casted.add(agent.agent_id)

            targets = [agents[n] for n in routing_message.targets if
                       all([self.digraph.has_edge(agent.agent_id, n),
                            n not in agent_queue,
                            n not in end_casted])]
            if targets:
                agent_queue.extend(targets)
                targets_descriptions = "\n".join(
                    [f"{n.agent_id}: {n.role_description}" for n in targets])
                message = agent.step(
                    routing=routing_message.routing_type.routing_name,
                    agents=targets_descriptions
                )

                yield targets, message
                agent.update_memory([message])
