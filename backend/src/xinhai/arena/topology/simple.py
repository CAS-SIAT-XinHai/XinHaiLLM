"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Vimos Tan
"""
import logging

from xinhai.arena.topology import register_topology, BaseTopology
from xinhai.types.routing import XinHaiRoutingType

logger = logging.getLogger(__name__)


@register_topology("simple")
class SimpleTopology(BaseTopology):

    def __call__(self, agents, *args, **kwargs):
        agent_queue = [agents[self.start]]
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
