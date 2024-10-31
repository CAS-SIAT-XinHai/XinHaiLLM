"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Ancheng Xu
         Vimos Tan
"""

from xinhai.arena.topology import register_topology, BaseTopology


@register_topology("chatdev")
class ChatdevTopology(BaseTopology):

    def __call__(self, agents, *args, **kwargs):
        agent_queue = [agents[self.start]]
        while agent_queue:
            agent = agent_queue.pop(0)
            candidate_agents = [agents[n] for n in self.digraph.neighbors(agent.agent_id)]
            targets = []

            
            targets = [agents[candidate_agents[0].agent_id]]
            routing_message = agent.prompt_for_static_routing([n.agent_id for n in targets])
            

            agent_queue.extend(targets)

            targets_descriptions = "\n".join([f"{n.agent_id}: {n.role_description}" for n in targets])
            message = agent.step(
                routing=routing_message.routing_type.routing_name,
                agents=targets_descriptions
            )

            yield targets, message
            agent.update_memory([message])
