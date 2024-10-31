"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Ancheng Xu
         Vimos Tan
"""

from xinhai.arena.topology import register_topology, BaseTopology


@register_topology("autocbt")
class AutoCBTTopology(BaseTopology):

    def __call__(self, agents, *args, **kwargs):
        agent_queue = [agents[self.start]]
        while agent_queue:
            agent = agent_queue.pop(0)
            candidate_agents = [agents[n] for n in self.digraph.neighbors(agent.agent_id)]
            targets = []

            # 这段「候选target的个数只有1，那么不再路由，直接让候选target作为targets」不知道能不能加，感觉破坏了自主路由的意思。但不加又太耗费token且触发api的限速
            if len(candidate_agents) == 1:
                targets = [agents[candidate_agents[0].agent_id]]
                routing_message = agent.prompt_for_static_routing([n.agent_id for n in targets])
            else:
                while not targets:
                    routing_message = agent.routing(candidate_agents)

                    targets = [agents[n] for n in routing_message.targets if
                               all([self.digraph.has_edge(agent.agent_id, n),
                                    n not in agent_queue])]

            agent_queue.extend(targets)

            targets_descriptions = "\n".join([f"{n.agent_id}: {n.role_description}" for n in targets])
            message = agent.step(
                routing=routing_message.routing_type.routing_name,
                agents=targets_descriptions
            )

            yield targets, message
            agent.update_memory([message])
