"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Vimos Tan
"""
import logging
import requests
from xinhai.arena.environments import register_environment
from xinhai.arena.environments.base import BaseEnvironment
from xinhai.types.arena import XinHaiArenaEnvironmentTypes
from xinhai.types.routing import XinHaiRoutingType
import itertools

logger = logging.getLogger(__name__)


@register_environment(XinHaiArenaEnvironmentTypes.SIMPLE)
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
        #上传图片到内存IO中
        url = f"{self.controller_address}/api/upload-file"
        image_path=self.image_path
        with open(image_path, "rb") as file:
            response = requests.post(url, files={"file": file}).json()
        self.image_path=response['Result']
        """Run one step of the environment"""
        agent_queue = [self.agents[0]]
        while agent_queue:
            agent = agent_queue.pop(0)
            candidate_agents = [self.agents[n] for n in self.topology.digraph.neighbors(agent.agent_id)]
            #如果是多模态模型，那么给他一个自己可以回答自己的能力。
            if agent.agent_id==0:
                candidate_agents.append(self.agents[agent.agent_id])
            routing_message = agent.routing(candidate_agents)

            self.cnt_turn += 1
            # logger.info("********************************************************")
            # logger.info(f"{agent.agent_id }:{self.cnt_turn}")
            # logger.info("********************************************************")
            if ((routing_message.routing_type == XinHaiRoutingType.END_CAST or self.cnt_turn >= self.max_turns) and agent.agent_id == 0) or self.cnt_turn > 3:
                # [a.store_long_term_memory() for a in diagraph.nodes]
                break
            targets = [self.agents[n] for n in routing_message.targets if
                       self.topology.digraph.has_edge(agent.agent_id, n)]
            if targets:
                agent_queue.extend(targets)
                agent_queue = [k for k, g in itertools.groupby(agent_queue)]
                targets_descriptions = "\n".join(
                    [f"{n.agent_id}: {n.role_description}" for n in targets])
                if agent.agent_id==0 or agent.agent_id==2:
                    message = agent.step(
                        routing=routing_message.routing_type.routing_name,
                        agents=targets_descriptions,
                        image_url=self.image_path
                    )
                else:
                    message = agent.step(
                        routing=routing_message.routing_type.routing_name,
                        agents=targets_descriptions
                    )
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
