"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Renhao Li, Vimos Tan
"""
import logging
from typing import List

from xinhai.arena.agents.base import BaseAgent
from xinhai.arena.environments import register_environment
from xinhai.arena.environments.base import BaseEnvironment

logger = logging.getLogger(__name__)


@register_environment("stage")
class StageEnvironment(BaseEnvironment):
    """
    A basic environment implementing the logic of conversation.

    Args:
        agents: List of agents
        max_turns: Maximum number of turns
        cnt_turn: Current turn number
        stages: List of stage names
    """
    def __init__(self, agents: List[BaseAgent], topology, routing_method="static", max_turns=10, cnt_turn=0):
        super().__init__(agents, topology, max_turns, cnt_turn)
        self.stages = topology.stages
        self.routing_method = routing_method
    
    def step(self):
        """Run one step of the environment"""
        for stage_indx in range(len(self.stages)):
            logger.debug(f"************Current Stage: [{self.stages[stage_indx]}]************")
            self.__step_in_one_stage(stage_indx)
        
    def reset(self) -> None:
        """Reset the environment"""
        self.cnt_turn = 0
        for agent in self.agents:
            agent.reset()

    def is_done(self) -> bool:
        """Check if the environment is done"""
        return self.cnt_turn >= self.max_turns

    def __step_in_one_stage(self, stage_indx: int):
        start_node = self.topology.start_nodes[stage_indx]
        diagraph = self.topology.diagraph[stage_indx]
        env_status = self.topology.env_status[stage_indx]
        stage_conv_budget = self.topology.budget[stage_indx]
        ref_info_config = self.topology.ref_info[stage_indx]
        iter_num = self.topology.iter_num[stage_indx]
        
        for _ in range(iter_num):
            agent_queue = [self.agents[start_node]]
            stage_conv_num = 0

            while agent_queue:
                agent = agent_queue.pop(0)

                candidate_agent_ids = list(diagraph.neighbors(agent.agent_id))
                candidate_agent_roles = "\n".join(
                    [f"{n}: {self.agents[n].env_role}" for n in candidate_agent_ids])
                if self.routing_method == "dynamic":
                    data = agent.dynamic_routing(candidate_agent_roles, env_status)
                elif self.routing_method == "static":
                    data = agent.static_routing(candidate_agent_ids)
                logger.debug(data)

                if data["method"] == "[Endcast]":
                    # [a.store_long_term_memory() for a in diagraph.nodes]
                    break
                
                targets = data["target"]
                if isinstance(data['target'], int):
                    targets = [data['target']]
                targets = [self.agents[n] for n in targets]

                agent_queue.extend(targets)

                # Response generation
                if agent.agent_id in ref_info_config.keys():
                    ref_info = agent.get_ref_info(ref_info_config[agent.agent_id])
                else:
                    ref_info = ""

                targets_agents = "\n".join(
                    [f"{n.agent_id}: {n.env_role}" for n in targets])
                message = agent.step(routing=data["method"], agents=targets_agents, env_status=env_status, ref_info=ref_info)

                agent.update_memory([message])
                [a.update_memory([message]) for a in targets]
                
                stage_conv_num += 1
                if stage_conv_num >= stage_conv_budget:
                    # [a.store_long_term_memory() for a in diagraph.nodes]
                    break

        self.cnt_turn += 1
