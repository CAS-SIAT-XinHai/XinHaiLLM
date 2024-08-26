"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Renhao Li, Vimos Tan
"""
import os
import json
import logging
from typing import List

from xinhai.arena.agents.base import BaseAgent
from xinhai.arena.environments import register_environment
from xinhai.arena.environments.base import BaseEnvironment
from xinhai.types.routing import XinHaiRoutingType

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

    def __init__(self, environment_id, agents: List[BaseAgent], topology, routing_method="static", max_turns=10,
                 cnt_turn=0, controller_address=None):
        super().__init__(environment_id, agents, topology, max_turns, cnt_turn, controller_address)
        self.stages = topology.stages
        self.routing_method = routing_method
        self.envolve_agents = list(set().union(*[set(nodes) for nodes in topology.nodes]))
        self.__load_var_from_cache()
        self.message_cache = []

    async def step(self):
        """Run one step of the environment"""
        for stage_indx in range(0, len(self.stages)):
            logger.debug(f"************Current Stage: [{self.stages[stage_indx]}]************")
            self.__step_in_one_stage(stage_indx)

        for agent in self.agents:
            agent.update_long_term_memory()
            agent.clear_short_term_memory()

        self.cross_turn_info = self.message_cache[-1].content
        self.__dump_var_to_cache()
        
        self.message_cache = []
        self.cnt_turn += 1

    def reset(self) -> None:
        """Reset the environment"""
        self.cnt_turn = 0
        for agent in self.agents:
            agent.reset()

    def reset_iter(self) -> None:
        for agent in self.agents:
            agent.reset_iter()

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

        self.cnt_iter = 0
        for _ in range(iter_num):
            agent_queue = [self.agents[start_node]]
            stage_conv_num = 0

            while agent_queue:
                agent = agent_queue.pop(0)

                candidate_agents = [self.agents[n] for n in diagraph.neighbors(agent.agent_id)]
                routing_message = agent.routing(candidate_agents, env_status=env_status)
                if routing_message.routing_type == XinHaiRoutingType.END_CAST:
                    # [a.store_long_term_memory() for a in diagraph.nodes]
                    break

                targets = [self.agents[n] for n in routing_message.targets if diagraph.has_edge(n, agent.agent_id)]

                if targets:
                    agent_queue.extend(targets)
                    # Response generation
                    if agent.agent_id in ref_info_config.keys():
                        ref_info = agent.get_ref_info(ref_info_config[agent.agent_id])
                    else:
                        ref_info = ""

                    targets_agents = "\n".join(
                        [f"{n.agent_id}: {n.env_role}" for n in targets])
                    message = agent.step(
                        routing=routing_message.routing_type.routing_name,
                        agents=targets_agents,
                        env_status=env_status,
                        ref_info=ref_info
                    )

                    agent.update_short_term_memory([message])
                    [a.update_short_term_memory([message]) for a in targets]
                    self.message_cache.append(message)
                    
                    stage_conv_num += 1
                    if stage_conv_num >= stage_conv_budget:
                        [agent.pop_ref_info_cache() for agent in self.agents]
                        break
            
            self.cnt_iter += 1
            self.reset_iter()

    def __load_var_from_cache(self):
        cache_folder_path = "../../examples/PsyTraArena/cache/"
        cache_file_path = "simulation_records.json"
        self.records_path = os.path.join(cache_folder_path, cache_file_path)

        if not os.path.exists(cache_folder_path):
            os.makedirs(cache_folder_path)
        if not os.path.exists(self.records_path):
            tmp = []
            with open(self.records_path, 'w') as f:
                json.dump(tmp, f)
            self.cross_turn_info = "认知行为疗法"
            self.retrieved_ids = []
            self.excluded_ids = []
        else:
            with open(self.records_path, 'r', encoding="utf-8") as f:
                records = json.load(f)
            last_record = records[-1]
            self.cnt_turn = last_record["cnt_turn"] + 1
            
            self.cross_turn_info = last_record["cross_turn_info"]
            self.retrieved_ids = []
            self.excluded_ids = sum([r["retrieved_ids"] for r in records], [])

    def __dump_var_to_cache(self):
        new_record = {
            "cnt_turn": self.cnt_turn,
            "retrieved_ids": self.retrieved_ids,
            "cross_turn_info": self.cross_turn_info,
        }
        with open(self.records_path, 'r', encoding="utf-8") as f:
            records = json.load(f)
        records.append(new_record)
        with open(self.records_path, 'w', encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False)
