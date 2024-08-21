import asyncio
import json
import logging
from argparse import ArgumentParser
from typing import List

import yaml
import uuid

from xinhai.arena.agents import AGENT_REGISTRY
from xinhai.arena.agents.base import BaseAgent
from xinhai.arena.environments import ENVIRONMENT_REGISTRY, BaseEnvironment
from xinhai.arena.topology import TOPOLOGY_REGISTRY

logger = logging.getLogger(__name__)


class CBT_Single_Simple_Simulation():
    def __init__(self, agents: List[BaseAgent], environment: BaseEnvironment):
        self.agents = agents
        self.environment = environment

    @classmethod
    def from_config(cls, config_path):
        config = yaml.safe_load(open(config_path))

        arena_config = config['arena']
        env_config = arena_config["environment"]
        env_config["environment_id"] = uuid.uuid4()

        # Build the agents
        agents = []
        for agent_configs in arena_config["agents"]:
            agent_type = agent_configs.pop("agent_type")
            agent = AGENT_REGISTRY[agent_type](**agent_configs, controller_address=env_config['controller_address'], environment_id=env_config["environment_id"])
            agents.append(agent)

        # Build the environment
        env_config["agents"] = agents
        # env_config["topology"] = BaseTopology.from_config(env_config["topology"])
        environment_type = env_config.pop("environment_type")
        env_config["topology"] = TOPOLOGY_REGISTRY[environment_type].from_config(env_config["topology"])
        environment = ENVIRONMENT_REGISTRY[environment_type](**env_config)
        [setattr(agent, "environment", environment) for agent in agents]
        return cls(agents, environment)

    def run(self):
        """Run the environment from scratch until it is done."""
        self.environment.reset()
        while not self.environment.is_done():
            asyncio.run(self.environment.step())
        # self.environment.report_metrics()
        print(json.dumps(self.agents[0].memory.model_dump_json(), indent=2))

    def reset(self):
        self.environment.reset()
        for agent in self.agents:
            agent.reset()

    def next(self, *args, **kwargs):
        """Run the environment for one step and return the return message."""
        return_message = asyncio.run(self.environment.step(*args, **kwargs))
        return return_message

    def update_state(self, *args, **kwargs):
        """Run the environment for one step and return the return message."""
        self.environment.update_state(*args, **kwargs)

def get_need_rerun_dict_list():
    with open("/data/xuancheng/final_cbtagency/therapistqa_balanced_cbtagency.json", 'r', encoding='utf-8') as f:
        dataset_list = json.load(f)
    with open("/data/xuancheng/therapistqa_balanced.json", 'r', encoding='utf-8') as f:
        dataset_reference_list = json.load(f)

    result_list = []
    for ref_dict in dataset_reference_list:
        if ref_dict["question"] not in result_list:
            result_list.append(ref_dict["question"])
    need_to_rerun_list = []
    for ref_dict in dataset_reference_list:
        flag = False
        for test_dict in dataset_list:
            if ref_dict["question"] == test_dict["question"]:
                flag = True
                break
        if flag is False:
            need_to_rerun_list.append(ref_dict)
    return need_to_rerun_list

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_path", type=str, default="../../../../configs/xinhai_cbt_en.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # dataset_reference_list = get_need_rerun_dict_list()
    with open("/data/xuancheng/therapistqa_balanced.json", 'r', encoding='utf-8') as f:
        dataset_reference_list = json.load(f)

    for index, top_dict in enumerate(dataset_reference_list):
        print(f"=================== {index} start==========================")
        simulator = CBT_Single_Simple_Simulation.from_config(args.config_path)
        simulator.agents[0].role_description = top_dict["question"] + top_dict["description"]
        simulator.run()
        print(f"=================== {index} end===========================")