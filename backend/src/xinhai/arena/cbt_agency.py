import asyncio
import json
import logging
from argparse import ArgumentParser
from typing import List

import yaml
import uuid

from xinhai.arena.agents import AGENT_REGISTRY
from agents.base import BaseAgent
from xinhai.arena.environments import ENVIRONMENT_REGISTRY, BaseEnvironment
from xinhai.arena.topology import TOPOLOGY_REGISTRY

logger = logging.getLogger(__name__)


class CBT_Single_Simulation:
    def __init__(self, agents: List[BaseAgent], environment: BaseEnvironment):
        self.agents = agents
        self.environment = environment

    @classmethod
    def from_config(cls, config_path):
        config = yaml.safe_load(open(config_path))

        arena_config = config['arena']
        env_config = arena_config["environment"]
        config["arena"]["environment"]["environment_id"] = uuid.uuid4()
        # Build the agents
        agents = []
        for agent_configs in arena_config["agents"]:
            agent_type = agent_configs.pop("agent_type")
            agent = AGENT_REGISTRY[agent_type](**agent_configs, controller_address=env_config['controller_address'], environment_id=env_config["environment_id"])
            agents.append(agent)

        env_config["agents"] = agents
        environment_type = env_config.pop("environment_type")
        env_config["topology"] = TOPOLOGY_REGISTRY[environment_type].from_config(env_config["topology"])
        environment = ENVIRONMENT_REGISTRY[environment_type](**env_config)
        [setattr(agent, "environment", environment) for agent in agents]
        result = cls(agents, environment)
        return result

    def run(self, question):
        """Run the environment from scratch until it is done."""
        self.environment.reset()
        while not self.environment.is_done():
            asyncio.run(self.environment.step(question))
        # self.environment.report_metrics()
        # print(json.dumps(self.agents[0].memory.model_dump_json(), indent=2))

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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_path", type=str, default="../../../../configs/xinhai_cbt_single.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    def question_length(d):
        return len(d["question"] + d['description'])
    with open("/data/datasets/AI4Psychology/PsyQA/PsyQA_full.json", 'r', encoding='utf-8') as f:
        psyqa_full_list = json.load(f)
        top_dict_list = sorted(psyqa_full_list, key=question_length, reverse=True)[:100]

    for index, top_dict in enumerate(top_dict_list):
        simulator = CBT_Single_Simulation.from_config(args.config_path)
        simulator.environment.environment_id = uuid.uuid4()
        simulator.run(json.dumps(top_dict))
