import asyncio
import json
import logging
import os.path
from argparse import ArgumentParser

import yaml

from xinhai.arena.agents import AGENT_REGISTRY
from xinhai.arena.environments import ENVIRONMENT_REGISTRY
from xinhai.arena.simulation import Simulation
from xinhai.arena.topology import TOPOLOGY_REGISTRY

logger = logging.getLogger(__name__)


class AutoCBT(Simulation):

    @classmethod
    def from_config_with_role_description(cls, config_path, env_index, meta):
        config = yaml.safe_load(open(config_path))

        arena_config = config['arena']
        env_config = arena_config["environment"]
        env_config['environment_id'] = f"{env_config['environment_id']}-{env_index}"

        # Build the agents
        agents = []
        for agent_configs in arena_config["agents"]:
            agent_type = agent_configs.pop("agent_type")
            if agent_configs['name'] in ["咨询者", "patient"]:
                agent_configs['role_description'] = meta['question'] + meta['description']

            agent = AGENT_REGISTRY[agent_type](**agent_configs,
                                               controller_address=env_config['controller_address'],
                                               environment_id=env_config["environment_id"])
            agents.append(agent)

        # Build the environment
        env_config["agents"] = agents
        environment_type = env_config.pop("environment_type")
        topologies = []
        for topology_config in env_config["topologies"]:
            topology_type = topology_config.pop("type")
            topologies.append(TOPOLOGY_REGISTRY[topology_type].from_config(topology_config))

        env_config["topologies"] = topologies
        environment = ENVIRONMENT_REGISTRY[environment_type](**env_config)
        [setattr(agent, "environment", environment) for agent in agents]
        return cls(agents, environment)

    def run(self):
        """Run the environment from scratch until it is done."""
        self.environment.reset()
        while not self.environment.is_done():
            asyncio.run(self.environment.step())

        print(json.dumps(self.agents[0].memory.model_dump_json(), indent=2))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/xinhai.yaml")
    parser.add_argument("--data_path", type=str, default="configs/xinhai.yaml")
    parser.add_argument("--language", type=str, default="zh")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # self.environment.report_metrics()
    dataset_name = "psyqa_balanced" if args.language == "zh" else "therapistqa_balanced"

    with open(os.path.join(args.data_path, f"{dataset_name}.json"), 'r', encoding='utf-8') as f:
        for i, item in enumerate(json.load(f)):
            simulator = AutoCBT.from_config_with_role_description(args.config_path, i, item)
            simulator.run()
            break