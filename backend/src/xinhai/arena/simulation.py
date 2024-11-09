import asyncio
import json
import logging
from argparse import ArgumentParser
from typing import List

import yaml

from xinhai.arena.agents import AGENT_REGISTRY, BaseAgent
from xinhai.arena.environments import ENVIRONMENT_REGISTRY, BaseEnvironment
from xinhai.arena.topology import TOPOLOGY_REGISTRY

logger = logging.getLogger(__name__)


class Simulation:
    def __init__(self, agents: List[BaseAgent], environment: BaseEnvironment):
        self.agents = agents
        self.environment = environment

    @classmethod
    def from_config(cls, config_path, environment_id=None):
        config = yaml.safe_load(open(config_path))

        arena_config = config['arena']
        env_config = arena_config["environment"]
        env_config["environment_id"] = environment_id

        # Build the agents
        agents = []
        for agent_configs in arena_config["agents"]:
            logger.info(agent_configs)
            agent_type = agent_configs.pop("agent_type")

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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/xinhai.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    simulator = Simulation.from_config(args.config_path)
    simulator.run()
