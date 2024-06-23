"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Vimos Tan
"""

import asyncio
import json
import logging
from argparse import ArgumentParser
from typing import List

import yaml

from xinhai.arena.agents import AGENT_REGISTRY
from xinhai.arena.agents.base import BaseAgent
from xinhai.arena.environments import ENVIRONMENT_REGISTRY, BaseEnvironment
from xinhai.arena.topology.base import BaseTopology

logger = logging.getLogger(__name__)


class Simulation:
    def __init__(self, agents: List[BaseAgent], environment: BaseEnvironment):
        self.agents = agents
        self.environment = environment

    @classmethod
    def from_config(cls, config_path):
        config = yaml.safe_load(open(config_path))

        arena_config = config['arena']

        # Build the agents
        agents = []
        for agent_configs in arena_config["agents"]:
            agent_type = agent_configs.pop("agent_type")

            agent = AGENT_REGISTRY[agent_type](**agent_configs)
            agents.append(agent)

        # Build the environment
        env_config = arena_config["environment"]
        env_config["agents"] = agents
        env_config["topology"] = BaseTopology.from_config(env_config["topology"])
        environment_type = env_config.pop("environment_type")
        environment = ENVIRONMENT_REGISTRY[environment_type](**env_config)

        return cls(agents, environment)

    def run(self):
        """Run the environment from scratch until it is done."""
        self.environment.reset()
        while not self.environment.is_done():
            asyncio.run(self.environment.step())
        # self.environment.report_metrics()
        print(json.dumps(self.agents[0].memory, indent=2))

    def reset(self):
        self.environment.reset()
        for agent in self.agents:
            agent.reset()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/xinhai.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    simulator = Simulation.from_config(args.config_path)
    simulator.run()
