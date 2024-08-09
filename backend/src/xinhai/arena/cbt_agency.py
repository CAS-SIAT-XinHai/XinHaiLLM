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


class CBT_Single_Simulation:
    def __init__(self, agents: List[BaseAgent], environment: BaseEnvironment):
        self.agents = agents
        self.environment = environment

    @classmethod
    def from_config(cls, config_path):
        config = yaml.safe_load(open(config_path))

        def question_length(d):
            return len(d["question"]+d['description'])

        with open("/data/datasets/AI4Psychology/PsyQA/PsyQA_full.json", 'r', encoding='utf-8') as f:
            psyqa_full_list = json.load(f)
            top_dict_list = sorted(psyqa_full_list, key=question_length, reverse=True)[:2235]

        # psyqa_list = ["我该遵循自己的内心还是照顾男友的感受？去年偶然又和初恋联系上了，他已成家，我也已有男订婚男友。大家再次谈笑风生，说说以前，聊聊日常和各自现在的生活，也相互理解和关怀，就像是认识多年熟悉的老朋友。最近他不太顺心，要出去远行一趟，刚好我也有时间，就想着，要不一块同行，也有个照应，再来，由于曾经分开其实心里一直有个小遗憾，觉得那时候没有好好的陪他走一段，所以很想弥补这种缺憾。男友觉得我们快结婚了，不同意我和朋友出去（不想他多想和误会所以没有说明白是曾经的关系），无论我怎样解释怎样表达我想要去一趟的想法，他都不同意，我也不想一意孤行，但是我又确实很想在自己成家之前，陪他走这一段，然后好好放下缺憾，好好结婚和老公经营我们的生活，但又担心他得知真实情况后产生误会而影响我和他的感情。我是爱未婚夫的，但我又很想弥补曾经青春的缺憾，所以我该怎么做？"]
        for index, top_dict in enumerate(top_dict_list):
            print(f"开始处理第{index}条question：{top_dict['question'] + top_dict['description']}")
            config["arena"]["environment"]["environment_id"] = uuid.uuid4()
            config["arena"]["agents"][0]["role_description"] = top_dict['question'] + top_dict['description']
            arena_config = config['arena']
            env_config = arena_config["environment"]

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

    def run(self):
        """Run the environment from scratch until it is done."""
        self.environment.reset()
        while not self.environment.is_done():
            asyncio.run(self.environment.step())
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

    simulator = CBT_Single_Simulation.from_config(args.config_path)
    simulator.run()
