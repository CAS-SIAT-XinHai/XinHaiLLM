import json
import logging

from xinhai.arena.environments import register_environment
from xinhai.arena.environments.base import BaseEnvironment
from typing import List, Dict
from xinhai.arena.agents.base import BaseAgent
import matplotlib.pyplot as plt
import networkx as nx
import uuid
import datetime

logger = logging.getLogger(__name__)

@register_environment("cbt_single")
class CBTEnvironment(BaseEnvironment):

    def __init__(self, environment_id, agents: List[BaseAgent], topology, max_turns=10, cnt_turn=0, controller_address=None, save_path=""):
        self.environment_id = environment_id
        self.agents = agents
        self.topology = topology
        self.max_turns = max_turns
        self.cnt_turn = cnt_turn
        self.controller_address = controller_address
        self.save_path = save_path

    def process_single_turn_dialog(self) -> str:
        agent_queue = [self.agents[0]]
        memory_list = []
        while agent_queue:
            agent = agent_queue.pop(0)
            agent_descriptions = "\n".join(
                [f"{n}: {self.agents[n].role_description}" for n in self.topology.digraph.neighbors(agent.agent_id)])
            data = agent.routing(agent_descriptions)
            print(data)
            targets = data["target"]
            if isinstance(data['target'], int):
                targets = [data['target']]
            targets = [self.agents[n] for n in targets]

            if targets:
                targets_descriptions = "\n".join([f"{n.agent_id}: {n.role_description}" for n in targets])
                message = agent.step(routing=data["method"], agents=targets_descriptions)
                # 咨询师与督导师共享全局的记忆
                memory_list.append(message)
                agent.update_memory(memory_list)

                for a in targets:
                    a.update_memory(memory_list)

                # 合并最后相同的target
                if agent_queue:
                    last_element = agent_queue[-1]
                    for target in targets:
                        if type(target) == type(last_element):
                            # 先移除，再并入
                            agent_queue.pop(-1)
                        agent_queue.append(target)
                else:
                    agent_queue.extend(targets)
            print(f"{agent.agent_id} 已完成")
        self.cnt_turn += 1
        return memory_list[-1].content

    async def step(self, question):
        top_dict_from_qa = json.loads(question)
        self.agents[0].role_description = top_dict_from_qa["question"] + top_dict_from_qa["description"]
        answer = self.process_single_turn_dialog()
        top_dict_from_qa["cbtagency_answer"] = answer
        top_dict_from_qa["generate_time"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        result_str = json.dumps(top_dict_from_qa, ensure_ascii=False, indent=4) + ","
        self.read_and_append(self.save_path, result_str)

    def read_and_append(self, file_path, content):
        # 读取文件内容
        with open(file_path, 'r') as file:
            data = file.read()
            print("Current file contents:")
            print(data)
        # 在文件末尾追加新内容
        self.append_to_file(file_path, content)

    def append_to_file(self, file_path, content):
        # 使用'a'模式打开文件，这样可以在文件末尾追加内容而不会覆盖原有内容
        with open(file_path, 'a') as file:
            # 写入内容
            file.write(content)
            # 通常情况下，使用with语句时不需要手动调用flush()，但为了确保所有数据都被写入磁盘，可以显式调用
            file.flush()
    def reset(self) -> None:
        """Reset the environment"""
        self.cnt_turn = 0
        for agent in self.agents:
            agent.reset()

    def is_done(self) -> bool:
        """Check if the environment is done"""
        return self.cnt_turn >= self.max_turns
