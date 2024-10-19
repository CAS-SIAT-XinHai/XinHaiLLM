import logging

from xinhai.arena.environments import register_environment
from xinhai.arena.environments.base import BaseEnvironment
from xinhai.types.arena import XinHaiArenaEnvironmentTypes
from xinhai.types.routing import XinHaiRoutingType
import itertools

logger = logging.getLogger(__name__)

@register_environment("autocbt")
class AutoCBTEnvironment(BaseEnvironment):
    async def step(self):
        """Run one step of the environment"""
        agent_queue = [self.agents[0]]
        while agent_queue:
            agent = agent_queue.pop(0)
            # 将base的self.topology.digraph.neighbors(agent.agent_id)换为self.topology.digraph.successors(agent.agent_id)，因为前者查看邻居时会同时出现入边和出边的邻居，successors只关心出边邻居
            candidate_agents = [self.agents[n] for n in self.topology.digraph.successors(agent.agent_id)]
            self.cnt_turn += 1
            while True:
                routing_message = agent.routing(candidate_agents)
                # 可能找到的targets不在candidate_agents中，所以循环至targets不为空。后续可以改为从yaml传入一个变量判断此时是多轮还是单轮，在这里做判断
                targets = [self.agents[n] for n in routing_message.targets if self.topology.digraph.has_edge(agent.agent_id, n)]
                if targets:
                    break

                # 这段「候选target的个数只有1，那么不再路由，直接让候选target作为targets」不知道能不能加，感觉破坏了自主路由的意思。但不加又太耗费token且触发api的限速
                if candidate_agents is not None and len(candidate_agents) == 1:
                    targets = [self.agents[candidate_agents[0].agent_id]]
                    break

            agent_queue.extend(targets)
            agent_queue = [k for k, g in itertools.groupby(agent_queue)]
            targets_descriptions = "\n".join([f"{n.agent_id}: {n.role_description}" for n in targets])
            message = agent.step(
                routing=routing_message.routing_type.routing_name,
                agents=targets_descriptions
            )
            update_memory_list = [message]
            agent.update_memory(update_memory_list)

            # 后续可以改为从yaml传入一个变量判断此时是多轮还是单轮，在这里做判断
            for target in targets:
                if agent.agent_id == 1:  # 单轮对话中，咨询师路由至督导师时，督导师应该有患者的问题+咨询师的预回复内容，才能给出指导意见
                    whole_memory_list = agent.memory.short_term_memory.messages
                    update_memory_list = [whole_memory_list[0], whole_memory_list[-1]]
                # 结束此次单轮qa的唯一结束的地方
                if agent.agent_id == 1 and target.agent_id == 0:
                    agent_queue.clear()
                    # 通过simulation的while not self.environment.is_done()的校验
                    self.cnt_turn = self.max_turns
                    break
                # 单轮对话中不重复路由，删除已路由过的边
                self.topology.digraph.remove_edge(agent.agent_id, target.agent_id)
                # 目前的update_memory机制不允许同时插入多条记忆，只能一条一条的加入，原因在于base的generate_message_id的机制有问题，如果不一条一条地插入，那么self.memory.short_term_memory.messages的长度就会一直不变，那么就会产生两个indexID相同的情况，造成id重复
                for memory_element in update_memory_list:
                    target.update_memory([memory_element])
    def reset(self) -> None:
        """Reset the environment"""
        self.cnt_turn = 0
        for agent in self.agents:
            agent.reset()

    def is_done(self) -> bool:
        """Check if the environment is done"""
        return self.cnt_turn >= self.max_turns
