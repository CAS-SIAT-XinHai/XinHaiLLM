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
            candidate_agents = [self.agents[n] for n in self.topology.digraph.neighbors(agent.agent_id)]
            routing_message = agent.routing(candidate_agents)

            self.cnt_turn += 1
            # 单轮qa中，任qa一直进行下去不受max_trun的限制，直至咨询师选择与咨询者对话。可以改为从yaml传入一个变量判断此时是多轮还是单轮，在这里做判断
            targets = [self.agents[n] for n in routing_message.targets if self.topology.digraph.has_edge(agent.agent_id, n)]
            if targets:
                agent_queue.extend(targets)
                agent_queue = [k for k, g in itertools.groupby(agent_queue)]
                targets_descriptions = "\n".join([f"{n.agent_id}: {n.role_description}" for n in targets])
                message = agent.step(
                    routing=routing_message.routing_type.routing_name,
                    agents=targets_descriptions
                )
                agent.update_memory([message])

                for a in targets:
                    a.update_memory([message])

                #单轮qa中当咨询师选中了咨询者，令cnt等于最大轮数，停止当前轮次。
                # 后续可以改为从yaml传入一个变量判断此时是多轮还是单轮，在这里做判断
                if agent.agent_id == 1:
                    for target in targets:
                        if target.agent_id == 0:
                            self.cnt_turn = self.max_turns #设置这个是为了通过simulation的while not self.environment.is_done()的校验
                            agent_queue.clear() #设置这个是为了跳出当前的while agent_queue循环
                            break

        self.cnt_turn += 1

    def reset(self) -> None:
        """Reset the environment"""
        self.cnt_turn = 0
        for agent in self.agents:
            agent.reset()

    def is_done(self) -> bool:
        """Check if the environment is done"""
        return self.cnt_turn >= self.max_turns
