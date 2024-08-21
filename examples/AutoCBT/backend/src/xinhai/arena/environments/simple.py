import logging
import datetime, json
from xinhai.arena.environments import register_environment
from xinhai.arena.environments.base import BaseEnvironment
from xinhai.types.arena import XinHaiArenaEnvironmentTypes

logger = logging.getLogger(__name__)


@register_environment(XinHaiArenaEnvironmentTypes.SIMPLE)
class SimpleEnvironment(BaseEnvironment):
    """
    A basic environment implementing the logic of conversation.

    Args:
        agents: List of agents
        llm:
        max_turns: Maximum number of turns
        cnt_turn: Current turn number
    """

    async def step(self):
        """Run one step of the environment"""
        agent_queue = [self.agents[0]]
        while agent_queue:
            agent = agent_queue.pop(0)
            candidate_agents = [self.agents[n] for n in self.topology.digraph.neighbors(agent.agent_id)]
            routing_message = agent.routing(candidate_agents)
            logger.debug(routing_message)
            targets = [self.agents[n] for n in routing_message.targets if
                       self.topology.digraph.has_edge(agent.agent_id, n)]
            if targets:
                targets_descriptions = "\n".join(
                    [f"{n.agent_id}: {n.role_description}" for n in targets])
                message = agent.step(
                    routing=routing_message.routing_type.routing_name,
                    agents=targets_descriptions
                )
                agent.update_memory([message])

                for a in targets:
                    a.update_memory([message])

                if agent.agent_id == 1:
                    break_flag = False
                    for target in targets:
                        if target.agent_id == 0:
                            self.save_result_to_file(original_question=self.agents[0].role_description, cbt_answer=message.content, history=agent.get_history())
                            break_flag = True
                            break
                    if break_flag:
                        break


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

        self.cnt_turn += 1

    def save_result_to_file(self, original_question, cbt_answer, history):
        with open("/data/xuancheng/therapistqa_balanced.json", 'r', encoding='utf-8') as f:
            psyqa_full_list = json.load(f)

        for reference_dict in psyqa_full_list:
            if reference_dict["question"] in original_question:
                reference_dict["cbt_answer"] = cbt_answer
                reference_dict["cbt_history"] = history
                reference_dict["cbt_generate_time"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                result_str = json.dumps(reference_dict, ensure_ascii=False, indent=4) + ", "
                # 使用'a'模式打开文件，这样可以在文件末尾追加内容而不会覆盖原有内容
                with open("/data/xuancheng/final_cbtagency/therapistqa_balanced_cbtagency.json", 'a') as file:
                    # 写入内容
                    file.write(result_str)
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
