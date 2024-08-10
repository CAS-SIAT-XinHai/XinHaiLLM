from datetime import datetime
from xinhai.types.message import XinHaiChatMessage
from xinhai.arena.agents import register_agent
from xinhai.arena.agents.base import BaseAgent
from openai import OpenAI, OpenAIError
import re, json


@register_agent("consultant")
class ConsultantAgent(BaseAgent):

    def __init__(self, name, agent_id, role_description, llm, api_key, api_base,
                 routing_prompt_template, summary_prompt_template, prompt_template,
                 environment_id, controller_address, max_retries=5, advice_border_line="4", improve_response_by_advice_prompt=""):
        self.name = name
        self.agent_id = agent_id
        self.role_description = role_description

        self.llm = llm
        self.api_key = api_key
        self.api_base = api_base

        self.max_retries = max_retries
        self.routing_prompt_template = routing_prompt_template
        self.summary_prompt_template = summary_prompt_template
        self.prompt_template = prompt_template

        # self.memory = []  # memory of current agent
        # self.messages = {}  # messages between current agent and other agents
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
        )
        self.response_pattern = re.compile(r"\[Response]([\s\S]+)\[End of Response]")
        self.summary_chunk_size = 5

        self.controller_address = controller_address
        self.environment_id = environment_id

        self.memory = self.retrieve_memory()
        self.back_direction = False
        self.generate_response = ""
        self.advice_border_line = float(str(advice_border_line))
        self.improve_response_by_advice_prompt = improve_response_by_advice_prompt

    def reset(self) -> None:
        pass

    def routing(self, agent_descriptions):
        candidate_agent_ids = list(self.environment.topology.digraph.neighbors(self.agent_id))
        return self.prompt_for_static_routing(candidate_agent_ids)

    def step(self, routing, agents):
        chat_summary = self.get_summary()
        chat_history = self.get_history(is_str_type=False)

        t = datetime.now()
        if self.back_direction is False:
            prompt = self.prompt_template.format(chat_history='\n'.join(self.get_history()), chat_summary=chat_summary, role_description=self.role_description, routing=routing, agents=agents)
            role, content = self.complete_conversation(prompt)
            content_json = json.loads(content)
            content_str = f"[{content_json['strategy']}]{content_json['response']}"
            self.generate_response = content_json
        else:
            change_advice = ""
            for advice in chat_history:
                advice_json = json.loads(advice.content)
                score = float(str(advice_json["score"]))
                if score < self.advice_border_line:
                    change_advice += f"修改意见：{advice_json['response']}\n"
            if change_advice:
                prompt = self.improve_response_by_advice_prompt.format(question=agents, draft_response=f"[{self.generate_response['strategy']}]{self.generate_response['response']}", change_advice=change_advice)
                role, content = self.complete_conversation(prompt)
                content_json = json.loads(content)
                content_str = f"[{content_json['strategy']}]{content_json['response']}"
            else:
                content_str = f"[{self.generate_response['strategy']}]{self.generate_response['response']}"

        result_message = XinHaiChatMessage(indexId='-1', content=content_str, senderId=self.name, username=self.name, role="user", date=t.strftime("%a %b %d %Y"), timestamp=t.strftime("%H:%M"))
        self.back_direction = True
        return result_message

    def complete_conversation(self, prompt, num_retries=5):
        answer_form = "The generated response should be enclosed by [Response] and [End of Response]."
        messages = [{
            "role": "user",
            "content": prompt + "\n\n" + answer_form,
        }]
        while True:
            print(messages)
            chat_response = self.chat_completion(self.client, model=self.llm, agent_id=self.agent_id, messages=messages)
            if chat_response:
                text_list = self.response_pattern.findall(chat_response)
                if text_list and len(text_list) == 1:
                    new_text = text_list[0]
                    is_correct, new_text = self.is_the_format_correct(new_text)
                    if is_correct:
                        break
        return self.name, new_text.replace("\n", "")

    def prompt_for_static_routing(self, agent_ids):
        if self.back_direction or agent_ids is None or len(agent_ids) == 0:
            pre_nodes = list(self.environment.topology.digraph.predecessors(self.agent_id))
            agent_ids = pre_nodes

        method = "[Unicast]" if len(agent_ids) == 1 else "[Multicast]"
        ans = {"method": method, "target": agent_ids}
        return ans

    def is_the_format_correct(self, text):
        strategy_list = ["信息收集", "设定议程", "状态回顾", "治疗目标", "心理教育", "自动化思维", "增强动机", "核心信念", "行为技巧", "防止复发", "家庭作业布置", "请求反馈", "总结"]
        try:
            # 尝试将找到的字符串转换为 JSON 对象
            text = text.strip().replace(" ", "").replace("\n", "")
            data = json.loads(text)
            # 检查 JSON 对象的结构是否符合要求
            if isinstance(data, dict) and 'strategy' in data and 'response' in data:
                strategy_generate = data["strategy"]
                for strategy in strategy_list:
                    if strategy in strategy_generate:
                        data["strategy"] = strategy
                    # 如果不在13种之内，直接保留，后面可以再自行处理
                    return True, json.dumps(data)
            return False, ""
        except json.JSONDecodeError:
            return False, ""
