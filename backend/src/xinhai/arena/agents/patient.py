from datetime import datetime
from xinhai.types.message import XinHaiChatMessage
from xinhai.arena.agents import register_agent
from xinhai.arena.agents.base import BaseAgent
from openai import OpenAI, OpenAIError
import re, json

@register_agent("patient")
class PatientAgent(BaseAgent):

    def __init__(self, name, agent_id, role_description, llm, api_key, api_base,
                 routing_prompt_template, summary_prompt_template, prompt_template,
                 environment_id, controller_address,
                 max_retries=5):
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

    def reset(self) -> None:
        pass

    def routing(self, agent_descriptions):
        candidate_agent_ids = list(self.environment.topology.digraph.neighbors(self.agent_id))
        return self.prompt_for_static_routing(candidate_agent_ids)

    def step(self, routing, agents):
        # chat_summary = self.get_summary()
        # chat_history = '\n'.join(self.get_history())
        # prompt = self.prompt_template.format(chat_history=chat_history, chat_summary=chat_summary, role_description=self.role_description, routing=routing, agents=agents)
        # role, content = self.complete_conversation(prompt)
        t = datetime.now()
        result_message = XinHaiChatMessage(indexId='-1', content=self.role_description, senderId=self.name, username=self.name, role="user", date=t.strftime("%a %b %d %Y"), timestamp=t.strftime("%H:%M"))
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
                rr = self.response_pattern.findall(chat_response)
                if rr:
                    break

        return self.name, rr[0].replace("\n", "")

    def prompt_for_static_routing(self, agent_ids):
        if self.back_direction or agent_ids is None or len(agent_ids) == 0:
            pre_nodes = list(self.environment.topology.digraph.predecessors(self.agent_id))
            agent_ids = pre_nodes

        method = "[Unicast]" if len(agent_ids) == 1 else "[Multicast]"
        ans = {"method": method, "target": agent_ids}
        return ans