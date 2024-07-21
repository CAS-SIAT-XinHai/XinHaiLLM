"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Vimos Tan
"""
import json
import logging
import re
from abc import abstractmethod

import requests
from openai import OpenAI, OpenAIError

logger = logging.getLogger(__name__)


class BaseAgent:
    name: str
    agent_id: int
    role_description: str

    llm: str
    api_key: str
    api_base: str

    prompt_template: str

    environment: object = None

    def __init__(self, name, agent_id, role_description, llm, api_key, api_base,
                 routing_prompt_template, prompt_template, max_retries=5):
        self.name = name
        self.agent_id = agent_id
        self.role_description = role_description

        self.llm = llm
        self.api_key = api_key
        self.api_base = api_base

        self.max_retries = max_retries
        self.routing_prompt_template = routing_prompt_template
        self.prompt_template = prompt_template

        # self.memory = []  # memory of current agent
        self.messages = {}  # messages between current agent and other agents
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
        )
        self.response_pattern = re.compile(r"\[Response]([\s\S]+)\[End of Response]")

    @abstractmethod
    def routing(self, agent_descriptions):
        """Routing logic for agent"""
        pass

    @abstractmethod
    def step(self, routing, agents):
        """Get one step response"""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the agent"""
        pass

    @staticmethod
    def chat_completion(client, model, agent_id, messages):
        try:
            logger.debug(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            logger.debug(f"Sending messages to Agent-{agent_id}: {messages}")
            chat_response = client.chat.completions.create(
                model=model,
                messages=messages
            )
            logger.debug("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            content = chat_response.choices[0].message.content
            if content.strip():
                logger.debug(f"Get response from Agent-{agent_id}: {content}")
                return content.strip()
            else:
                usage = chat_response.usage
                logger.error(f"Error response from Agent-{agent_id}: {usage}")
        except OpenAIError as e:
            # Handle all OpenAI API errors
            logger.warning("*****************************************")
            logger.warning(f"Error response from Agent-{agent_id}: {e}")

    def prompt_for_routing(self, routing_prompt, num_retries=5):
        messages = [{
            "role": "user",
            "content": routing_prompt,
        }]

        while num_retries:
            chat_response = self.chat_completion(self.client, model=self.llm, agent_id=self.agent_id, messages=messages)
            if chat_response:
                evaluate_ans = re.findall(r'\{(?:[^{}]|(?:\{(?:[^{}])*?\}))*?\}', chat_response)
                if evaluate_ans:
                    evaluate_ans = evaluate_ans[0]
                    try:
                        d = json.loads(evaluate_ans)
                        if isinstance(d, dict) and len(d) > 0:
                            return d
                        else:
                            logger.error(f"Evaluation {evaluate_ans} error.")
                    except Exception as e:
                        logger.error(f"Evaluation {evaluate_ans} error: {e}")
            # num_retries -= 1

    def complete_conversation(self, prompt, num_retries=5):
        answer_form = "The generated response should be enclosed by [Response] and [End of Response]."
        messages = [{
            "role": "user",
            "content": prompt + "\n\n" + answer_form,
        }]

        while True:
            logger.debug(messages)
            chat_response = self.chat_completion(self.client, model=self.llm, agent_id=self.agent_id, messages=messages)
            if chat_response:
                rr = self.response_pattern.findall(chat_response)
                if rr:
                    break

        return self.name, rr[0]

    def retrieve_memory(self):
        params = {
            "user_id": f"Agent-{self.agent_id}",
        }

        # Get Agent's short-term chat history
        # Get Agent's long-term chat summary/highlights
        try:
            r = requests.post(f"{self.environment.controller_address}/api/storage/chat-get", json=params, timeout=60)
        except requests.exceptions.RequestException as e:
            logger.error(f"Get status fails: {self.environment.controller_address}, {e}")
            return None

        if r.status_code != 200:
            logger.error(f"Get status fails: {self.environment.controller_address}, {r}")
            return None

        data = json.loads(r.json())

        logger.debug("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        logger.debug(f"Get memories of Agent {self.agent_id}: {json.dumps(data, ensure_ascii=False, indent=4)}")
        return data

    def update_memory(self, memories):

        params_for_shot_term_memory = {
            "user_id": f"Agent-{self.agent_id}",
            "documents": [content for _, content in memories],
            "metadatas": [{"source": role} for role, _ in memories],
            "short_memory": True
        }

        # 1. flush new memories to short-term chat history
        # 2. if short-term chat history exceeds maximum rounds, automatically summarize earliest n rounds and flush to
        # long-term chat summary
        
        ## 先存储短期记忆
        try:
            r = requests.post(f"{self.environment.controller_address}/api/storage/chat-insert", json=params_for_shot_term_memory, timeout=60)
        except requests.exceptions.RequestException as e:
            logger.error(f"Get status fails: {self.environment.controller_address}, {e}")
            return None

        if r.status_code != 200:
            logger.error(f"Get status fails: {self.environment.controller_address}, {r}")
            return None

        logger.debug("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        logger.debug(f"Adding {memories} to Agent {self.agent_id}")
        
        ## 再存储长期记忆
        new_dialogues_length = len(params_for_shot_term_memory["documents"])
        summary = self.dialogue_summary(new_dialogues_length)
        params_for_long_term_memory = {
            "user_id": f"Agent-{self.agent_id}",
            "documents": [summary],
            "metadatas": [{"source": self.llm}],
            
            "short_memory":False
        }
        logger.debug("=====================================")
        logger.debug(params_for_long_term_memory)
        try:
            r = requests.post(f"{self.environment.controller_address}/api/storage/chat-insert", json=params_for_long_term_memory)
        except requests.exceptions.RequestException as e:
            logger.error(f"Get status fails: {self.environment.controller_address}, {e}")
            return None

        if r.status_code != 200:
            logger.error(f"Get status fails: {self.environment.controller_address}, {r}")
            return None

        logger.debug("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        logger.debug(f"Adding {summary} to Agent {self.agent_id}")
        
        return r.json()
    
    def dialogue_summary(self, new_dialogue_length):
        ### 暂时使用递归总结的办法，即上一轮的内容总结 + 本轮的对话 = 本轮的内容总结。
        r = self.retrieve_memory()
        logger.debug("=======================================")
        logger.debug(type(r))
        short_term_documents = r.get("short_term_documents")
        logger.debug("=======================================")
        logger.debug(short_term_documents)
        short_term_metadatas = r.get("short_term_metadatas")
        summary_dialogues = r.get("summary_dialogues")
        pre_dialogue_summary = summary_dialogues[-1] if len(summary_dialogues) > 0 else " "
        short_term_dialogue = [f'{meta}["source"]: {doc}' for meta, doc in zip(short_term_metadatas, short_term_documents)]
        prompt = f"""请根据之前的对话摘要和新的对话内容，给出新的对话摘要。
                新的对话摘要应当包含之前摘要的内容。
                摘要长度不应过长或过短，应该根据之前对话摘要和对话内容而定。
                 ####
                 以前的对话摘要：{pre_dialogue_summary}
                 
                 ####
                 新的对话内容：{short_term_dialogue[-new_dialogue_length: ]}
                 
                 ###Attention###
                 仅返回新的对话摘要内容，不要返回分析过程！
                """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        response = self.chat_completion(client=self.client, model=self.llm, agent_id=self.agent_id, messages=messages)
        return response