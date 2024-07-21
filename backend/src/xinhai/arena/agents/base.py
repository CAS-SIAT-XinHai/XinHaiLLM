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
from typing import List

import requests
from openai import OpenAI, OpenAIError

from xinhai.types.memory import XinHaiMemory, XinHaiShortTermMemory, XinHaiLongTermMemory, XinHaiChatSummary
from xinhai.types.message import XinHaiChatMessage
from xinhai.types.storage import XinHaiFetchMemoryResponse, XinHaiStoreMemoryRequest, XinHaiFetchMemoryRequest

logger = logging.getLogger(__name__)


class BaseAgent:
    name: str
    agent_id: int
    role_description: str

    llm: str
    api_key: str
    api_base: str

    prompt_template: str

    def __init__(self, name, agent_id, role_description, llm, api_key, api_base,
                 routing_prompt_template, prompt_template,
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

    def generate_message_id(self):
        messages = self.memory.short_term_memory.messages
        index_id = 0 if len(messages) == 0 else int(messages[-1].indexId)
        return index_id + 1

    def generate_summary_id(self):
        summaries = self.memory.long_term_memory.summaries
        index_id = 0 if len(summaries) == 0 else int(summaries[-1].indexId)
        return index_id + 1

    def get_summary(self):
        summaries = self.memory.long_term_memory.summaries
        chat_summary = "" if len(summaries) == 0 else summaries[-1].content
        return chat_summary

    def get_history(self):
        dialogue_context = []
        for i, message in enumerate(self.memory.short_term_memory.messages[-self.summary_chunk_size:]):
            dialogue_context.append(f"{message.senderId}: {message.content}")
        return dialogue_context

    @property
    def storage_key(self):
        return f"{self.environment_id}-{self.agent_id}"

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

    def retrieve_memory(self) -> XinHaiMemory:
        fetch_request = XinHaiFetchMemoryRequest(storage_key=self.storage_key)

        # Get Agent's short-term chat history
        # Get Agent's long-term chat summary/highlights
        try:
            r = requests.post(f"{self.controller_address}/api/storage/fetch-memory",
                              json=fetch_request.model_dump(), timeout=60)
        except requests.exceptions.RequestException as e:
            logger.error(f"Get status fails: {self.controller_address}, {e}")
            return None

        if r.status_code != 200:
            logger.error(f"Get status fails: {self.controller_address}, {r}")
            return None

        logger.debug(r.json())
        memory_response = XinHaiFetchMemoryResponse.model_validate(r.json())

        logger.debug("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        logger.debug(
            f"Get memories of Agent {self.agent_id}: {json.dumps(memory_response.model_dump_json(), ensure_ascii=False, indent=4)}")

        return memory_response.memory

    def update_memory(self, messages: List[XinHaiChatMessage]):
        self.memory = self.retrieve_memory()
        # 1. flush new memories to short-term chat history
        # 2. if short-term chat history exceeds maximum rounds, automatically summarize earliest n rounds and flush to
        # long-term chat summary
        message_next_id = self.generate_message_id()
        if message_next_id % self.summary_chunk_size == 0:
            summary = self.dialogue_summary()
            summaries = [summary]
        else:
            summaries = []

        memory_request = XinHaiStoreMemoryRequest(
            storage_key=self.storage_key,
            memory=XinHaiMemory(
                storage_key=self.storage_key,
                short_term_memory=XinHaiShortTermMemory(messages=messages),
                long_term_memory=XinHaiLongTermMemory(summaries=summaries),
            )
        )

        try:
            r = requests.post(f"{self.controller_address}/api/storage/store-memory",
                              json=memory_request.model_dump(), timeout=60)
        except requests.exceptions.RequestException as e:
            logger.error(f"Get status fails: {self.controller_address}, {e}")
            return None

        if r.status_code != 200:
            logger.error(f"Get status fails: {self.controller_address}, {r}")
            return None

        logger.debug("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        logger.debug(f"Adding {messages} to Agent {self.agent_id}")
        logger.debug("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        logger.debug(f"Adding {summaries} to Agent {self.agent_id}")

        return r.json()

    def dialogue_summary(self) -> XinHaiChatSummary:
        chat_summary = self.get_summary()
        chat_history = '\n'.join(self.get_history())
        prompt = f"""请根据之前的对话摘要和新的对话内容，给出新的对话摘要。
                新的对话摘要应当包含之前摘要的内容。
                摘要长度不应过长或过短，应该根据之前对话摘要和对话内容而定。
                 ####
                 以前的对话摘要：{chat_summary}
                 
                 ####
                 新的对话内容：{chat_history}
                 
                 ###Attention###
                 仅返回新的对话摘要内容，不要返回分析过程！
                """
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        response = self.chat_completion(client=self.client, model=self.llm, agent_id=self.agent_id, messages=messages)
        return XinHaiChatSummary(
            indexId=str(self.generate_summary_id()),
            content=response,
            messages=self.memory.short_term_memory.messages[-self.summary_chunk_size:]
        )
