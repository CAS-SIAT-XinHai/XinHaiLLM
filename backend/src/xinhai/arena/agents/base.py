"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Vimos Tan
"""
import logging
import re
from abc import abstractmethod

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

    def __init__(self, name, agent_id, role_description, llm, api_key, api_base, prompt_template, max_retries=5):
        self.name = name
        self.agent_id = agent_id
        self.role_description = role_description
        self.llm = llm
        self.api_key = api_key
        self.api_base = api_base
        self.max_retries = max_retries
        self.prompt_template = prompt_template

        self.memory = []  # memory of current agent
        self.messages = {}  # messages between current agent and other agents
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
        )
        self.response_pattern = re.compile(r"Action Input:([\s\S]+)")

    @abstractmethod
    def step(self):
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

    def complete_conversation(self, prompt, num_retries=5):
        # answer_form = "The generated response should be enclosed by [Response] and [End of Response]."
        messages = [{
            "role": "user",
            "content": prompt,
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
        # worker_addr = self.storage
        # try:
        #     r = requests.post(worker_addr + "/worker_storage_get", json=params, timeout=60)
        # except requests.exceptions.RequestException as e:
        #     logger.error(f"Get status fails: {worker_addr}, {e}")
        #     return None
        #
        # if r.status_code != 200:
        #     logger.error(f"Get status fails: {worker_addr}, {r}")
        #     return None
        #
        # return r.json()

        return self.messages

    def update_memory(self, memories):
        # worker_addr = self.storage
        # try:
        #     r = requests.post(worker_addr + "/worker_storage_insert", json=params, timeout=60)
        # except requests.exceptions.RequestException as e:
        #     logger.error(f"Get status fails: {worker_addr}, {e}")
        #     return None
        #
        # if r.status_code != 200:
        #     logger.error(f"Get status fails: {worker_addr}, {r}")
        #     return None
        #
        # return r.json()
        self.memory.extend(memories)
