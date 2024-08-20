"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Vimos Tan
"""
from __future__ import annotations

import json
import logging
import re
import sys
from abc import abstractmethod
from typing import List

import requests
from openai import OpenAI, OpenAIError

from xinhai.types.arena import XinHaiArenaAgentTypes
from xinhai.types.i18n import XinHaiI18NLocales
from xinhai.types.memory import XinHaiMemory, XinHaiShortTermMemory, XinHaiLongTermMemory, XinHaiChatSummary
from xinhai.types.message import XinHaiChatMessage
from xinhai.types.routing import XinHaiRoutingMessage, XinHaiRoutingType
from xinhai.types.prompt import XinHaiPromptType
from xinhai.types.storage import XinHaiFetchMemoryResponse, XinHaiStoreMemoryRequest, XinHaiFetchMemoryRequest

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

logger = logging.getLogger(__name__)


class BaseAgent:
    name: str
    agent_id: int
    agent_type: XinHaiArenaAgentTypes
    role_description: str

    llm: str
    api_key: str
    api_base: str

    prompt_template: str

    def __init__(self, name, agent_id, role_description, llm, api_key, api_base,
                 routing_prompt_template, summary_prompt_template, prompt_template,
                 environment_id, controller_address, locale,
                 allowed_routing_types, format_prompt_type, 
                 static_routing=False, max_retries=4):
        self.name = name
        self.agent_id = agent_id
        self.role_description = role_description

        self.llm = llm
        self.api_key = api_key
        self.api_base = api_base or f'{controller_address}/v1'

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
        self.summary_chunk_size = 5

        self.controller_address = controller_address
        self.environment_id = environment_id
        self.locale = XinHaiI18NLocales(locale)
        self.allowed_routing_types = [XinHaiRoutingType.from_str(t) for t in allowed_routing_types]
        self.format_prompt_type = XinHaiPromptType.from_str(format_prompt_type)
        self.static_routing = static_routing

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

    def get_routing_prompt(self, candidate_agents, **kwargs):
        """Get one step response"""
        chat_summary = self.get_summary()
        chat_history = '\n'.join(self.get_history())
        agent_descriptions = self.agent_descriptions(candidate_agents)
        return self.routing_prompt_template.format(agent_name=self.name,
                                                   role_description=self.role_description,
                                                   chat_summary=chat_summary,
                                                   chat_history=chat_history,
                                                   agent_descriptions=agent_descriptions,
                                                   routing_descriptions=XinHaiRoutingType.to_description(
                                                       locale=self.locale,
                                                       allowed_routing_types=self.allowed_routing_types
                                                   ))

    def routing(self, candidate_agents: List[Self], **kwargs) -> XinHaiRoutingMessage:
        """Routing logic for agent"""
        targets = [a.agent_id for a in candidate_agents]
        # if len(self.allowed_routing_types) == 1:
        #     routing_message = XinHaiRoutingMessage(
        #         agent_id=self.agent_id,
        #         routing_type=self.allowed_routing_types[0],
        #         targets=targets,
        #         routing_prompt="Static Routing"
        #     )
        #     return routing_message
        # else:
        if self.static_routing:
            routing_message = self.prompt_for_static_routing(targets)
        else:
            routing_prompt = self.get_routing_prompt(candidate_agents, **kwargs)
            routing_message = None
            while not routing_message:
                data = self.prompt_for_routing(routing_prompt)
                logger.debug(data)
                try:
                    targets = data["target"]
                except KeyError:
                    continue
                
                if isinstance(data['target'], int):
                    targets = [data['target']]

                routing_type = XinHaiRoutingType.from_str(data['method'])
                if self.agent_id not in targets and routing_type in self.allowed_routing_types:
                    routing_message = XinHaiRoutingMessage(
                        agent_id=self.agent_id,
                        routing_type=routing_type,
                        targets=targets,
                        routing_prompt=routing_prompt
                    )

        return routing_message

    @abstractmethod
    def step(self, routing, agents, **kwargs):
        """Get one step response"""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the agent"""
        pass

    @staticmethod
    def chat_completion(client, model, agent_id, messages):
        try:
            logger.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            logger.info(f"Sending messages to Agent-{agent_id}: {messages}")
            chat_response = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                temperature=0.98
            )
            logger.info("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            content = ""
            for chunk in chat_response:
                if chunk.choices[0].delta.content:
                    content += chunk.choices[0].delta.content
            if content.strip():
                logger.info(f"Get response from Agent-{agent_id}: {content}")
                return content.strip()
            else:
                usage = chat_response.usage
                logger.error(f"Error response from Agent-{agent_id}: {usage}")
        except OpenAIError as e:
            # Handle all OpenAI API errors
            logger.warning("*****************************************")
            logger.warning(f"Error response from Agent-{agent_id}: {e}")

    def agent_descriptions(self, candidate_agents: List[Self]):
        return "\n".join([f"{a.agent_id}: {a.role_description}" for a in candidate_agents])

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

    def prompt_for_static_routing(self, agent_ids):
        method = XinHaiRoutingType.UNICAST.routing_name if len(agent_ids) == 1 else XinHaiRoutingType.MULTICAST.routing_name
        return XinHaiRoutingMessage(
            agent_id=self.agent_id,
            routing_type=XinHaiRoutingType.from_str(method),
            targets=agent_ids,
            routing_prompt="Static Routing",
        )

    def complete_conversation(self, prompt, num_retries=5):
        format_prompt, format_regex = XinHaiPromptType.get_content(
            locale=self.locale,
            format_prompt_type=self.format_prompt_type
        )

        messages = [{
            "role": "user",
            "content": prompt + "\n\n" + format_prompt,
        }]

        while True:
            logger.debug(messages)
            chat_response = self.chat_completion(self.client, model=self.llm, agent_id=self.agent_id, messages=messages)
            if chat_response:
                rr = re.compile(format_regex).findall(chat_response)
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
            if r.status_code != 200:
                logger.error(f"Get status fails: {self.controller_address}, {r}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Get status fails: {self.controller_address}, {e}")

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
        if self.generate_message_id() % self.summary_chunk_size == 0:
            summary = self.dialogue_summary()
            summaries = [summary]
        else:
            summaries = []

        for m in messages:
            m.indexId = str(self.generate_message_id())

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
        self.memory.short_term_memory.messages.extend(messages)
        logger.debug("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        logger.debug(f"Adding summaries: {summaries} to Agent {self.agent_id}")
        self.memory.long_term_memory.summaries.extend(summaries)

        return r.json()

    def dialogue_summary(self) -> XinHaiChatSummary:
        chat_summary = self.get_summary()
        chat_history = '\n'.join(self.get_history())
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",
             "content": self.summary_prompt_template.format(chat_summary=chat_summary, chat_history=chat_history)},
        ]
        response = self.chat_completion(client=self.client, model=self.llm, agent_id=self.agent_id, messages=messages)
        return XinHaiChatSummary(
            indexId=str(self.generate_summary_id()),
            content=response,
            messages=self.memory.short_term_memory.messages[-self.summary_chunk_size:]
        )
