"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Vimos Tan
"""
from __future__ import annotations

import importlib
import json
import logging
import os
import re
import sys
from abc import abstractmethod
from typing import List

import requests
from openai import OpenAI, OpenAIError

from xinhai.types.arena import XinHaiArenaAgentTypes, XinHaiArenaLLMConfig
from xinhai.types.i18n import XinHaiI18NLocales
from xinhai.types.memory import XinHaiMemory, XinHaiShortTermMemory, XinHaiLongTermMemory, XinHaiChatSummary
from xinhai.types.message import XinHaiChatMessage
from xinhai.types.prompt import XinHaiPromptType
from xinhai.types.routing import XinHaiRoutingMessage, XinHaiRoutingType
from xinhai.types.storage import XinHaiFetchMemoryResponse, XinHaiStoreMemoryRequest, XinHaiFetchMemoryRequest

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

logger = logging.getLogger(__name__)

AGENT_REGISTRY = {}


def register_agent(name, subname=None):
    """
    New model types can be added to fairseq with the :func:`register_model`
    function decorator.

    For example::

        @register_model('lstm')
        class LSTM(FairseqEncoderDecoderModel):
            (...)

    .. note:: All models must implement the :class:`BaseFairseqModel` interface.
        Typically you will extend :class:`FairseqEncoderDecoderModel` for
        sequence-to-sequence tasks or :class:`FairseqLanguageModel` for
        language modeling tasks.

    Args:
        name (str): the name of the model
        :param name:
        :param subname:
    """

    def register_evaluator_cls(cls):
        if subname is None:
            if name in AGENT_REGISTRY:
                raise ValueError('Cannot register duplicate model ({})'.format(name))
            AGENT_REGISTRY[name] = cls
        else:
            if name in AGENT_REGISTRY and subname in AGENT_REGISTRY[name]:
                raise ValueError('Cannot register duplicate model ({}/{})'.format(name, subname))
            AGENT_REGISTRY.setdefault(name, {})
            AGENT_REGISTRY[name][subname] = cls
        return cls

    return register_evaluator_cls


class BaseAgent:
    name: str
    agent_id: int
    agent_type: XinHaiArenaAgentTypes
    role_description: str

    llm: str
    api_key: str
    api_base: str

    prompt_template: str

    def __init__(self, name, agent_id, role_description, llm,
                 routing_prompt_template, summary_prompt_template, prompt_template,
                 environment_id, controller_address, locale,
                 allowed_routing_types,
                 use_summary=True,
                 static_routing=False,
                 id_template=None,
                 max_retries=4,
                 summary_chunk_size=16):
        self.name = name
        self.agent_id = agent_id
        self.role_description = role_description

        self.max_retries = max_retries
        self.routing_prompt_template = routing_prompt_template
        self.summary_prompt_template = summary_prompt_template
        self.prompt_template = prompt_template
        self.id_template = id_template if id_template else "{id:d}"

        # self.memory = []  # memory of current agent
        # self.messages = {}  # messages between current agent and other agents

        self.llm: XinHaiArenaLLMConfig = XinHaiArenaLLMConfig.from_config(llm, controller_address)
        self.client = OpenAI(
            api_key=self.llm.api_key,
            base_url=self.llm.api_base,
        )
        self.summary_chunk_size = summary_chunk_size

        self.controller_address = controller_address
        self.environment_id = environment_id
        self.locale = XinHaiI18NLocales(locale)
        self.allowed_routing_types = [XinHaiRoutingType.from_str(t) for t in allowed_routing_types]

        self.use_summary = use_summary
        self.static_routing = static_routing

        self.format_prompt, self.format_regex = XinHaiPromptType.get_content(
            locale=self.locale,
            format_prompt_type=XinHaiPromptType.from_str("[FormatResponse]")
        )
        self.format_pattern = re.compile(self.format_regex, re.DOTALL)

        self.memory = self.retrieve_memory()

    def get_summary(self):
        summaries = self.memory.long_term_memory.summaries
        chat_summary = "" if len(summaries) == 0 else summaries[-1].content
        return chat_summary

    @abstractmethod
    def get_history(self, target_agents: List[Self] = None):
        raise NotImplementedError

    @property
    def storage_key(self):
        return f"{self.environment_id}-{self.agent_id}"

    @abstractmethod
    def get_routing_prompt(self, candidate_agents, **kwargs):
        raise NotImplementedError

    def routing(self, candidate_agents: List[Self], **kwargs) -> XinHaiRoutingMessage:
        """Routing logic for agent"""
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
            targets = [a.agent_id for a in candidate_agents]
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
    def step(self,
             routing_message_in: XinHaiRoutingMessage,
             routing_message_out: XinHaiRoutingMessage,
             target_agents: List[Self], **kwargs):
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
                stream=False,
                temperature=0.98
            )
            logger.info("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            content = chat_response.choices[0].message.content
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

    def prompt_for_routing(self, routing_prompt, num_retries=5):
        messages = [{
            "role": "user",
            "content": routing_prompt,
        }]

        while num_retries:
            chat_response = self.chat_completion(self.client, model=self.llm.model, agent_id=self.agent_id,
                                                 messages=messages)
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
        method = XinHaiRoutingType.UNICAST.routing_name if len(
            agent_ids) == 1 else XinHaiRoutingType.MULTICAST.routing_name
        return XinHaiRoutingMessage(
            agent_id=self.agent_id,
            routing_type=XinHaiRoutingType.from_str(method),
            targets=agent_ids,
            routing_prompt="Static Routing",
        )

    def complete_conversation(self, prompt, num_retries=5):
        messages = [{
            "role": "user",
            "content": prompt + "\n\n" + self.format_prompt,
        }]

        while True:
            logger.debug(messages)
            chat_response = self.chat_completion(self.client, model=self.llm.model, agent_id=self.agent_id,
                                                 messages=messages)
            if chat_response:
                rr = self.format_pattern.findall(chat_response)
                if rr:
                    rr = rr[0]
                    try:
                        d = json.loads(rr)
                        if isinstance(d, dict) and len(d) > 0:
                            return self.name, d["response"]
                        else:
                            logger.error(f"Evaluation {rr} error.")
                    except Exception as e:
                        logger.error(f"Evaluation {rr} error: {e}")

    def retrieve_memory(self) -> XinHaiMemory:
        fetch_request = XinHaiFetchMemoryRequest(storage_key=self.storage_key)

        # Get Agent's short-term chat history
        # Get Agent's long-term chat summary/highlights
        try:
            r = requests.post(f"{self.controller_address}/api/storage/fetch-memory",
                              json=fetch_request.model_dump(), timeout=60)
            if r.status_code != 200:
                logger.error(f"Get status fails: {self.controller_address}, {r}")
            memory_response = XinHaiFetchMemoryResponse.model_validate(r.json())

            logger.info("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            logger.info(
                f"Get memories of Agent {self.agent_id}: {json.dumps(memory_response.model_dump_json(), ensure_ascii=False, indent=4)}")

            return memory_response.memory
        except requests.exceptions.RequestException as e:
            logger.error(f"Get status fails: {self.controller_address}, {e}")

    def update_memory(self, messages: List[XinHaiChatMessage]):
        self.memory = self.retrieve_memory()
        # 1. flush new memories to short-term chat history
        # 2. if short-term chat history exceeds maximum rounds, automatically summarize earliest n rounds and flush to
        # long-term chat summary
        current_messages = self.memory.short_term_memory.messages
        if len(current_messages) % self.summary_chunk_size == 0:
            summary = self.dialogue_summary()
            summaries = [summary]
        else:
            summaries = []

        logger.info("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        logger.info(f"Adding summaries: {summaries} to Agent {self.agent_id}")
        self.memory.long_term_memory.summaries.extend(summaries)

        logger.info("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        logger.info(f"Adding {messages} to Agent {self.agent_id}")
        for m in messages:
            current_messages.append(m)

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

        return r.json()

    @abstractmethod
    def dialogue_summary(self) -> XinHaiChatSummary:
        raise NotImplementedError


# automatically import any Python files in the models/ directory
agent_dir = os.path.dirname(__file__)
for file in os.listdir(agent_dir):
    path = os.path.join(agent_dir, file)
    if (
            not file.startswith('_')
            and not file.startswith('.')
            and (file.endswith('.py') or os.path.isdir(path))
    ):
        model_name = file[:file.find('.py')] if file.endswith('.py') else file
        module = importlib.import_module(f'xinhai.arena.agents.{model_name}')
