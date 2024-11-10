"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Renhao Li, Vimos Tan
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from typing import Dict, List

import requests

from xinhai.arena.agents import register_agent, BaseAgent
from xinhai.types.memory import XinHaiMemory, XinHaiShortTermMemory, XinHaiLongTermMemory, XinHaiChatSummary
from xinhai.types.message import XinHaiChatMessage
from xinhai.types.routing import XinHaiRoutingType
from xinhai.types.storage import XinHaiFetchMemoryResponse, XinHaiStoreMemoryRequest, XinHaiFetchMemoryRequest, \
    XinHaiRecallMemoryRequest, XinHaiRecallMemoryResponse, \
    XinHaiDeleteMemoryRequest, XinHaiDeleteMemoryResponse

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

logger = logging.getLogger(__name__)


@register_agent("school")
class SchoolAgent(BaseAgent):
    env_role: str

    def __init__(self, name, agent_id, role_description, env_role, llm, api_key, api_base, routing_prompt_template,
                 summary_prompt_template, reflect_prompt_template, prompt_template, id_template, summary_mode,
                 environment_id, controller_address, locale,
                 allowed_routing_types, static_routing):
        super().__init__(name, agent_id, role_description, llm, api_key, api_base,
                         routing_prompt_template, summary_prompt_template, prompt_template,
                         environment_id, controller_address, locale,
                         allowed_routing_types, static_routing,
                         id_template, max_retries=5)
        self.env_role = env_role
        self.last_message = None
        self.ref_info_cache = []
        self.message_mask_list = []
        self.cnt_conv_turn = 0

        self.reflect_prompt_template = reflect_prompt_template
        self.summary_mode = summary_mode

    def reset(self):
        self.last_message = None
        self.ref_info_cache = []
        self.cnt_conv_turn = 0

    def reset_iter(self, stage_conv_budget):
        self.cnt_conv_turn = 0

        mask_len = stage_conv_budget
        mask_end_index = len(self.memory.short_term_memory.messages)
        mask_start_index = mask_end_index - mask_len
        self.message_mask_list += range(mask_start_index, mask_end_index)

    def reset_stage(self):
        self.message_mask_list.clear()

    # def generate_message_id(self):
    #     return len(self.memory.short_term_memory.messages) + 1

    # def generate_summary_id(self):
    #     return len(self.memory.long_term_memory.summaries) + 1

    def get_history(self):
        dialogue_context = []
        for i, message in enumerate(self.memory.short_term_memory.messages):
            if i not in self.message_mask_list:
                dialogue_context.append(f"{message.senderId}: {message.content}")
        return dialogue_context

    def agent_descriptions(self, candidate_agents: List[Self]):
        return "\n".join([f"{a.agent_id}: {a.env_role}" for a in candidate_agents])

    def get_routing_prompt(self, candidate_agents, **kwargs):
        chat_history = '\n'.join(self.get_history())
        agent_descriptions = self.agent_descriptions(candidate_agents)
        return self.routing_prompt_template.format(agent_name=self.name,
                                                   role_description=self.role_description,
                                                   env_status=kwargs['env_status'],
                                                   cnt_conv_turn=self.cnt_conv_turn,
                                                   chat_history=chat_history,
                                                   agent_descriptions=agent_descriptions,
                                                   routing_descriptions=XinHaiRoutingType.to_description(
                                                       locale=self.locale,
                                                       allowed_routing_types=self.allowed_routing_types
                                                   ))

    def step(self, routing, agents, **kwargs):
        self.cnt_conv_turn += 1
        chat_history = '\n'.join(self.get_history())
        prompt = self.prompt_template.format(chat_history=chat_history,
                                             role_description=self.role_description,
                                             env_status=kwargs['env_status'],
                                             cnt_conv_turn=self.cnt_conv_turn,
                                             ref_info=kwargs['ref_info'],
                                             routing=routing,
                                             agents=agents)

        role, content = self.complete_conversation(prompt)
        t = datetime.now()

        return XinHaiChatMessage(
            indexId='-1',
            content=content,
            senderId=self.name,
            username=self.name,
            role=self.env_role,
            date=t.strftime("%a %b %d %Y"),
            timestamp=t.strftime("%H:%M"),
        )

    def get_ref_info(self, ref_info_config: Dict):
        use_ref_cache = ref_info_config["use_ref_cache"]
        if self.ref_info_cache:
            if use_ref_cache:
                return self.ref_info_cache[0]
            else:
                self.pop_ref_info_cache()

        method = ref_info_config["ref_method"]
        source = ref_info_config["ref_source"]
        worker = ref_info_config["ref_worker"]
        query_var = ref_info_config["ref_query"]

        if query_var == "cross_turn_info":
            query = self.environment.cross_turn_info
        elif query_var == "last_message":
            query = self.last_message.content
        elif query_var == "[all]":
            query = None
        else:
            raise NotImplementedError

        if method == "retrieve":
            res = self.retrieve_from_db(worker, source, query)
        elif method == "recall":
            res = self.recall_from_memory(query)
        else:
            raise NotImplementedError
        self.ref_info_cache += res
        return self.ref_info_cache[0]

    # memory-related functions
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

    def update_short_term_memory(self, messages: List[XinHaiChatMessage]):
        # flush new memories to short-term chat history
        self.last_message = messages[-1]
        # self.memory = self.retrieve_memory()  ## 为了不影响masked_memory所作的暂时修改

        for m in messages:
            m.indexId = str(self.generate_message_id())

        memory_request = XinHaiStoreMemoryRequest(
            storage_key=self.storage_key,
            memory=XinHaiMemory(
                storage_key=self.storage_key,
                short_term_memory=XinHaiShortTermMemory(messages=messages),
                long_term_memory=XinHaiLongTermMemory()
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

        return r.json()

    def update_long_term_memory(self):
        # flush current short-term memory to long-term-memory        
        stored_long_term_mem = self.dialogue_summary(self.summary_mode)

        memory_request = XinHaiStoreMemoryRequest(
            storage_key=self.storage_key,
            memory=XinHaiMemory(
                storage_key=self.storage_key,
                short_term_memory=XinHaiShortTermMemory(),
                long_term_memory=XinHaiLongTermMemory(summaries=[stored_long_term_mem])
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
        logger.debug(f"Adding {stored_long_term_mem} to Agent {self.agent_id}")
        self.memory.long_term_memory.summaries.append(stored_long_term_mem)

        return r.json()

    def clear_short_term_memory(self):
        delete_request = XinHaiDeleteMemoryRequest(
            storage_key=self.storage_key,
            memory_type="short"
        )

        try:
            r = requests.post(f"{self.controller_address}/api/storage/delete-memory",
                              json=delete_request.model_dump(), timeout=60)
        except requests.exceptions.RequestException as e:
            logger.error(f"Get status fails: {self.controller_address}, {e}")
            return None

        if r.status_code != 200:
            logger.error(f"Get status fails: {self.controller_address}, {r}")
            return None

        delete_response = XinHaiDeleteMemoryResponse.model_validate(r.json())
        logger.debug("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        logger.debug(f"Delete {delete_response.num_delete} messages from Agent {self.agent_id}")
        self.memory.short_term_memory.messages = []

        return delete_response

    def dialogue_summary(self, summary_mode: str) -> XinHaiChatSummary:
        chat_history = '\n'.join(self.get_history())
        if summary_mode == "full":
            content = chat_history
        elif summary_mode == "summary":
            messages = [
                {
                    "role": "user",
                    "content": self.summary_prompt_template.format(chat_history=chat_history)
                },
            ]
            content = self.chat_completion(client=self.client, model=self.llm, agent_id=self.agent_id,
                                           messages=messages)
        elif summary_mode == "reflect":
            messages = [
                {
                    "role": "user",
                    "content": self.reflect_prompt_template.format(chat_history=chat_history)
                },
            ]
            content = self.chat_completion(client=self.client, model=self.llm, agent_id=self.agent_id,
                                           messages=messages)
        elif summary_mode == "reflect_w_exp":
            past_exp = self.recall_from_memory(query=chat_history)[0]
            messages = [
                {
                    "role": "user",
                    "content": self.reflect_prompt_template.format(chat_history=chat_history, past_exp=past_exp)
                },
            ]
            content = self.chat_completion(client=self.client, model=self.llm, agent_id=self.agent_id,
                                           messages=messages)
        else:
            raise NotImplementedError

        return XinHaiChatSummary(
            indexId=str(self.generate_summary_id()),
            content=content,
            messages=self.memory.short_term_memory.messages
        )

    def recall_from_memory(self, query) -> List[XinHaiChatSummary]:
        top_k = 3
        threshold = 50
        recall_request = XinHaiRecallMemoryRequest(storage_key=self.storage_key, query=query, top_k=top_k,
                                                   threshold=threshold)

        try:
            r = requests.post(f"{self.environment.controller_address}/api/storage/recall-memory",
                              json=recall_request.model_dump(), timeout=60)
            if r.status_code != 200:
                logger.error(f"Get status fails: {self.controller_address}, {r}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Get status fails: {self.controller_address}, {e}")

        logger.debug(r.json())
        memory_response = XinHaiRecallMemoryResponse.model_validate(r.json())

        logger.debug("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        logger.debug(
            f"Get long-term memories of Agent {self.agent_id}: {json.dumps(memory_response.model_dump_json(), ensure_ascii=False, indent=4)}")

        return self.prompt_for_experience(memory_response.recalled_memory)

    # knowledge-related functions
    def retrieve_from_db(self, worker, source, query, top_k=5):
        if source == "Catalogs":
            with open("../../examples/PsyTraArena/resources/ProDB_Catalogs.txt", 'r', encoding='utf-8') as f:
                tmp_res = f.read()
            return [tmp_res]
        elif source == "ProDB":
            params_for_query_search = {
                "user_query": query,
                "source": source,
                "top_k": 5,
                "exclude": self.environment.excluded_knowledge_ids
            }
        elif source == "CPsyExamDB":
            params_for_query_search = {
                "user_query": query,
                "source": source,
                "collections": ["kg_collection", "ca_collection"],
                "top_k": [3, 2],
            }

        try:
            call_func = "query-search-meta" if worker == "knowledge" else "query-search"
            r = requests.post(
                f"{self.environment.controller_address}/api/{worker}/{call_func}",
                json=params_for_query_search,
                timeout=60
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"Get status fails: {self.environment.controller_address}, {e}")
            return None

        if r.status_code != 200:
            logger.error(f"Get status fails: {self.environment.controller_address}, {r}")
            return None

        logger.debug("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        logger.debug(f"Retrieving data from {worker}-{source}")

        retrieved_res = r.json()

        # 仅针对现有固定流程，还不具备通用性
        if source == "ProDB":
            res = self.prompt_for_teach(retrieved_res)
        elif source == "CPsyExamDB":
            res = self.prompt_for_test(retrieved_res)
        # elif source == "Catalogs":
        #     res = self.prompt_for_schedule(retrieved_res)
        else:
            raise NotImplementedError

        return res

    def prompt_for_teach(self, retrieved_res):
        """
        joint top-k documents into one prompt
        """

        prompt = "[教案开头]\n"
        i = 1
        doc_key = "rag_pro_knowledge_docs"
        meta_key = "rag_pro_knowledge_metas"

        retrieved_knowledge_ids = [c["id"] for c in retrieved_res[meta_key]]
        self.environment.retrieved_knowledge_ids = retrieved_knowledge_ids
        self.environment.excluded_knowledge_ids += retrieved_knowledge_ids

        for item in retrieved_res[doc_key]:
            prompt = prompt + f"#{i}\n" + item + "\n"
            i += 1
        prompt += "[教案结尾]"
        return [prompt]

    def prompt_for_test(self, retrieved_res):
        """
        construct k prompts with top-k documents
        """
        prompts = []
        retrieved_qa_ids = []
        for r in retrieved_res:
            text_option = ""
            dict_r = eval(r)
            for opt_indx, opt_ctx in dict_r["options"].items():
                if opt_ctx:
                    temp = (" - ").join([opt_indx, opt_ctx])
                    text_option = text_option + temp + "\n"
                else:
                    break
            dict_r["options"] = text_option
            prompt = "<问题>\n{question}\n<问题结束>\n<选项>\n{options}\n<选项结束><标准答案>\n{answer}\n<标准答案结束><题解>\n{explanation}\n<题解结束>".format(
                **dict_r)
            prompts.append(prompt)
            retrieved_qa_ids.append(dict_r["id"])

        self.environment.retrieved_qa_ids = retrieved_qa_ids

        return prompts

    def prompt_for_experience(self, recalled_memory):
        prompt = ""
        summaries = recalled_memory.long_term_memory.summaries
        if len(summaries) > 0:
            prompt = "[可供参考的经验]\n以下经验可能对你有所帮助，请自行决定是否参考以做出回应。\n"
            i = 1
            for s in summaries:
                prompt = prompt + f"#{i}\n" + s.content + "\n"
                i += 1
            prompt += "[可供参考的经验结束]"

        return [prompt]

    def pop_ref_info_cache(self):
        if self.ref_info_cache:
            return self.ref_info_cache.pop(0)
