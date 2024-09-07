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

from xinhai.arena.agents import register_agent
from xinhai.arena.agents.base import BaseAgent
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
                 summary_prompt_template, prompt_template, environment_id, controller_address, locale,
                 allowed_routing_types):
        super().__init__(name, agent_id, role_description, llm, api_key, api_base,
                         routing_prompt_template, summary_prompt_template, prompt_template,
                         environment_id, controller_address, locale, allowed_routing_types,
                         max_retries=5)
        self.env_role = env_role
        self.last_message = None
        self.ref_info_cache = []
        self.cnt_conv_turn = 0

    def reset(self):
        self.last_message = None
        self.ref_info_cache = []
        self.cnt_conv_turn = 0

    def reset_iter(self):
        self.cnt_conv_turn = 0

    def get_history(self):
        dialogue_context = []
        for i, message in enumerate(self.memory.short_term_memory.messages):
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
        if self.ref_info_cache:
            return self.ref_info_cache[0]

        method = ref_info_config["ref_method"]
        source = ref_info_config["ref_source"]
        worker = ref_info_config["ref_worker"]

        if ref_info_config["ref_query"] == "cross_turn_info":
            query = self.environment.cross_turn_info
        elif ref_info_config["ref_query"] == "last_message":
            query = self.last_message.content
        elif ref_info_config["ref_query"] == "[all]":
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
        self.memory = self.retrieve_memory()

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

    def update_long_term_memory(self, summary_mode="full"):
        # flush current short-term memory to long-term-memory        
        stored_long_term_mem = self.dialogue_summary(summary_mode)

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
        self.memory.long_term_memory.summaries.extend(stored_long_term_mem)

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
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",
                 "content": self.summary_prompt_template.format(chat_history=chat_history)},
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
        top_k = 2
        threshold = 2
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
    def retrieve_from_db(self, worker, source, query):
        if source == "Catalogs":
            with open("../../source_data/ProDB_Catalogs.txt", 'r', encoding='utf-8') as f:
                tmp_res = f.read()
            return [tmp_res]

        params_for_query_search = {
            "user_query": query,
            "source": source,
            "top_k": 10
        }

        try:
            r = requests.post(f"{self.environment.controller_address}/api/{worker}/query-search",
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
        for key in retrieved_res.keys():
            if "rag_pro_knowledge" in key:
                prompt = prompt + f"#{i}\n" + retrieved_res[key] + "\n"
                i += 1
        prompt += "[教案结尾]"
        return [prompt]

    def prompt_for_test(self, retrieved_res):
        """
        construct k prompts with top-k documents
        """
        prompts = []
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
            prompt = "<问题>\n{question}\n<选项>\n{options}\n<标答>\n{answer}\n<题解>\n{explanation}\n".format(**dict_r)
            prompts.append(prompt)
        return prompts

    def prompt_for_experience(self, recalled_memory):
        prompt = ""
        if len(recalled_memory) > 0:
            prompt = "[过去的经验开头]\n"
            i = 1
            for m in recalled_memory:
                prompt = prompt + f"#{i}\n" + m.content + "\n"
                i += 1
            prompt += "[过去的经验结尾]"

        return [prompt]

    def pop_ref_info_cache(self):
        if self.ref_info_cache:
            return self.ref_info_cache.pop(0)

    # legacy functions
    def __retrieve_from_db(self, source, queries):
        if source == "book_content":
            tmp_res = "[教案]\n" + \
                      "心理治疗（英语：Psychotherapy）是由经过受过心理治疗专业训练并通过考核的人员，主要是心理师或接受心理治疗训练的精神科医师以对话为主的治疗模式。而在某些国家，受过相关培训的精神科护士或临床社会工作者亦可取得进行心理治疗的相关资格。建立一种独特的人际关系来协助当事人（或称案主、个案）处理心理问题、减轻主观痛苦经验、医治精神疾病及促进心理健康、个人成长。心理治疗一般是基于心理治疗理论及相关实证研究（主要是咨商心理学、临床心理学和精神病学）而建立的治疗系统，以建立关系、对话、沟通、深度自我探索、行为改变等的技巧来达到治疗目标，例如改善受助者的心理健康或减轻精神疾病症状等。"
        elif source == "qa_test":
            tmp1 = {
                "subject_name": "普通心理学",
                "question_type": "multi",
                "kind": "knowledge",
                "question": "下列选项中，属于感觉适应现象的有( )",
                "options": {
                    "A": "入芝兰之室，久而不闻其香",
                    "B": "刚从暗处走到亮处，两眼什么也看不到，经过几秒钟后才恢复正常",
                    "C": "月明星稀",
                    "D": "音乐会开始后，全场灯光熄灭",
                    "E": ""
                },
                "answer": "AB",
                "explanation": "选项D是人为现象，选项C是感觉对比",
                "id": "acbb8bcb3327f7a4405ce4e41bf274ef909110dd"
            }
            tmp2 = {
                "subject_name": "高等学校教师心理学",
                "question_type": "multi",
                "kind": "knowledge",
                "question": "个体自我意识的发展过程包括（　）。确认答案",
                "options": {
                    "A": "自我中心期",
                    "B": "客观化时期",
                    "C": "主观化时期",
                    "D": "社会化时期",
                    "E": "心理化时期"
                },
                "answer": "ABC",
                "explanation": "个体自我意识从发生、发展到相对稳定和成熟，大约需要20多年的时间，经历自我中心期、客观化时期和主观化时期。",
                "id": "eaf5cf52ccb1d5e90f64289e9ddaf4d0a712173e"
            }

            if not hasattr(self, "count"):
                self.count = 1
            else:
                self.count += 1
            dict_content = tmp1 if self.count <= 2 else tmp2
            text_option = ""
            for opt_indx, opt_ctx in dict_content["options"].items():
                if opt_ctx:
                    temp = (" - ").join([opt_indx, opt_ctx])
                    text_option = text_option + temp + "\n"
                else:
                    break
            dict_content["options"] = text_option

            text_content = "<问题>\n{question}\n<选项>\n{options}\n<标答>\n{answer}\n<题解>\n{explanation}\n".format(
                **dict_content)
            return text_content
        elif source == "course_outline":
            with open("../../course_outline.txt", 'r', encoding='utf-8') as f:
                tmp_res = f.read()
            tmp_res = "[Start of Course Outline]\n" + tmp_res + "[End of Course Outline]"
        return tmp_res

    def __recall_from_db(self, source, queries):
        return ""
