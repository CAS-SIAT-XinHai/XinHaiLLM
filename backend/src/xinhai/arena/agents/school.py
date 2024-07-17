"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Renhao Li, Vimos Tan
"""

import logging

from xinhai.arena.agents import register_agent
from xinhai.arena.agents.base import BaseAgent

from typing import Dict
import requests
import json
import re

logger = logging.getLogger(__name__)


@register_agent("school")
class SchoolAgent(BaseAgent):
    env_role: str

    def __init__(self, name, agent_id, role_description, env_role, llm, api_key, api_base, routing_prompt_template, prompt_template, max_retries=5):
        super().__init__(name, agent_id, role_description, llm, api_key, api_base, routing_prompt_template, prompt_template, max_retries)
        self.env_role = env_role
    
    def reset(self) -> None:
        pass

    def get_history(self):
        dialogue_context = []
        for i, (agent_name, response) in enumerate(self.memory):
            dialogue_context.append(f"{agent_name}: {response}")
        return dialogue_context

    def dynamic_routing(self, agent_descriptions, env_status):
        chat_history = '\n'.join(self.get_history())
        routing_prompt = self.routing_prompt_template.format(agent_name=self.name,
                                                             role_description=self.role_description,
                                                             env_status=env_status,
                                                             chat_history=chat_history,
                                                             agent_descriptions=agent_descriptions)
        return self.prompt_for_routing(routing_prompt)

    def static_routing(self, agent_ids):
        return self.prompt_for_static_routing(agent_ids)

    def step(self, routing, agents, env_status, ref_info):
        chat_history = '\n'.join(self.get_history())
        prompt = self.prompt_template.format(chat_history=chat_history,
                                            role_description=self.role_description,
                                            env_status=env_status,
                                            ref_info=ref_info,
                                            routing=routing,
                                            agents=agents)
        return self.complete_conversation(prompt)
    
    def get_ref_info(self, ref_info_config: Dict, queries=None):
        method = ref_info_config["ref_method"]
        source = ref_info_config["ref_source"]
        if method == "retrieve":
            return self.__retrieve_from_db(source, queries)
        elif method == "recall":
            return self.__recall_from_db(source, queries)
        else:
            raise NotImplementedError

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
            dict_content = tmp1 if self.count ==1 else tmp2
            text_option = ""
            for opt_indx, opt_ctx in dict_content["options"].items():
                if opt_ctx:
                    temp = (" - ").join([opt_indx, opt_ctx])
                    text_option = text_option + temp + "\n"
                else:
                    break
            dict_content["options"] = text_option
            
            text_content = "<question>\n{question}\n<options>\n{options}\n<answer>\n{answer}\n<explanation>\n{explanation}\n".format(**dict_content)
            return text_content
        elif source == "course_outline":
            with open("../../course_outline.txt", 'r', encoding='utf-8') as f:
                tmp_res = f.read()
            tmp_res = "[Start of Course Outline]\n" + tmp_res + "[End of Course Outline]"
        return tmp_res

    def __recall_from_db(self, source, queries):
        return ""

    @staticmethod
    def post_chat_completion(client, model, agent_id, messages):
        try:
            logger.debug(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            logger.debug(f"Sending messages to Agent-{agent_id}: {messages}")
            url = client.base_url
            payload = {
                "model": model,
                "messages": messages
            }
            headers = {
                "accept": "application/json",
                "content-type": "application/json",
                "authorization": f"Bearer {client.api_key}"
            }
            response = requests.post(url, json=payload, headers=headers)

            logger.debug("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            content = response.json().get("choices", [{}])[0].get(
                    "message", {}).get("content", "__error__")
            if content.strip():
                logger.debug(f"Get response from Agent-{agent_id}: {content}")
                return content.strip()
            else:
                usage = str(content)
                logger.error(f"Error response from Agent-{agent_id}: {usage}")
        except Exception as e:
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

    def prompt_for_static_routing(self, agent_ids):
        method = "[Unicast]" if len(agent_ids) == 1 else "[Multicast]"
        ans = {"method": method, "target": agent_ids}
        return ans
