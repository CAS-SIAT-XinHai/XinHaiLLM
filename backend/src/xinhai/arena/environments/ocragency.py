"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Vimos Tan
"""
import json
import logging

import requests
from llamafactory.api.protocol import Role
from xinhai.arena.environments import register_environment
from xinhai.arena.environments.base import BaseEnvironment
from xinhai.types.arena import XinHaiArenaAgentTypes, XinHaiArenaEnvironmentTypes
from xinhai.types.message import XinHaiChatMessage

logger = logging.getLogger(__name__)

@register_environment(XinHaiArenaEnvironmentTypes.OCRAGENCY)
class AgencyEnvironment(BaseEnvironment):
    """
    A basic environment implementing the logic of conversation.

    Args:
        agents: List of agents
        llm:
        max_turns: Maximum number of turns
        cnt_turn: Current turn number
    """

    async def step(self,
                   user_question,
                   image_path,
                   save_path
                   ):
        # 上传图片到内存IO中
        url = f"{self.controller_address}/api/upload-file"
        with open(image_path, "rb") as file:
            response = requests.post(url, files={"file": file}).json()
        image_path = response['Result']

        """Run one step of the environment"""
        # 先跑mllm 1.再跑ocr2 ,最后跑验证 0。
        mllm_agent = self.agents[1]
        ocr_agent = self.agents[2]
        verify_agent = self.agents[0]
        routing = ""
        agents = ""
        mllm_agent_message = mllm_agent.step(
            routing=routing,
            agents=agents,
            image_url=image_path,
            user_question=user_question
        ).content
        ocr_agent_message = ocr_agent.step(
            routing=routing,
            agents=agents,
            image_url=image_path,
            user_question=user_question
        ).content
        verify_agent_message = verify_agent.step(
            routing=routing,
            agents=agents,
            ocr_agent_answer=ocr_agent_message,
            mllm_agent_answer=mllm_agent_message,
            user_question=user_question
        ).content
        result = json.loads(verify_agent_message)
        # 存进一个result文件里面
        with open(save_path, 'a', encoding='utf-8') as file:
            json.dump(result, file, ensure_ascii=False)
            file.write("\n")
        return


    def reset(self) -> None:
        """Reset the environment"""
        self.cnt_turn = 0
        for agent in self.agents:
            agent.reset()

    def is_done(self) -> bool:
        """Check if the environment is done"""
        return self.cnt_turn >= self.max_turns
