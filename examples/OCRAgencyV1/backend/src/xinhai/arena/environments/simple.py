"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Vimos Tan
"""
import json
import logging
import requests
from xinhai.arena.environments import register_environment
from xinhai.arena.environments.base import BaseEnvironment
from xinhai.types.arena import XinHaiArenaEnvironmentTypes
from xinhai.types.routing import XinHaiRoutingType
import itertools

logger = logging.getLogger(__name__)


@register_environment(XinHaiArenaEnvironmentTypes.SIMPLE)
class SimpleEnvironment(BaseEnvironment):
    """
    A basic environment implementing the logic of conversation.

    Args:
        agents: List of agents
        llm:
        max_turns: Maximum number of turns
        cnt_turn: Current turn number
    """


    async def step(self):
        #上传图片到内存IO中
        url = f"{self.controller_address}/api/upload-file"
        image_path=self.image_path
        with open(image_path, "rb") as file:
            response = requests.post(url, files={"file": file}).json()
        self.image_path=response['Result']

        """Run one step of the environment"""
        #先跑mllm 1.再跑ocr2 ,最后跑验证 0。
        mllm_agent=self.agents[1]
        ocr_agent=self.agents[2]
        verify_agent=self.agents[0]
        mllm_agent_message = mllm_agent.step(
            image_url=self.image_path
        ).content
        ocr_agent_message = ocr_agent.step(
            image_url=self.image_path
        ).content
        verify_agent_message = verify_agent.step(
            ocr_agent_answer=ocr_agent_message,
            mllm_agent_answer=mllm_agent_message
        ).content
        result=json.loads(verify_agent_message)
        #存进一个result文件里面
        with open('/home/whh/project/Xinhai/examples/OCRAgencyV1/result/result.jsonl', 'a', encoding='utf-8') as file:
            json.dump(result, file, ensure_ascii=False)
            file.write("\n")

        return

        # self.cnt_turn += 1

    def reset(self) -> None:
        """Reset the environment"""
        self.cnt_turn = 0
        for agent in self.agents:
            agent.reset()

    def is_done(self) -> bool:
        """Check if the environment is done"""
        return self.cnt_turn >= self.max_turns
