"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: wuhaihong`
"""
import logging

from llamafactory.chat.base_engine import Response
from xinhai.arena.environments import register_environment, BaseEnvironment
from xinhai.types.arena import XinHaiArenaEnvironmentTypes

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

    # 提取文字和图片 URL 的函数
    def extract_text_and_images(self, input_messages):
        texts = []
        image_urls = []

        for item in input_messages:
            for content_item in item['content']:
                if content_item.type == 'text' and content_item.text:
                    texts.append(content_item.text)
                elif content_item.type == 'image_url' and content_item.image_url:
                    image_urls.append(content_item.image_url.url)
        text_str = " ".join(texts) if texts else "No text found"
        image_str = " ".join(image_urls) if image_urls else "No image URL found"
        return text_str, image_str

    async def step(self,
                   input_messages,
                   system,
                   tools,
                   do_sample,
                   temperature,
                   top_p,
                   max_new_tokens,
                   num_return_sequences):

        user_question, image_path = self.extract_text_and_images(input_messages)
        # print("image_path:" + str(image_path))
        # 上传图片到内存IO中
        # url = f"{self.controller_address}/api/upload-file"
        # with open(image_path, "rb") as file:
        #     response = requests.post(url, files={"file": file}).json()
        # image_path = response['Result']

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

        # 存进一个result文件里面
        result_message = verify_agent_message
        results = []
        results.append(
            Response(
                response_text=result_message,
                response_length=len(result_message),
                prompt_length=len(result_message),
                finish_reason="stop",
            )
        )
        return results

    def reset(self) -> None:
        """Reset the environment"""
        self.cnt_turn = 0
        for agent in self.agents:
            agent.reset()

    def is_done(self) -> bool:
        """Check if the environment is done"""
        return self.cnt_turn >= self.max_turns
