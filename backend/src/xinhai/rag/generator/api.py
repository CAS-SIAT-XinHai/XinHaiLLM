"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Vimos Tan
"""
import logging
from typing import List

from openai import OpenAIError, OpenAI
from pydantic import BaseModel

from xinhai.rag.generator import XinHaiRAGGeneratorBase, register_generator
from xinhai.types.rag import XinHaiRAGGeneratorTypes, XinHaiRAGAugmentedResult

logger = logging.getLogger(__name__)


class GenerationParams(BaseModel):
    """
          logprobs: Whether to return log probabilities of the output tokens or not. If true,
              returns the log probabilities of each output token returned in the `content` of
              `message`.

          max_tokens: The maximum number of [tokens](/tokenizer) that can be generated in the chat
              completion.

              The total length of input tokens and generated tokens is limited by the model's
              context length.
              [Example Python code](https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken)
              for counting tokens.

          n: How many chat completion choices to generate for each input message. Note that
              you will be charged based on the number of generated tokens across all of the
              choices. Keep `n` as `1` to minimize costs.
    """
    logprobs: bool = False
    max_tokens: int
    n: int = 1


@register_generator(XinHaiRAGGeneratorTypes.API)
class APIGenerator(XinHaiRAGGeneratorBase):
    """Class for api-based openai models"""

    def __init__(self, config):
        super().__init__(config)
        self.model_name = config["model_name"]
        self.generation_params = GenerationParams.model_validate(config["generation_params"])

        self.api_base = config["api_base"]
        self.api_key = config["api_key"]

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
        )

    #     self.client = AsyncOpenAI(
    #         api_key=self.api_key,
    #         base_url=self.api_base,
    #     )
    #
    # async def get_response(self, messages: List[ChatCompletionMessageParam]):
    #     response = await self.client.chat.completions.create(model=self.model_name, messages=messages,
    #                                                          **self.generation_params.model_dump())
    #     return response.choices[0]

    @staticmethod
    def chat_completion(client, model, messages):
        try:
            logger.debug(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            logger.debug(messages)
            chat_response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.98
            )
            logger.debug("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            content = chat_response.choices[0].message.content
            if content.strip():
                logger.debug(f"Get response from {model}: {content}")
                return content.strip()
            else:
                usage = chat_response.usage
                logger.error(f"Error response from {model}: {usage}")
        except OpenAIError as e:
            # Handle all OpenAI API errors
            logger.warning("*****************************************")

    def _generate(self, refined_result: XinHaiRAGAugmentedResult, *args, **kwargs) -> List[str]:
        messages = [
            {
                "role": "system",
                "content": refined_result.system_prompt,
            },
            {
                "role": "user",
                "content": refined_result.user_prompt,
            },
        ]
        chat_response = self.chat_completion(self.client, model=self.model_name, messages=messages)
        return chat_response
