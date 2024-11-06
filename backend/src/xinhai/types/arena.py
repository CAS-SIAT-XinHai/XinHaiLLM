"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Vimos Tan
"""
from enum import Enum

from pydantic import BaseModel


class XinHaiArenaAgentTypes(str, Enum):
    SIMPLE = 'simple'
    PROXY = 'proxy'
    VERIFY_AGENT = 'verify_agent'
    OCR = 'ocr_agent'
    MLLM = 'mllm_agent'


class XinHaiArenaEnvironmentTypes(str, Enum):
    SIMPLE = 'simple'
    AGENCY = 'agency'
    OCRAGENCY = 'ocragency'


class XinHaiArenaLLMConfig(BaseModel):
    model: str
    api_key: str
    api_base: str = None
    do_sample: bool
    temperature: float
    top_p: float
    max_new_tokens: int
    num_return_sequences: int = 1

    @classmethod
    def from_config(cls, llm_config, controller_address):
        if 'api_base' not in llm_config:
            llm_config['api_base'] = f'{controller_address}/v1'
        return cls(**llm_config)
