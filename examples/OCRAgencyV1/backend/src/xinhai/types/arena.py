"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Vimos Tan
"""
from enum import Enum


class XinHaiArenaAgentTypes(str, Enum):
    VERIFY_AGENT = "verify_agent"
    SIMPLE = 'simple'
    PROXY = 'proxy'
    MLLM_AGENT = 'mllm_agent'
    OCR_AGENT = 'ocr_agent'

class XinHaiArenaEnvironmentTypes(str, Enum):
    SIMPLE = 'simple'
    AGENCY = 'agency'
