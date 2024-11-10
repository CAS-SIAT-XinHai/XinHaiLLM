"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Vimos Tan
"""
from enum import Enum


class XinHaiWorkerTypes(str, Enum):
    AGENCY = 'agency'
    BRIDGE = 'bridge'
    FEEDBACK = 'feedback'
    KNOWLEDGE = 'knowledge'
    LLM = 'llm'
    MLLM = 'mllm'
    OCR = 'ocr'
    STORAGE = 'storage'
