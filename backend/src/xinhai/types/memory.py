"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Vimos Tan
"""
from enum import Enum
from typing import List

from pydantic import BaseModel

from xinhai.types.message import XinHaiChatMessage


class XinHaiMemoryType(str, Enum):
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"


class XinHaiChatSummary(BaseModel):
    indexId: str
    content: str
    messages: List[XinHaiChatMessage] = []


class XinHaiLongTermMemory(BaseModel):
    summaries: List[XinHaiChatSummary] = []


class XinHaiShortTermMemory(BaseModel):
    messages: List[XinHaiChatMessage] = []


class XinHaiMemory(BaseModel):
    storage_key: str
    short_term_memory: XinHaiShortTermMemory
    long_term_memory: XinHaiLongTermMemory
