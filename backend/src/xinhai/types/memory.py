"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Vimos Tan
"""
from enum import Enum
from typing import List

import uuid6
from pydantic import BaseModel, Field

from xinhai.types.message import XinHaiChatMessage


class XinHaiMemoryType(str, Enum):
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"


class XinHaiChatSummary(BaseModel):
    id: str = Field(default_factory=uuid6.uuid7)
    content: str
    messages: List[XinHaiChatMessage] = []


class XinHaiLongTermMemory(BaseModel):
    summaries: List[XinHaiChatSummary] = []
    # messages: List[XinHaiChatMessage] = []


class XinHaiShortTermMemory(BaseModel):
    messages: List[XinHaiChatMessage] = []


class XinHaiMemory(BaseModel):
    storage_key: str
    short_term_memory: XinHaiShortTermMemory
    long_term_memory: XinHaiLongTermMemory
