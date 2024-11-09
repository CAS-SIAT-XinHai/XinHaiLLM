"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Vimos Tan
"""
from enum import Enum
from typing import List

from pydantic import BaseModel

from xinhai.types.memory import XinHaiMemory
from xinhai.types.message import XinHaiChatMessage
from xinhai.types.room import XinHaiChatRoom


class XinHaiStorageErrorCode(Enum):
    OK = 0
    ERROR = 1


class XinHaiFetchMessagesRequest(BaseModel):
    room: XinHaiChatRoom


class XinHaiFetchMessagesResponse(BaseModel):
    messages: List[XinHaiChatMessage] = []


class XinHaiStoreMessagesRequest(BaseModel):
    room: XinHaiChatRoom
    messages: List[XinHaiChatMessage] = []


class XinHaiFetchMemoryRequest(BaseModel):
    storage_key: str


class XinHaiFetchMemoryResponse(BaseModel):
    memory: XinHaiMemory
    error_code: XinHaiStorageErrorCode


class XinHaiStoreMemoryRequest(BaseModel):
    storage_key: str
    memory: XinHaiMemory


class XinHaiStoreMemoryResponse(BaseModel):
    storage_key: str
    short_term_messages_count: int
    long_term_summaries_count: int
    error_code: XinHaiStorageErrorCode


class XinHaiRecallMemoryRequest(BaseModel):
    storage_key: str
    query: str
    top_k: int
    threshold: int


class XinHaiRecallMemoryResponse(BaseModel):
    recalled_memory: XinHaiMemory
    error_code: XinHaiStorageErrorCode


class XinHaiDeleteMemoryRequest(BaseModel):
    storage_key: str
    memory_type: str = "short"


class XinHaiDeleteMemoryResponse(BaseModel):
    num_delete: int
    error_code: XinHaiStorageErrorCode
