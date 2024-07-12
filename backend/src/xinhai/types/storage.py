from typing import List

from pydantic import BaseModel

from xinhai.types.message import XinHaiChatMessage
from xinhai.types.room import XinHaiChatRoom


class XinHaiFetchMessagesRequest(BaseModel):
    room: XinHaiChatRoom


class XinHaiFetchMessagesResponse(BaseModel):
    messages: List[XinHaiChatMessage]


class XinHaiStoreMessagesRequest(BaseModel):
    room: XinHaiChatRoom
    messages: List[XinHaiChatMessage]
