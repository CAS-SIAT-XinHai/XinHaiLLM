"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Vimos Tan
"""
import os
from datetime import datetime
from typing import List

from pydantic import BaseModel

from llamafactory.api.protocol import MultimodalInputItem, ImageURL


class XinHaiChatFile(BaseModel):
    name: str
    size: int
    type: str
    audio: bool = False
    duration: int = 0
    url: str = ''
    preview: str = ''
    progress: int = 0
    extension: str = ""


class XinHaiChatMessage(BaseModel):
    _id: str
    indexId: str
    content: str
    senderId: str
    username: str
    role: str
    # avatar: Optional[str]
    date: str
    timestamp: str
    # system: Optional[bool]
    # saved: Optional[bool]
    # distributed: Optional[bool]
    # seen: Optional[bool]
    # deleted: Optional[bool]
    # failure: Optional[bool]
    # disableActions: Optional[bool]
    # disableReactions: Optional[bool]
    files: List[XinHaiChatFile] = []

    # reactions: Optional[Dict]
    # replyMessage: Optional['XinHaiChatMessage']

    def to_chat(self, static_path):
        if self.files:
            content = [MultimodalInputItem(
                type="image_url",
                text=self.content,
                image_url=ImageURL(url=os.path.join(static_path,
                                                    f"{f.url.split(os.path.sep)[-1]}.{f.extension}"))) for f in self.files]
        else:
            content = self.content
        return {
            "role": self.role,
            "content": content,
        }

    @classmethod
    def from_chat(cls, messages, role_mapping):
        t = datetime.now()
        return [cls(
            indexId=f'{i}',
            content=m['content'],
            senderId=role_mapping[m['role']],
            username=role_mapping[m['role']],
            role="user",
            date=t.strftime("%a %b %d %Y"),
            timestamp=t.strftime("%H:%M"),
        ) for i, m in enumerate(messages)]


class XinHaiChatCompletionRequest(BaseModel):
    model: str
    messages: List[XinHaiChatMessage]

    # tools: Optional[List[FunctionAvailable]] = None
    # do_sample: bool = True
    # temperature: Optional[float] = None
    # top_p: Optional[float] = None
    # n: int = 1
    # max_tokens: Optional[int] = None
    # stop: Optional[Union[str, List[str]]] = None
    # stream: bool = False

    def to_chat(self, static_path):
        messages = []
        for m in self.messages:
            messages.append(m.to_chat(static_path))
        return messages
