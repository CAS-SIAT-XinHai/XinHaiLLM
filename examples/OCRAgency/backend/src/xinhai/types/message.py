"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Vimos Tan
"""
from __future__ import annotations

import base64
import io
import os
import sys
from datetime import datetime
from typing import List

from more_itertools import split_when
from pydantic import BaseModel

from llamafactory.api.protocol import MultimodalInputItem, ImageURL

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


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

    @staticmethod
    def to_base64(filename: str) -> str:
        buf = io.BytesIO(open(filename, "rb").read())
        img_b64_str = base64.b64encode(buf.getvalue()).decode()
        return f"data:image,{img_b64_str}"

    @classmethod
    def squeeze_to_chat(cls, messages: List[Self], static_path):
        content_str = ""
        content = None
        for message in messages:
            content_str = content_str + "\n" + message.content
            if message.files is not None:
                content = [
                              MultimodalInputItem(
                                  type="text",
                                  text=message.content)
                          ] + [
                              MultimodalInputItem(
                                  type="image_url",
                                  text="",
                                  image_url=ImageURL(
                                      url=cls.to_base64(os.path.join(static_path,
                                                                     f"{f.url.split(os.path.sep)[-1]}.{f.extension}"))))
                              for
                              f in message.files
                          ]
        if content is not None:
            content[0].text = content_str
            return {
                "role": message.role,
                "content": content,
            }
        else:
            return {
                "role": message.role,
                "content": content_str,
            }

    def to_chat(self, static_path):
        if self.files:
            content = [
                          MultimodalInputItem(
                              type="text",
                              text=self.content)
                      ] + [
                          MultimodalInputItem(
                              type="image_url",
                              text="",
                              image_url=ImageURL(
                                  url=self.to_base64(os.path.join(static_path,
                                                                  f"{f.url.split(os.path.sep)[-1]}.{f.extension}"))))
                          for
                          f in self.files
                      ]
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
        for ms in split_when(self.messages, lambda x, y: x.role != y.role):
            if len(ms) > 1:
                messages.append(XinHaiChatMessage.squeeze_to_chat(ms, static_path))
            else:
                messages.append(ms[0].to_chat(static_path))
        return messages
