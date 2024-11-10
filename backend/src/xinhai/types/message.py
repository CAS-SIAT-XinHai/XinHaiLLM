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
import time
import uuid
from datetime import datetime

from more_itertools import split_when
from openai.types.chat import ChatCompletionMessage

from .prompt import XinHaiMMPrompt
from .room import XinHaiChatRoom

if sys.version_info >= (3, 11):
    from typing import Self, Literal, Any, Dict
else:
    from typing_extensions import Self
from enum import Enum, unique
from typing import List, Optional, Union

from pydantic import BaseModel, Field


# The protocols are extracted from llamafactory

@unique
class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    TOOL = "tool"


@unique
class DataRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    OBSERVATION = "observation"


ROLE_MAPPING = {
    Role.USER: DataRole.USER.value,
    Role.ASSISTANT: DataRole.ASSISTANT.value,
    Role.SYSTEM: DataRole.SYSTEM.value,
    Role.FUNCTION: DataRole.FUNCTION.value,
    Role.TOOL: DataRole.OBSERVATION.value,
}


class Function(BaseModel):
    name: str
    arguments: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]


class FunctionAvailable(BaseModel):
    type: Literal["function", "code_interpreter"] = "function"
    function: Optional[FunctionDefinition] = None


class FunctionCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: Function


class ImageURL(BaseModel):
    url: str


@unique
class Finish(str, Enum):
    STOP = "stop"
    LENGTH = "length"
    TOOL = "tool_calls"


class MultimodalInputItem(BaseModel):
    type: Literal["text", "image_url"]
    text: Optional[str] = None
    image_url: Optional[ImageURL] = None


class ChatMessage(BaseModel):
    role: Role
    content: Optional[Union[str, List[MultimodalInputItem]]] = None
    tool_calls: Optional[List[FunctionCall]] = None


class ChatCompletionMessage(BaseModel):
    role: Optional[Role] = None
    content: Optional[str] = None
    tool_calls: Optional[List[FunctionCall]] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    tools: Optional[List[FunctionAvailable]] = None
    do_sample: Optional[bool] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: int = 1
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatCompletionMessage
    finish_reason: Finish


class ChatCompletionStreamResponseChoice(BaseModel):
    index: int
    delta: ChatCompletionMessage
    finish_reason: Optional[Finish] = None


class ChatCompletionResponseUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: ChatCompletionResponseUsage


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionStreamResponseChoice]


class ScoreEvaluationRequest(BaseModel):
    model: str
    messages: List[str]
    max_length: Optional[int] = None


class ScoreEvaluationResponse(BaseModel):
    id: str
    object: Literal["score.evaluation"] = "score.evaluation"
    model: str
    scores: List[float]


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
    id: str
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
    receiverIds: Optional[List[str]] = []

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
        xinhai_messages = []
        for i, m in enumerate(messages):
            if isinstance(m['content'], str):
                xinhai_message = cls(
                    indexId=uuid.uuid4().hex,
                    content=m['content'],
                    senderId=role_mapping['role2id'][m['role']],
                    username=role_mapping['role2name'][m['role']],
                    role="user",
                    date=t.strftime("%a %b %d %Y"),
                    timestamp=t.strftime("%H:%M"),
                    receiverIds=role_mapping['role2receivers'][m['role']],
                )
            elif isinstance(m['content'], list):
                content = ''
                files = []
                for item in m['content']:
                    if item.type == "text":
                        content += item.text
                    else:
                        files.append(item.image_url)

                xinhai_message = cls(
                    indexId=uuid.uuid4().hex,
                    content=content,
                    senderId=role_mapping['role2id'][m['role']],
                    username=role_mapping['role2name'][m['role']],
                    role="user",
                    date=t.strftime("%a %b %d %Y"),
                    timestamp=t.strftime("%H:%M"),
                    files=files,
                    receiverIds=role_mapping['role2receivers'][m['role']],
                )
            else:
                raise ValueError
            xinhai_messages.append(xinhai_message)
        return xinhai_messages


class XinHaiChatCompletionRequest(BaseModel):
    id: str
    model: str
    messages: List[XinHaiChatMessage]
    room: XinHaiChatRoom = None
    tools: Optional[List[FunctionAvailable]] = None
    do_sample: bool = True
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: int = 1
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False

    def to_chat(self, static_path):
        messages = []
        for ms in split_when(self.messages, lambda x, y: x.role != y.role):
            if len(ms) > 1:
                messages.append(XinHaiChatMessage.squeeze_to_chat(ms, static_path))
            else:
                messages.append(ms[0].to_chat(static_path))
        return messages


class XinHaiMMRequest(BaseModel):
    id: str
    type: str
    prompts: List[XinHaiMMPrompt]
    version: str
    model: str
    image: str


class XinHaiMMResult(BaseModel):
    name: str  # the same as XinHaiMMPrompt.name
    value: str  # the OCR result


class XinHaiMMResponse(BaseModel):
    id: str = Field(default="1")
    type: str = Field(default="key")
    result: List[XinHaiMMResult] = Field(default_factory=list)
    version: str = Field(default="1.0")
    model: str = Field(default="default_model")
