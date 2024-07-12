from typing import List, Dict, Optional

from pydantic import BaseModel


class XinHaiChatFile(BaseModel):
    name: str
    size: int
    type: str
    audio: bool
    duration: int
    url: str
    preview: str
    progress: int


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

    def to_chat(self):
        messages = []
        for m in self.messages:
            messages.append({
                "role": m.role,
                "content": m.content,
            })
        return messages
