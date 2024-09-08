from pydantic import BaseModel


class XinHaiChatUser(BaseModel):
    _id: str
    username: str
    role: str


class XinHaiChatRoom(BaseModel):
    roomId: str
    roomName: str
    avatar: str
    users: list[XinHaiChatUser]
