"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Vimos Tan
"""
from dataclasses import dataclass
from enum import Enum


@dataclass
class XinHaiRoutingTypeMixin:
    routing_name: str
    description: str


class XinHaiRoutingType(XinHaiRoutingTypeMixin, Enum):
    SINGLE_CAST = "[SingleCast]", "单播，发给一个可通讯的智能体"
    MULTICAST = "[Multicast]", "组播，发给若干可通讯的智能体"
    BROADCAST = "[Broadcast]", "广播，发给所有可通讯的智能体"
    END_CAST = "[EndCast]", "终止，终止与指定智能体之间的通讯"

    @classmethod
    def to_description(cls):
        return "\n".join(f"{member.routing_name}: {member.description}" for name, member in cls.__members__.items())
