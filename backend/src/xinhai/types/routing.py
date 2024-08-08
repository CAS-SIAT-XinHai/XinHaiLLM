"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Vimos Tan  Di Yang
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

from pydantic import BaseModel

from xinhai.types.i18n import XinHaiI18NLocales


@dataclass
class XinHaiRoutingTypeMixin:
    routing_name: str
    description: Dict[XinHaiI18NLocales, str]


class XinHaiRoutingType(XinHaiRoutingTypeMixin, Enum):
    LOOPBACK = "[LOOPBACK]", {
        XinHaiI18NLocales.CHINESE: "环回，继续进行陈述",
        XinHaiI18NLocales.ENGLISH: "Loop back, continue with the statement."
    }

    UNICAST = "[Unicast]", {
        XinHaiI18NLocales.CHINESE: "单播，发给一个可通讯的智能体",
        XinHaiI18NLocales.ENGLISH: "Unicast, send to a communicable agent."
    }

    MULTICAST = "[Multicast]", {
        XinHaiI18NLocales.CHINESE: "组播，发给若干可通讯的智能体",
        XinHaiI18NLocales.ENGLISH: "Multicast, send to several communicable agents."
    }

    BROADCAST = "[Broadcast]", {
        XinHaiI18NLocales.CHINESE: "广播，发给所有可通讯的智能体",
        XinHaiI18NLocales.ENGLISH: "Broadcast, send to all communicable agents."
    }

    END_CAST = "[EndCast]", {
        XinHaiI18NLocales.CHINESE: "终止，终止与指定智能体之间的通讯",
        XinHaiI18NLocales.ENGLISH: "Terminated broadcast, end communication with the specified agent."
    }

    @classmethod
    def from_str(cls, label):
        if label in ('[LOOPBACK]', 'LOOPBACK'):
            return cls.LOOPBACK
        elif label in ('[SingleCast]', 'SingleCast', '[Unicast]', 'Unicast'):
            return cls.UNICAST
        elif label in ('[MultiCast]', 'MultiCast'):
            return cls.MULTICAST
        elif label in ('[Broadcast]', 'Broadcast'):
            return cls.BROADCAST
        elif label in ('[EndCast]', 'EndCast'):
            return cls.END_CAST
        else:
            raise NotImplementedError

    @classmethod
    def to_description(cls, locale: XinHaiI18NLocales, allowed_routing_types: List[str]) -> str:
        return "\n".join(
            f" - {member.routing_name}: {member.description[locale]}" for name, member in cls.__members__.items() if
            member.routing_name in allowed_routing_types)


@dataclass
class XinHaiRoutingErrorTypeMixin:
    error_name: str
    description: Dict[XinHaiI18NLocales, str]


class XinHaiRoutingErrorType(XinHaiRoutingErrorTypeMixin, Enum):
    TIMEOUT = "[Timeout]", {
        XinHaiI18NLocales.CHINESE: "超时，通信操作超时",
        XinHaiI18NLocales.ENGLISH: "Timeout, communication operation timed out."
    }

    SELF_ROUTING = "[SelfRouting]", {
        XinHaiI18NLocales.CHINESE: "自我路由，路由到了自身",
        XinHaiI18NLocales.ENGLISH: "Self-routing, routed to itself."
    }

    @classmethod
    def to_description(cls, locale: XinHaiI18NLocales) -> str:
        return "\n".join(
            f"{member.error_name}: {member.description[locale]}" for name, member in cls.__members__.items())


class XinHaiRoutingMessage(BaseModel):
    _id: str
    agent_id: int
    routing_type: XinHaiRoutingType
    targets: List[int]
    routing_prompt: str
