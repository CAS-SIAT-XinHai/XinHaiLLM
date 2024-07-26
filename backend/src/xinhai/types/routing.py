"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Vimos Tan
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict

from xinhai.types.i18n import XinHaiI18NLocales


@dataclass
class XinHaiRoutingTypeMixin:
    routing_name: str
    description: Dict[XinHaiI18NLocales, str]


class XinHaiRoutingType(XinHaiRoutingTypeMixin, Enum):
    LOOPBACK = "[LOOPBACK]", {
        XinHaiI18NLocales.CHINESE: "环回，继续进行陈述"
    }

    SINGLE_CAST = "[SingleCast]", {
        XinHaiI18NLocales.CHINESE: "单播，发给一个可通讯的智能体"
    }

    MULTICAST = "[Multicast]", {
        XinHaiI18NLocales.CHINESE: "组播，发给若干可通讯的智能体"
    }

    BROADCAST = "[Broadcast]", {
        XinHaiI18NLocales.CHINESE: "广播，发给所有可通讯的智能体"
    }

    END_CAST = "[EndCast]", {
        XinHaiI18NLocales.CHINESE: "终止，终止与指定智能体之间的通讯"
    }

    @classmethod
    def to_description(cls, locale: XinHaiI18NLocales) -> str:
        return "\n".join(
            f"{member.routing_name}: {member.description[locale]}" for name, member in cls.__members__.items())
