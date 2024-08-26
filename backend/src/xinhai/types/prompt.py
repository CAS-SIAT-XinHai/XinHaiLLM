"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Renhao Li
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

from pydantic import BaseModel

from xinhai.types.i18n import XinHaiI18NLocales

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


@dataclass
class XinHaiPromptTypeMixin:
    prompt_name: str
    description: Dict[XinHaiI18NLocales, List[str]]


class XinHaiPromptType(XinHaiPromptTypeMixin, Enum):
    FORMATPROMPT = "[FormatResponse]", {
        XinHaiI18NLocales.CHINESE: {
            "prompt": "生成的回复用 [回复] 和 [回复结束] 括起来。",
            "regex": r"\[\s*回复\s*\](.*)\[\s*回复结束\s*\]"
        },
        XinHaiI18NLocales.ENGLISH: {
            "prompt": "The generated response should be enclosed by [Response] and [End of Response].",
            "regex": r"\[\s*Response\s*\](.*)\[\s*End of Response\s*\]"
        }
    }

    @classmethod
    def from_str(cls, label):
        if label.lower() in '[formatresponse]':
            return cls.FORMATPROMPT
        else:
            raise NotImplementedError

    @classmethod
    def get_content(cls, locale: XinHaiI18NLocales, format_prompt_type: Self) -> str:
        cnt = format_prompt_type.description[locale]
        return cnt["prompt"], cnt["regex"]
