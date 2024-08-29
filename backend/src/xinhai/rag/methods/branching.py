"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Vimos Tan
"""

import logging

from xinhai.rag.methods import XinHaiRAGMethodBase, register_rag
from xinhai.types.rag import XinHaiRAGMethodTypes

logger = logging.getLogger(__name__)


class XinHaiRAGBranchingMethod(XinHaiRAGMethodBase):

    def __init__(self, config):
        super().__init__(config)


@register_rag(XinHaiRAGMethodTypes.REPLUG)
class XinHaiRAGReplugMethod(XinHaiRAGBranchingMethod):
    method_type = XinHaiRAGMethodTypes.REPLUG

    def __init__(self, name):
        super().__init__(name)
        self.name = name


@register_rag(XinHaiRAGMethodTypes.SU_RE)
class XinHaiRAGSuReMethod(XinHaiRAGBranchingMethod):
    method_type = XinHaiRAGMethodTypes.SU_RE

    def __init__(self, name):
        super().__init__(name)
        self.name = name
