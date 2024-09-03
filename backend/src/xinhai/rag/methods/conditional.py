"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Vimos Tan
"""

import logging

from xinhai.rag.methods import register_rag, XinHaiRAGMethodBase
from xinhai.types.rag import XinHaiRAGMethodTypes

logger = logging.getLogger(__name__)


@register_rag(XinHaiRAGMethodTypes.CONDITIONAL)
class XinHaiRAGConditionalMethod(XinHaiRAGMethodBase):
    method_type = XinHaiRAGMethodTypes.CONDITIONAL

    def __init__(self, config):
        """
        inference stage:
            query -> judger -> sequential pipeline or naive generate
        """
        super().__init__(config)
