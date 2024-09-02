"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Vimos Tan
"""

import logging

from xinhai.rag.methods import register_rag, XinHaiRAGMethodBase
from xinhai.types.rag import XinHaiRAGMethodTypes, XinHaiRAGDocumentIn

logger = logging.getLogger(__name__)


@register_rag(XinHaiRAGMethodTypes.SEQUENTIAL)
class XinHaiRAGSequentialMethod(XinHaiRAGMethodBase):
    method_type = XinHaiRAGMethodTypes.SEQUENTIAL

    def __init__(self, config):
        super().__init__(config)

    async def __call__(self, document: XinHaiRAGDocumentIn):
        query = document.text
        logger.debug("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        retrieval_results = await self.retriever.search(query)
        logger.debug(retrieval_results)

        logger.debug("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        augmented_results = self.refiner.refine(query, retrieval_results.documents)
        logger.debug(augmented_results)

        logger.debug("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        generated_results = self.generator.generate(augmented_results)
        logger.debug(generated_results)
        return generated_results
