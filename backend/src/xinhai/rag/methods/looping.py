"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Vimos Tan
"""

import logging
from bisect import insort
from collections import Counter
from operator import attrgetter

from xinhai.rag.methods import register_rag, XinHaiRAGMethodBase
from xinhai.types.rag import XinHaiRAGMethodTypes, XinHaiRAGDocumentIn

logger = logging.getLogger(__name__)


class XinHaiRAGLoopingMethod(XinHaiRAGMethodBase):

    def __init__(self, config):
        super().__init__(config)
        self.iter_num = config['iter_num']

    async def __call__(self, *args, **kwargs):
        pass


@register_rag(XinHaiRAGMethodTypes.ITERATIVE)
class XinHaiRAGIterativeMethod(XinHaiRAGLoopingMethod):
    method_type = XinHaiRAGMethodTypes.ITERATIVE

    def __init__(self, config):
        super().__init__(config)

    async def __call__(self, document: XinHaiRAGDocumentIn):
        query = document.text
        generated_results = ""
        iter_num = self.iter_num
        while iter_num:
            logger.debug("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            retrieval_results = await self.retriever.search(query)
            logger.debug(retrieval_results)

            logger.debug("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            augmented_results = self.refiner.refine(document.text, retrieval_results)
            logger.debug(augmented_results)

            logger.debug("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            generated_results = self.generator.generate(augmented_results)
            logger.debug(generated_results)

            query = f"{query} {generated_results}"
            iter_num -= 1
        return generated_results


@register_rag(XinHaiRAGMethodTypes.SELF_ASK)
class XinHaiRAGSelfAskMethod(XinHaiRAGLoopingMethod):
    method_type = XinHaiRAGMethodTypes.SELF_ASK

    def __init__(self, config):
        super().__init__(config)


@register_rag(XinHaiRAGMethodTypes.FLARE)
class XinHaiRAGFlareMethod(XinHaiRAGLoopingMethod):
    method_type = XinHaiRAGMethodTypes.FLARE

    def __init__(self, config):
        super().__init__(config)
        self.threshold = config['threshold']
        self.max_generation_length = config['max_generation_length']
        self.look_ahead_steps = config['look_ahead_steps']
        self.stop_sym = list("!@#$%^&*()\n\n)(*&^%$#@!")


@register_rag(XinHaiRAGMethodTypes.IR_COT)
class XinHaiRAGIRCoTMethod(XinHaiRAGLoopingMethod):
    method_type = XinHaiRAGMethodTypes.IR_COT

    def __init__(self, config):
        super().__init__(config)
        self.stop_condition = config['stop_condition']

    async def __call__(self, document: XinHaiRAGDocumentIn):
        query = document.text
        iter_num = self.iter_num
        by_score = attrgetter('score')

        retrieved_documents = []
        while iter_num:
            logger.debug("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            retrieval_results = await self.retriever.search(query)
            # insert doc according to score
            for doc in retrieval_results.documents:
                insort(retrieved_documents, doc, key=by_score)
            # keep only higher score of a same doc
            doc_counter = Counter([doc.document.id for doc in retrieved_documents])
            # remove start from low score docs
            for i in range(len(retrieved_documents) - 1, 0, -1):
                doc = retrieved_documents[i]
                doc_id = doc.document.id
                if doc_counter[doc_id] > 1:
                    retrieved_documents.pop(i)
                    doc_counter[doc_id] -= 1

            logger.debug("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            augmented_results = self.refiner.refine(query, retrieved_documents)
            logger.debug(augmented_results)

            logger.debug("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            generated_thought = self.generator.generate(augmented_results)
            logger.debug(generated_thought)

            if self.stop_condition in generated_thought:
                return generated_thought

            query = f"{query} {generated_thought}"
            iter_num -= 1
