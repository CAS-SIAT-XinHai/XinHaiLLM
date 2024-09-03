"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Vimos Tan
"""
import logging
from typing import List

from xinhai.rag.augmentor import register_augmentor, XinHaiRAGAugmentorBase
from xinhai.types.rag import XinHaiRAGAugmentorTypes, XinHaiRAGAugmentedResult, XinHaiRAGDocumentOut

logger = logging.getLogger(__name__)


@register_augmentor(XinHaiRAGAugmentorTypes.NOOP)
class NoopAugmentor(XinHaiRAGAugmentorBase):
    name = XinHaiRAGAugmentorTypes.NOOP

    def __init__(self, config):
        super().__init__(config)
        self.reference_identifiers = self.reference_template.get_identifiers()

    def format_reference(self, documents: List[XinHaiRAGDocumentOut]):
        format_reference = []
        for idx, doc_item in enumerate(documents):
            metadata = doc_item.document.metadata
            mapping = {
                'id': doc_item.document.id,
                'idx': idx,
                'text': doc_item.document.text,
            }
            metadata.update(mapping)
            format_reference.append(self.reference_template.safe_substitute(metadata))

        return "\n".join(format_reference)

    def _augment(self, query: str, retrieved_documents: List[XinHaiRAGDocumentOut], *args,
                 **kwargs) -> XinHaiRAGAugmentedResult:
        formatted_reference = self.format_reference(retrieved_documents)
        input_params = {"query": query, "reference": formatted_reference}

        system_prompt = self.system_prompt_template.safe_substitute(input_params)
        logger.debug(system_prompt)

        user_prompt = self.user_prompt_template.safe_substitute(input_params)
        logger.debug(user_prompt)

        return XinHaiRAGAugmentedResult(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
