"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Vimos Tan
"""
import logging

from xinhai.rag.indexer import INDEXER_REGISTRY, XinHaiRAGDenseIndexer
from xinhai.rag.retriever import XinHaiRAGRetrieverBase, register_retriever
from xinhai.types.rag import XinHaiRAGRetrieverTypes, XinHaiRAGIndexerTypes, XinHaiRAGRetrievedResult, \
    XinHaiRAGDocumentOut, XinHaiRAGDocumentIn

logger = logging.getLogger(__name__)


@register_retriever(XinHaiRAGRetrieverTypes.DENSE)
class XinHaiRAGDenseRetriever(XinHaiRAGRetrieverBase):
    r"""Dense retriever based on pre-built faiss index."""
    name = XinHaiRAGRetrieverTypes.DENSE

    def __init__(self, config: dict):
        super().__init__(config)
        self.indexer_type = XinHaiRAGIndexerTypes(config['indexer'].pop('type'))
        self.indexer: XinHaiRAGDenseIndexer = INDEXER_REGISTRY[self.indexer_type](config['indexer'])

    async def _search(self, query: str, num: int = None, return_score=False):
        if num is None:
            num = self.topk

        docs_with_scores = await self.indexer.vectorstore.asimilarity_search_with_score(query, k=num)

        return XinHaiRAGRetrievedResult(
            query=query,
            num=num,
            documents=[XinHaiRAGDocumentOut(
                document=XinHaiRAGDocumentIn(
                    id=str(doc.metadata['id']),
                    metadata=doc.metadata,
                    text=doc.page_content
                ),
                score=score
            ) for doc, score in docs_with_scores]
        )
