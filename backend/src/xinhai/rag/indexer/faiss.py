"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Vimos Tan
"""
import logging

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from xinhai.rag.indexer import XinHaiRAGDenseIndexer
from xinhai.rag.indexer import register_indexer
from xinhai.types.rag import XinHaiRAGIndexerTypes

logger = logging.getLogger(__name__)


@register_indexer(XinHaiRAGIndexerTypes.FAISS)
class XinHaiRAGFaissIndexer(XinHaiRAGDenseIndexer):
    name = XinHaiRAGIndexerTypes.FAISS
    vectorstore: FAISS

    def __init__(self, config):
        super().__init__(config)

        embeddings_config = config['embeddings']
        embedding_function = HuggingFaceEmbeddings(
            model_name=embeddings_config['model_name'],
            model_kwargs=embeddings_config['model_kwargs'],
            encode_kwargs=embeddings_config['encode_kwargs']
        )

        dimensions: int = len(embedding_function.embed_query("dummy"))
        index = faiss.IndexFlatL2(dimensions)

        self.vectorstore = FAISS(
            embedding_function=embedding_function,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )

    def reset_index(self):
        self.vectorstore.index.reset()
        self.vectorstore.docstore = InMemoryDocstore()
