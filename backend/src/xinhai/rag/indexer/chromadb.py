"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Vimos Tan
"""
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from xinhai.rag.indexer import XinHaiRAGDenseIndexer
from xinhai.rag.indexer import register_indexer
from xinhai.types.rag import XinHaiRAGIndexerTypes


@register_indexer(XinHaiRAGIndexerTypes.CHROMA)
class XinHaiRAGChromaIndexer(XinHaiRAGDenseIndexer):
    name = XinHaiRAGIndexerTypes.CHROMA
    vectorstore: Chroma

    def __init__(self, config):
        super().__init__(config)

        embeddings_config = config['embeddings']
        hf = HuggingFaceBgeEmbeddings(
            model_name=embeddings_config['model_name'],
            model_kwargs=embeddings_config['model_kwargs'],
            encode_kwargs=embeddings_config['encode_kwargs']
        )
        self.vectorstore = Chroma(
            config['collection_name'],
            embedding_function=hf,
            persist_directory=config['index_path'],
        )

    def reset_index(self):
        self.vectorstore.reset_collection()
