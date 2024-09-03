"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Vimos Tan
"""
import importlib
import logging
import os
from typing import Union, List

import torch
from langchain_core.vectorstores import VectorStore

from xinhai.types.rag import XinHaiRAGIndexerTypes, XinHaiRAGDocumentIn

logger = logging.getLogger(__name__)

INDEXER_REGISTRY = {}


def register_indexer(name, subname=None):
    """
    New model types can be added to fairseq with the :func:`register_model`
    function decorator.

    For example::

        @register_model('lstm')
        class LSTM(FairseqEncoderDecoderModel):
            (...)

    .. note:: All models must implement the :class:`BaseFairseqModel` interface.
        Typically you will extend :class:`FairseqEncoderDecoderModel` for
        sequence-to-sequence tasks or :class:`FairseqLanguageModel` for
        language modeling tasks.

    Args:
        name (str): the name of the model
        :param name:
        :param subname:
    """

    def register_indexer_cls(cls):
        if subname is None:
            if name in INDEXER_REGISTRY:
                raise ValueError('Cannot register duplicate model ({})'.format(name))
            INDEXER_REGISTRY[name] = cls
        else:
            if name in INDEXER_REGISTRY and subname in INDEXER_REGISTRY[name]:
                raise ValueError('Cannot register duplicate model ({}/{})'.format(name, subname))
            INDEXER_REGISTRY.setdefault(name, {})
            INDEXER_REGISTRY[name][subname] = cls
        return cls

    return register_indexer_cls


class XinHaiRAGIndexerBase:
    name: XinHaiRAGIndexerTypes

    def __init__(self, config):
        self.index_path = config['index_path']

    def build_index(self, corpus_path_or_documents: Union[str, List[XinHaiRAGDocumentIn]]):
        pass

    def search(self):
        pass


class XinHaiRAGDenseIndexer(XinHaiRAGIndexerBase):
    vectorstore: VectorStore

    def __init__(self, config):
        super().__init__(config)

    @torch.no_grad()
    def build_index(self, documents):
        ids = [d.id for d in documents]
        texts = [d.text for d in documents]
        metadatas = [d.metadata for d in documents]
        # build index
        logger.info("Creating index")
        self.vectorstore.add_texts(texts, metadatas, ids)
        logger.info("Finish!")

    def reset_index(self):
        pass

    def search(self):
        pass


# automatically import any Python files in the models/ directory
indexer_dir = os.path.dirname(__file__)
for file in os.listdir(indexer_dir):
    path = os.path.join(indexer_dir, file)
    if (
            not file.startswith('_')
            and not file.startswith('.')
            and (file.endswith('.py') or os.path.isdir(path))
    ):
        model_name = file[:file.find('.py')] if file.endswith('.py') else file
        module = importlib.import_module(f'xinhai.rag.indexer.{model_name}')
