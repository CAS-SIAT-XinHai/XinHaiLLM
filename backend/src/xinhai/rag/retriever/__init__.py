"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Vimos Tan
"""
import importlib
import logging
import os
from abc import abstractmethod
from typing import List, Dict, Union

from pyserini.search.lucene import LuceneSearcher

from xinhai.rag.indexer import XinHaiRAGIndexerBase

logger = logging.getLogger(__name__)

RETRIEVER_REGISTRY = {}


def register_retriever(name, subname=None):
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

    def register_retriever_cls(cls):
        if subname is None:
            if name in RETRIEVER_REGISTRY:
                raise ValueError('Cannot register duplicate model ({})'.format(name))
            RETRIEVER_REGISTRY[name] = cls
        else:
            if name in RETRIEVER_REGISTRY and subname in RETRIEVER_REGISTRY[name]:
                raise ValueError('Cannot register duplicate model ({}/{})'.format(name, subname))
            RETRIEVER_REGISTRY.setdefault(name, {})
            RETRIEVER_REGISTRY[name][subname] = cls
        return cls

    return register_retriever_cls


class XinHaiBaseRetriever:
    """Base object for all retrievers."""
    indexer: XinHaiRAGIndexerBase

    def __init__(self, config):
        self.config = config
        # self.retrieval_method = config["retrieval_method"]
        self.topk = config["topk"]

        # self.index_path = config["index_path"]
        # self.corpus_path = config["corpus_path"]

    @abstractmethod
    def _search(self, query: str, num: int, return_score: bool) -> List[Dict[str, str]]:
        r"""Retrieve topk relevant documents in corpus.

        Return:
            list: contains information related to the document, including:
                contents: used for building index
                title: (if provided)
                text: (if provided)

        """
        pass

    def search(self, *args, **kwargs):
        return self._search(*args, **kwargs)


# automatically import any Python files in the models/ directory
retriever_dir = os.path.dirname(__file__)
for file in os.listdir(retriever_dir):
    path = os.path.join(retriever_dir, file)
    if (
            not file.startswith('_')
            and not file.startswith('.')
            and (file.endswith('.py') or os.path.isdir(path))
    ):
        model_name = file[:file.find('.py')] if file.endswith('.py') else file
        module = importlib.import_module(f'xinhai.rag.retriever.{model_name}')
