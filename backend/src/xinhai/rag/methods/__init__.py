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

from xinhai.rag.generator import GENERATOR_REGISTRY
from xinhai.rag.refiner import REFINER_REGISTRY
from xinhai.rag.reranker import XinHaiBaseReranker
from xinhai.rag.retriever import RETRIEVER_REGISTRY
from xinhai.rag.retriever import XinHaiRAGRetrieverBase
from xinhai.types.rag import XinHaiRAGMethodTypes, XinHaiRAGRetrieverTypes, XinHaiRAGRefinerTypes, \
    XinHaiRAGGeneratorTypes

logger = logging.getLogger(__name__)

RAG_REGISTRY = {}


def register_rag(name, subname=None):
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

    def register_rag_cls(cls):
        if subname is None:
            if name in RAG_REGISTRY:
                raise ValueError('Cannot register duplicate model ({})'.format(name))
            RAG_REGISTRY[name] = cls
        else:
            if name in RAG_REGISTRY and subname in RAG_REGISTRY[name]:
                raise ValueError('Cannot register duplicate model ({}/{})'.format(name, subname))
            RAG_REGISTRY.setdefault(name, {})
            RAG_REGISTRY[name][subname] = cls
        return cls

    return register_rag_cls


class XinHaiRAGMethodBase:
    method_type: XinHaiRAGMethodTypes
    retriever: XinHaiRAGRetrieverBase
    reranker: XinHaiBaseReranker

    def __init__(self, config):
        self.retriever_type = XinHaiRAGRetrieverTypes(config['retriever'].pop('type'))
        self.retriever = RETRIEVER_REGISTRY[self.retriever_type](config['retriever'])

        self.refiner_type = XinHaiRAGRefinerTypes(config['refiner'].pop('type'))
        self.refiner = REFINER_REGISTRY[self.refiner_type](config['refiner'])

        generator_type = XinHaiRAGGeneratorTypes(config['generator'].pop('type'))
        self.generator = GENERATOR_REGISTRY[generator_type](config['generator'])

    @abstractmethod
    async def __call__(self, *args, **kwargs):
        pass


# automatically import any Python files in the models/ directory
methods_dir = os.path.dirname(__file__)
for file in os.listdir(methods_dir):
    path = os.path.join(methods_dir, file)
    if (
            not file.startswith('_')
            and not file.startswith('.')
            and (file.endswith('.py') or os.path.isdir(path))
    ):
        model_name = file[:file.find('.py')] if file.endswith('.py') else file
        module = importlib.import_module(f'xinhai.rag.methods.{model_name}')
