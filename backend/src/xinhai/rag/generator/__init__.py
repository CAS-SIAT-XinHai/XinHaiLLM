"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Vimos Tan
"""
import importlib
import logging
import os
from typing import List, Dict

from xinhai.types.rag import XinHaiRAGGeneratorTypes

logger = logging.getLogger(__name__)

GENERATOR_REGISTRY = {}


def register_generator(name, subname=None):
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

    def register_generator_cls(cls):
        if subname is None:
            if name in GENERATOR_REGISTRY:
                raise ValueError('Cannot register duplicate model ({})'.format(name))
            GENERATOR_REGISTRY[name] = cls
        else:
            if name in GENERATOR_REGISTRY and subname in GENERATOR_REGISTRY[name]:
                raise ValueError('Cannot register duplicate model ({}/{})'.format(name, subname))
            GENERATOR_REGISTRY.setdefault(name, {})
            GENERATOR_REGISTRY[name][subname] = cls
        return cls

    return register_generator_cls


class XinHaiRAGGeneratorBase:
    r"""Base object of Refiner method"""
    name: XinHaiRAGGeneratorTypes

    def __init__(self, config):
        pass

    def _generate(self, messages: List, *args, **kwargs) -> List[Dict[str, str]]:
        r"""Retrieve topk relevant documents in corpus.

        Return:
            list: contains information related to the document, including:
                contents: used for building index
                title: (if provided)
                text: (if provided)

        """
        pass

    def generate(self, *args, **kwargs):
        return self._generate(*args, **kwargs)


# automatically import any Python files in the models/ directory
generator_dir = os.path.dirname(__file__)
for file in os.listdir(generator_dir):
    path = os.path.join(generator_dir, file)
    if (
            not file.startswith('_')
            and not file.startswith('.')
            and (file.endswith('.py') or os.path.isdir(path))
    ):
        model_name = file[:file.find('.py')] if file.endswith('.py') else file
        module = importlib.import_module(f'xinhai.rag.generator.{model_name}')
