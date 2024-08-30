"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Vimos Tan
"""
import importlib
import logging
import os
from string import Template
from typing import List, Dict

from xinhai.types.rag import XinHaiRAGRefinerTypes, XinHaiRAGRetrievedResult

logger = logging.getLogger(__name__)

REFINER_REGISTRY = {}


def register_refiner(name, subname=None):
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

    def register_refiner_cls(cls):
        if subname is None:
            if name in REFINER_REGISTRY:
                raise ValueError('Cannot register duplicate model ({})'.format(name))
            REFINER_REGISTRY[name] = cls
        else:
            if name in REFINER_REGISTRY and subname in REFINER_REGISTRY[name]:
                raise ValueError('Cannot register duplicate model ({}/{})'.format(name, subname))
            REFINER_REGISTRY.setdefault(name, {})
            REFINER_REGISTRY[name][subname] = cls
        return cls

    return register_refiner_cls


class XinHaiRAGRefinerBase:
    r"""Base object of Refiner method"""
    name: XinHaiRAGRefinerTypes
    share_generator = False

    def __init__(self, config):
        self.system_prompt_template = Template(config['system_prompt_template'])
        self.user_prompt_template = Template(config['user_prompt_template'])
        self.reference_template = Template(config['reference_template'])

    def _refine(self, query: str, retrieval_result: XinHaiRAGRetrievedResult, *args, **kwargs) -> List[Dict[str, str]]:
        r"""Retrieve topk relevant documents in corpus.

        Return:
            list: contains information related to the document, including:
                contents: used for building index
                title: (if provided)
                text: (if provided)

        """
        pass

    def refine(self, *args, **kwargs):
        return self._refine(*args, **kwargs)


# automatically import any Python files in the models/ directory
refiner_dir = os.path.dirname(__file__)
for file in os.listdir(refiner_dir):
    path = os.path.join(refiner_dir, file)
    if (
            not file.startswith('_')
            and not file.startswith('.')
            and (file.endswith('.py') or os.path.isdir(path))
    ):
        model_name = file[:file.find('.py')] if file.endswith('.py') else file
        module = importlib.import_module(f'xinhai.rag.refiner.{model_name}')
