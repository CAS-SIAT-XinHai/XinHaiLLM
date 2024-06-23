"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Vimos Tan
"""
import importlib
import logging
import os

logger = logging.getLogger(__name__)

AGENT_REGISTRY = {}


def register_agent(name, subname=None):
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

    def register_evaluator_cls(cls):
        if subname is None:
            if name in AGENT_REGISTRY:
                raise ValueError('Cannot register duplicate model ({})'.format(name))
            AGENT_REGISTRY[name] = cls
        else:
            if name in AGENT_REGISTRY and subname in AGENT_REGISTRY[name]:
                raise ValueError('Cannot register duplicate model ({}/{})'.format(name, subname))
            AGENT_REGISTRY.setdefault(name, {})
            AGENT_REGISTRY[name][subname] = cls
        return cls

    return register_evaluator_cls


# automatically import any Python files in the models/ directory
agent_dir = os.path.dirname(__file__)
for file in os.listdir(agent_dir):
    path = os.path.join(agent_dir, file)
    if (
            not file.startswith('_')
            and not file.startswith('.')
            and (file.endswith('.py') or os.path.isdir(path))
    ):
        model_name = file[:file.find('.py')] if file.endswith('.py') else file
        module = importlib.import_module(f'xinhai.arena.agents.{model_name}')
