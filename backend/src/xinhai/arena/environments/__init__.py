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
from typing import List

from xinhai.arena.agents import BaseAgent
from xinhai.arena.topology import BaseTopology
from xinhai.types.message import XinHaiChatMessage

logger = logging.getLogger(__name__)

ENVIRONMENT_REGISTRY = {}


def register_environment(name, subname=None):
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

    def register_environment_cls(cls):
        if subname is None:
            if name in ENVIRONMENT_REGISTRY:
                raise ValueError('Cannot register duplicate model ({})'.format(name))
            ENVIRONMENT_REGISTRY[name] = cls
        else:
            if name in ENVIRONMENT_REGISTRY and subname in ENVIRONMENT_REGISTRY[name]:
                raise ValueError('Cannot register duplicate model ({}/{})'.format(name, subname))
            ENVIRONMENT_REGISTRY.setdefault(name, {})
            ENVIRONMENT_REGISTRY[name][subname] = cls
        return cls

    return register_environment_cls


class BaseEnvironment:
    """
    Base class for environment.

    Args:
        agents: List of agents
        max_turns: Maximum number of turns
        cnt_turn: Current turn number
    """
    environment_id: str
    agents: List[BaseAgent]
    topologies: List[BaseTopology]
    messages: List[XinHaiChatMessage] = []

    def __init__(self, environment_id, agents: List[BaseAgent], topologies: List[BaseTopology], controller_address,
                 max_turns=10,
                 cnt_turn=0):
        self.environment_id = environment_id
        self.agents = agents
        self.topologies = topologies
        self.max_turns = max_turns
        self.cnt_turn = cnt_turn
        self.controller_address = controller_address

        for a in self.agents:
            self.messages.extend(a.memory.short_term_memory.messages)

    @abstractmethod
    async def step(self, *args, **kwargs) -> None:
        """Run one step of the environment"""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the environment"""
        pass

    def is_done(self) -> bool:
        """Check if the environment is done"""
        return self.cnt_turn >= self.max_turns


# automatically import any Python files in the models/ directory
environment_dir = os.path.dirname(__file__)
for file in os.listdir(environment_dir):
    path = os.path.join(environment_dir, file)
    if (
            not file.startswith('_')
            and not file.startswith('.')
            and (file.endswith('.py') or os.path.isdir(path))
    ):
        model_name = file[:file.find('.py')] if file.endswith('.py') else file
        module = importlib.import_module(f'xinhai.arena.environments.{model_name}')
