"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Vimos Tan
"""
import logging
from abc import abstractmethod
from typing import List, Dict

from xinhai.arena.agents.base import BaseAgent
from xinhai.arena.topology.base import BaseTopology

logger = logging.getLogger(__name__)


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
    topology: BaseTopology

    max_turns: int = 10
    cnt_turn: int = 0

    controller_address: str = None

    def __init__(self, environment_id, agents: List[BaseAgent], topology, max_turns=10, cnt_turn=0,
                 controller_address=None):
        self.environment_id = environment_id
        self.agents = agents
        self.topology = topology
        self.max_turns = max_turns
        self.cnt_turn = cnt_turn
        self.controller_address = controller_address

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
