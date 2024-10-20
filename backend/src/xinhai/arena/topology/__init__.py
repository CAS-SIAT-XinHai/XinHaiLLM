"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Renhao Li
"""
import importlib
import logging
import os
from abc import abstractmethod
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)

TOPOLOGY_REGISTRY = {}


def register_topology(name, subname=None):
    def register_topology_cls(cls):
        if subname is None:
            if name in TOPOLOGY_REGISTRY:
                raise ValueError('Cannot register duplicate model ({})'.format(name))
            TOPOLOGY_REGISTRY[name] = cls
        else:
            if name in TOPOLOGY_REGISTRY and subname in TOPOLOGY_REGISTRY[name]:
                raise ValueError('Cannot register duplicate model ({}/{})'.format(name, subname))
            TOPOLOGY_REGISTRY.setdefault(name, {})
            TOPOLOGY_REGISTRY[name][subname] = cls
        return cls

    return register_topology_cls


class BaseTopology:

    def __init__(self, name, graph: nx.DiGraph, nodes=None, start=0, max_turns=0):
        self.name = name
        self.digraph = graph
        self.nodes = nodes or graph.nodes
        self.start = start
        self.max_turns = max_turns

    @classmethod
    def from_config(cls, config: Any):
        edges = []
        for e in config['edges']:
            tail, head = map(int, e.split("->"))
            edges.append((tail, head))
        return cls(
            config['name'],
            nx.DiGraph(edges),
            start=config['start'],
            max_turns=config['max_turns']
        )

    @abstractmethod
    def __call__(self, agents, input_messages, *args, **kwargs):
        raise NotImplementedError


# automatically import any Python files in the models/ directory
topology_dir = os.path.dirname(__file__)
for file in os.listdir(topology_dir):
    path = os.path.join(topology_dir, file)
    if (
            not file.startswith('_')
            and not file.startswith('.')
            and (file.endswith('.py') or os.path.isdir(path))
    ):
        model_name = file[:file.find('.py')] if file.endswith('.py') else file
        module = importlib.import_module(f'xinhai.arena.topology.{model_name}')
