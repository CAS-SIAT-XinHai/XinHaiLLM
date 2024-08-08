"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Vimos Tan
"""
from typing import Any, List

import networkx as nx
from xinhai.arena.topology import register_topology
from xinhai.arena.topology.base import BaseTopology


@register_topology("simple")
class SimpleTopology(BaseTopology):
    def __init__(self, graph: nx.DiGraph, nodes=None):
        super().__init__(graph, nodes)
