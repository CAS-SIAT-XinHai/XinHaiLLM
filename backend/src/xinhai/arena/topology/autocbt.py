from typing import Any, List

import networkx as nx
from xinhai.arena.topology import register_topology
from xinhai.arena.topology.base import BaseTopology


@register_topology("autocbt")
class SimpleTopology(BaseTopology):
    def __init__(self, graph: nx.DiGraph, nodes=None):
        super().__init__(graph, nodes)
