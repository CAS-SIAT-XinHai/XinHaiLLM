"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Vimos Tan
"""
from typing import Any

import networkx as nx


class BaseTopology:

    def __init__(self, digraph: nx.DiGraph, nodes=None):
        self.digraph = digraph
        self.nodes = nodes or digraph.nodes

    @classmethod
    def from_config(cls, config: Any):
        edges = []
        for e in config['edges']:
            tail, head = map(int, e.split("->"))
            edges.append((head, tail))
        return cls(nx.DiGraph(edges))
