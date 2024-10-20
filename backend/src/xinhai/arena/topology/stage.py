"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Renhao Li
"""
import re
from typing import Any, List

import networkx as nx

from xinhai.arena.topology import register_topology, BaseTopology


@register_topology("stage")
class StageTopology(BaseTopology):
    def __init__(self, config_in_each_stage: List):
        self.stages = [c[0] for c in config_in_each_stage]
        self.env_status = [c[1] for c in config_in_each_stage]
        self.diagraph = [c[2] for c in config_in_each_stage]
        self.start_nodes = [c[3] for c in config_in_each_stage]
        self.budget = [c[4] for c in config_in_each_stage]
        self.ref_info = [c[5] for c in config_in_each_stage]
        self.iter_num = [c[6] for c in config_in_each_stage]

        self.nodes = [c[2].nodes for c in config_in_each_stage]

    @classmethod
    def from_config(cls, config: Any):
        config_in_each_stage = []
        for stage_name, value in config.items():
            status = value['status']
            start = value['start']
            budget = value['budget'] if 'budget' in value.keys() else 100
            edges = []
            for e in value['edges']:
                tail, head = map(int, e.split("->"))
                edges.append((head, tail))
            ref_info = {}
            if 'ref_info' in value:
                for r in value['ref_info']:
                    ref_target, ref_method, ref_source, ref_query, use_ref_cache = r.split('->')
                    ref_worker = re.findall(r'\[(.*?)\]', ref_source)[0]
                    ref_source = ref_source.split(ref_worker)[0].strip('[').strip(']')
                    if use_ref_cache == "use_cache":
                        use_ref_cache = True
                    elif use_ref_cache == "no_cache":
                        use_ref_cache = False
                    else:
                        raise NotImplementedError
                    ref_info[int(ref_target)] = {
                        "ref_method": ref_method,
                        "ref_worker": ref_worker,
                        "ref_source": ref_source,
                        "ref_query": ref_query,
                        "use_ref_cache": use_ref_cache
                    }
            iter_num = value['iter_num'] if 'iter_num' in value else 1
            config_in_each_stage.append((stage_name, status, nx.DiGraph(edges), start, budget, ref_info, iter_num))

        return cls(config_in_each_stage)
