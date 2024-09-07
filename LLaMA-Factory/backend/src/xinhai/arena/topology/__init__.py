"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Renhao Li
"""
import importlib
import logging
import os

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
