"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Ancheng Xu
         Vimos Tan
"""
import logging

from xinhai.arena.environments import register_environment, BaseEnvironment

logger = logging.getLogger(__name__)


@register_environment("chatdev")
class ChatdevEnvironment(BaseEnvironment):
    async def step(self):
        """Run one step of the environment"""
        for topology in self.topologies:
            for turn_id, (targets, message) in enumerate(topology(self.agents, input_messages=None)):
                if turn_id > topology.max_turns:
                    break
                for a in targets:
                    a.update_memory([message])

    def reset(self) -> None:
        """Reset the environment"""
        self.cnt_turn = 0
        for agent in self.agents:
            agent.reset()

    def is_done(self) -> bool:
        """Check if the environment is done"""
        return self.cnt_turn >= self.max_turns
