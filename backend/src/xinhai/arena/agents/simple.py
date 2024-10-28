"""
Copyright (c) CAS-SIAT-XinHai.
Licensed under the CC0-1.0 license.

XinHai stands for [Sea of Minds].

Authors: Vimos Tan
         yangdi
Date: 2024-07-19 17:22:57
LastEditTime: 2024-07-19 17:28:20
"""
import logging
import uuid
from datetime import datetime
from typing import List

from xinhai.arena.agents import register_agent, BaseAgent
from xinhai.types.arena import XinHaiArenaAgentTypes
from xinhai.types.message import XinHaiChatMessage
from xinhai.types.routing import XinHaiRoutingMessage

logger = logging.getLogger(__name__)


@register_agent(XinHaiArenaAgentTypes.SIMPLE)
class SimpleAgent(BaseAgent):
    agent_type = XinHaiArenaAgentTypes.SIMPLE

    def reset(self) -> None:
        pass

    def get_history(self, target_agents=None):
        dialogue_context = []
        if not target_agents:
            for i, message in enumerate(self.memory.short_term_memory.messages[::-1]):
                if not message.files:
                    dialogue_context.insert(0, f"Agent-{message.senderId} {message.username}: {message.content}")
                else:
                    dialogue_context.insert(0,
                                            f"Agent-{message.senderId} {message.username}: [IMG] {message.content}")

                if len(dialogue_context) > self.summary_chunk_size:
                    break
        else:
            target_agent_ids = [str(n.agent_id) for n in target_agents]
            for i, message in enumerate(self.memory.short_term_memory.messages[::-1]):
                if message.senderId in target_agent_ids or any([n in target_agent_ids for n in message.receiverIds]):
                    if not message.files:
                        dialogue_context.insert(0, f"Agent-{message.senderId} {message.username}: {message.content}")
                    else:
                        dialogue_context.insert(0,
                                                f"Agent-{message.senderId} {message.username}: [IMG] {message.content}")

                    if len(dialogue_context) > self.summary_chunk_size:
                        break
        return dialogue_context

    def step(
            self,
            routing_message_in: XinHaiRoutingMessage,
            routing_message_out: XinHaiRoutingMessage,
            target_agents: List[BaseAgent], **kwargs
    ):
        chat_summary = self.get_summary()
        chat_history = '\n'.join(self.get_history(target_agents))
        target_agent_names = ", ".join([f"Agent-{n.agent_id} {n.name}" for n in target_agents])

        prompt = self.prompt_template.format(chat_history=chat_history,
                                             chat_summary=chat_summary,
                                             role_description=self.role_description,
                                             routing_type=routing_message_out.routing_type.routing_name,
                                             target_agent_names=target_agent_names)
        role, content = self.complete_conversation(prompt)

        t = datetime.now()

        return XinHaiChatMessage(
            indexId=uuid.uuid4().hex,
            content=content,
            senderId=str(self.agent_id),
            receiverIds=[str(a.agent_id) for a in target_agents],
            username=self.name,
            role="user",
            date=t.strftime("%a %b %d %Y"),
            timestamp=t.strftime("%H:%M"),
        )
