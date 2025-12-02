""" Abstract trader.
"""
from typing import Dict

from ..agent import Agent
from ...enums import AgentType


class Trader(Agent):
    """
    Abstract class for a trader agent.

    Notes
    -----
    Traders are stateless agents, so the history is only used to
    keep track of actions taken and the rewards received.
    """

    type: AgentType = AgentType.ABSTRACT


    def __init__(self, name: str = 'trader', seed: int|None = None):
        super().__init__(name, seed)
        return


    def update(self, reward: float, info: Dict) -> None:
        self.history.record_reward(reward)
        return


    def reset(self) -> None:
        super().reset()
        return
