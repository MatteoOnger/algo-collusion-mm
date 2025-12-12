""" Informed Memoryless Q-learning makers.
"""
import numpy as np

from typing import Dict

from ....enums import AgentType
from ..uninformed.mlql import MakerMLQL



class MakerIMLQL(MakerMLQL):
    """
    Market maker for the GM environment based on an informed memoryless Q-learning approach.

    This agent is similar to the `MakerMLQL` agent, but, after each action, it updates all 
    its Q-values using additional information from the environment (provided in the `info` dictionary)
    to estimate the reward it would have received if it had taken a different action.
    """

    type: AgentType = AgentType.MAKER_I


    def update(self, reward: float, info: Dict[str, np.ndarray]) -> None:
        """
        Update the agent's internal state based on the reward received
        and additional information from the enivironment.

        Parameters
        ----------
        reward : float
            The reward assigned to the agent for the most recent action.
        info : dict of str
            A dictionary containing environment feedback, with keys:
                - 'rewards' (np.ndarray): Rewards corresponding to each action
                    in the action space of this agent.
        """
        if self.last_action is None:
            return

        rewards = info['rewards']

        for idx, reward in enumerate(rewards):
            self.Q[idx] += self.alpha * (reward + self.gamma * np.max(self.Q) - self.Q[idx])

        self.epsilon = self.scheduler(self.epsilon_init, self.epsilon_decay_rate, self._t)
        self.history.record_reward(reward)
        self.last_action = None
        return
