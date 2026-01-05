""" Tit-for-tat maker.
"""
import numpy as np

from typing import Dict

from ....enums import AgentType
from ..maker import Maker



class MakerTFT(Maker):
    """
    Market-making agent for the GM environment based on a tit-for-tat (TFT) policy.

    This agent, at each step, selects the strategy that matches the most common
    action taken by the other agents in the previous step.

    The policy is fully deterministic given the observed actions of the other
    agents: the probability distribution over strategies is set to put all mass
    on the most frequently observed opponent action. Rewards are recorded for
    analysis purposes but do not influence future action selection.
    """

    type: AgentType = AgentType.MAKER_I


    def __init__(
        self,
        action_values_attr: str = 'probs',
        ticksize: float = 0.2,
        low: float = 0.0,
        high: float = 1.0,
        eq: bool = True,
        prices: np.ndarray|None = None,
        action_space: np.ndarray|None = None,
        decimal_places: int = 2,
        name: str = 'hedge',
        seed: int|None = None
    ):
        """
        Parameters
        ----------
        action_values_attr : str, default='probs'
            Name of the property that provides the action value representation.
        ticksize : float, default=0.2
            Minimum increment for prices in the action space.
        low : float, default=0.0
            Minimum price allowed.
        high : float, default=1.0
            Maximum price allowed.
        eq : bool, default=True
            Allow the bid price to be equal to the ask price.
            Not used if `action_space` is given.
        prices : np.ndarray or None, default=None
            Set of possible prices.
        action_space : np.ndarray or None, default=None
            All possible (ask_price, bid_price) pairs.
        decimal_places : int, default=2
            Number of decimal places to which rewards and actions are rounded.
        name : str, default='hedge'
            Name assigned to the agent.
        seed : int or None, default=None
            Seed for the internal random generator.
        """
        super().__init__(action_values_attr, ticksize, low, high, eq, prices, action_space, decimal_places, name, seed)

        self.probs = np.ones(self.n_arms) / self.n_arms
        """Current probability distribution over actions."""
        return


    def act(self, observation: Dict) -> Dict[str, float]:
        arm_idx = self._rng.choice(self.n_arms, p=self.probs)

        strategy = self.action_space[arm_idx]
        self.last_action = arm_idx

        self.history.record_action(strategy)
        return {
            'ask_price': strategy[0],
            'bid_price': strategy[1]
        }


    def update(self, reward: float, info: Dict) -> None:
        """
        Update the agent's internal state based on the reward received
        and the actions taken by other agents in the environment.

        Parameters
        ----------
        reward : float
            The reward assigned to the agent for the most recent action.
        info : dict of str
            A dictionary containing environment feedback, with keys:
                - 'agent_index' (int): Index of this agent in the joint action array.
                - 'actions' (np.ndarray): Array of actions taken by all agents,
                  including this one, in the previous step.
        """
        if self.last_action is None:
            return

        agent_index = info['agent_index']
        
        actions = np.delete(info['actions'], agent_index, axis=0)
        actions, freqs = np.unique(actions, axis=0, return_counts=True)
        most_common_action = actions[np.argmax(freqs)]
        
        self.probs = np.zeros(self.n_arms)
        self.probs[self.action_to_index(most_common_action)] = 1.0

        self.history.record_reward(reward)
        self.last_action = None   
        return


    def reset(self) -> None:
        super().reset()
        return
