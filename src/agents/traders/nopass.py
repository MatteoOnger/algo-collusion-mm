import numpy as np

from typing import Dict, List

from ..agent import Agent
from ...envs import GMEnv



class NoPassTrader(Agent):
    """
    Trader agent for the GM environment.

    The trader observes the current market state and chooses between two actions:
    buy or sell. The decision rule is based on the comparison between how favorable
    the market is for buying vs selling, relative to the true value of the asset.

    Attributes
    ----------
    action_space : list
        All possible available actions.
    n_arms : int
        Number of actions (arms) in the action space.
    """

    def __init__(self, name: str = 'trader', seed: int | None = None):
        """
        Parameters
        ----------
        name : str, default='trader'
            Unique identifier for the agent.
        seed : int or None, default=None
            Seed for the internal random generator.
        """
        super().__init__(name)

        self._rng = np.random.default_rng(seed)
        self._action_space = np.array([GMEnv.TraderAction.BUY, GMEnv.TraderAction.SELL])
        return


    @property
    def action_space(self) -> List:
        return self._action_space


    def act(self, observation: Dict) -> Dict:
        """
        Decide to BUY, SELL based on the current market observation.

        Parameters
        ----------
        observation : dict
            A dictionary containing:
            - 'true_value': float.
            - 'min_ask_price': float.
            - 'max_bid_price': float.

        Returns
        -------
        action : dict
            A dictionary with key 'operation' and value `GM2Enc.TraderAction`.
        """
        true_value = observation['true_value']
        min_ask = observation['min_ask_price']
        max_bid = observation['max_bid_price']

        if np.isclose(true_value - min_ask, max_bid - true_value):
            action = self._rng.choice([GMEnv.TraderAction.BUY, GMEnv.TraderAction.SELL])
        elif true_value - min_ask > max_bid - true_value:
            action = GMEnv.TraderAction.BUY
        else:
            action = GMEnv.TraderAction.SELL
        
        self.history.record_action(action)
        return {
            'operation': action 
        }


    def update(self, reward: float, info: Dict) -> None:
        self.history.record_reward(reward)
        return


    def reset(self) -> None:
        super().reset()
        return
