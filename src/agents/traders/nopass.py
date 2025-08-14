from typing import Dict

from ..agent import Agent
from ...envs import GMEnv



class NoPassTrader(Agent):
    """
    Trader agent for the GM environment.

    The trader observes the current market state and chooses between two actions:
    buy or sell. The decision rule is based on the comparison between how favorable
    the market is for buying vs selling, relative to the true value of the asset.
    """

    def __init__(self, name: str = 'trader'):
        """
        Parameters
        ----------
        name : str, default='trader'
            Unique identifier for the agent.
        """
        super().__init__(name)
        self.action_space = [GMEnv.TraderAction.BUY, GMEnv.TraderAction.SELL]
        self.n_arms = len(self.action_space)
        return


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
        true_value = observation["true_value"]
        min_ask = observation["min_ask_price"]
        max_bid = observation["max_bid_price"]

        if true_value - min_ask >= max_bid - true_value:
            action = {"operation": GMEnv.TraderAction.BUY}
        else:
            action = {"operation": GMEnv.TraderAction.SELL}
        return action


    def update(self, reward: float) -> None:
        """
        Not used for this trader agent.
        """
        return


    def reset(self) -> None:
        """
        Not used for this trader agent.
        """
        return
