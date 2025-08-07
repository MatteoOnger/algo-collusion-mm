from typing import Dict

from .agent import Agent
from ..envs import GM2Env



class Trader(Agent):
    """
    Trader agent for the GM2 environment.

    The trader observes the current market state and chooses between two actions:
    buying or selling. The decision rule is based on the comparison between how 
    favorable the market is for buying vs selling, relative to the true value of the asset.

    Parameters
    ----------
    name : str, default='agent'
        Unique identifier for the trader (e.g., "trader_0").
    """

    def act(self, observation: Dict) -> Dict:
        """
        Decide to BUY or SELL based on the current market observation.

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
            action = {"operation": GM2Env.TraderAction.BUY}
        else:
            action = {"operation": GM2Env.TraderAction.SELL}
        return action


    def reset(self) -> None:
        """
        Reset internal state (not used for this trader agent).
        """
        pass
