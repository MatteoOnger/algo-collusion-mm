import numpy as np

from typing import Dict, Literal

from .trader import Trader
from ...envs import GMEnv



class NoPassTrader(Trader):
    """
    Trader agent for the GM environment.

    The trader observes the current market state and chooses between two actions:
    buy or sell. The decision rule is based on the comparison between how favorable
    the market is for buying vs selling, relative to the true value of the asset.
    
    If both actions are equally favorable (a tie), the agent resolves the decision
    using a specified tie-breaking strategy.
    
    Attributes
    ----------
    tie_breaker : {'buy', 'sell', 'rand', 'alt'}
        Rule used to resolve ties between equally favorable actions.
        - 'buy': always prefer buying in case of tie.
        - 'sell': always prefer selling in case of tie.
        - 'rand': break ties randomly.
        - 'alt': alternate between buy and sell on each tie.
    last_action : GMEnv.TraderAction or None
        Stores the last action taken (used for 'alt' tie-breaker).
    """

    def __init__(self, tie_breaker: Literal['buy', 'sell', 'rand', 'alt'] = 'rand', name: str = 'no_pass_trader', seed: int|None = None):
        """ 
        Parameters
        ----------
        tie_breaker : {'buy', 'sell', 'rand', 'alt'}
            Strategy to resolve ties when both actions are equally favorable.
            - 'buy': always choose to buy.
            - 'sell': always choose to sell.
            - 'rand': choose randomly between buy and sell.
            - 'alt': alternate between buy and sell on each tie.
        name : str, default='agent'
            Unique identifier for the agent.
        seed : int or None, default=None
            Seed for the internal random generator.
        """
        super().__init__(name, seed)
        self.tie_breaker = tie_breaker
        """ Strategy to resolve ties.
        """        
        self.last_action = None
        """ Last chosen action.
        """

        self._action_space = np.array([GMEnv.TraderAction.BUY, GMEnv.TraderAction.SELL])
        return


    @property
    def action_space(self) -> np.ndarray:
        return self._action_space


    def act(self, observation: Dict) -> Dict:
        """
        Decides to BUY, SELL based on the current market observation.

        Parameters
        ----------
        observation : dict of str to float
            A dictionary containing:
            - 'true_value': float.
            - 'min_ask_price': float.
            - 'max_bid_price': float.

        Returns
        -------
        action : dict of str to GM2Enc.TraderAction
            A dictionary with key 'operation' and value `GM2Enc.TraderAction`.
        """
        true_value = observation['true_value']
        min_ask = observation['min_ask_price']
        max_bid = observation['max_bid_price']

        if np.isclose(true_value - min_ask, max_bid - true_value):
            if self.tie_breaker == 'rand' or (self.tie_breaker == 'alt' and self.last_action is None):
                action = self._rng.choice([GMEnv.TraderAction.BUY, GMEnv.TraderAction.SELL])
            elif self.tie_breaker == 'buy' or (self.tie_breaker == 'alt' and self.last_action == GMEnv.TraderAction.SELL):
                action = GMEnv.TraderAction.BUY
            else:
                action = GMEnv.TraderAction.SELL
            self.last_action = action
        elif true_value - min_ask > max_bid - true_value:
            action = GMEnv.TraderAction.BUY
        else:
            action = GMEnv.TraderAction.SELL
        
        self.history.record_action(action.value)
        return {
            'operation': action 
        }


    def reset(self) -> None:
        super().reset()
        self.last_action = None
        return
