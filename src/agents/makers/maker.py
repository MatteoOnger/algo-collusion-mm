import numpy as np

from abc import abstractmethod
from typing import Dict

from ..agent import Agent



class Maker(Agent):
    """
    Abstract class for a maker agent.

    Attributes
    ----------
    ticksize : float, default=0.02
        Minimum increment for prices in the action space.
    low : float, default=0.0
        Minimum price allowed.
    high : float, default=1.0
        Maximum price allowed.
    prices : np.ndarray
        Discrete set of possible prices.
    action_space : np.ndarray
        All possible (ask_price, bid_price) pairs.
    decimal_places : int, default=2
        Number of decimal places to which rewards and actions are rounded.
    last_action : tuple or int or None
        Last chosen action.
        It is None if no action has been taken yet, if it is an integer, it represents the index 
        of the action in the action space, otherwise it is a tuple containing the index of the action
        and additional information (e.g., the probability of selection).
    
    Notes
    -----
    If both `prices` and `action_space` are provided, a ValueError is raised. If neither is provided,
    `prices` is generated using `ticksize`, `low`, and `high`. If only one of them is provided, the other
    is derived from it.
    """

    def __init__(
        self,
        ticksize: float = 0.2,
        low: float = 0.0,
        high: float = 1.0,
        eq: bool = True,
        prices: np.ndarray|None = None,
        action_space: np.ndarray|None = None,
        decimal_places: int = 2,
        name: str = 'maker',
        seed: int|None = None
    ):
        """
        Parameters
        ----------
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
            Discrete set of possible prices.
        action_space : np.ndarray or None, default=None
            All possible (ask_price, bid_price) pairs.
        decimal_places : int, default=2
            Number of decimal places to which rewards are rounded.
        name : str, default='maker'
            Name assigned to the agent.
        seed : int or None, default=None
            Seed for the internal random generator.
        
        Raises
        ------
        ValueError
            If both `prices` and `action_space` are provided.
        """
        super().__init__(name, seed)
        self.ticksize = ticksize
        self.low = low
        self.high = high
        self.decimal_places = decimal_places

        op = (lambda a, b: a >= b) if eq else (lambda a, b: a > b)

        if prices is not None and action_space is not None:
            raise ValueError('Cannot specify both `prices` and `action_space`')
        elif prices is not None:
            self.prices = np.round(prices, decimal_places)
            self._action_space = np.array([(ask, bid) for ask in self.prices for bid in self.prices if op(ask, bid)])
        elif action_space is not None:
            self._action_space = np.round(action_space, decimal_places)
            self.prices = np.unique(action_space)
        else:
            self.prices =  np.round(np.arange(self.low, self.high + self.ticksize, self.ticksize), decimal_places)
            self._action_space = np.array([(ask, bid) for ask in self.prices for bid in self.prices if op(ask, bid)])
        
        self.last_action = None
        return


    @property
    def action_space(self) -> np.ndarray:
        return self._action_space


    def price_to_index(self, prices :np.ndarray) -> np.ndarray:
        """
        Converts an array of price to their corresponding indices based on the price list.

        This method takes a 1D numpy array of prices and maps each action to an index based on
        its position in the predefined price list.

        Parameters:
        -----------
        prices : np.ndarray
            A 1D numpy array representing the prices.

        Returns:
        --------
        : np.ndarray
            The function returns a numpy array of indices corresponding to each price.

        Raises:
        -------
        ValueError
            If the input array has more than one dimensions.
        """
        if prices.ndim > 1:
            raise ValueError(f'unable to process arrays of shape {prices.shape}')
        
        indexes = np.where(
            np.broadcast_to(self.prices, (len(prices),) + self.prices.shape) == prices[:, None]
        )[1]
        return indexes


    def reset(self) -> None:
        super().reset()
        self.last_action = None
        return


    @abstractmethod
    def act(self, observation: Dict) -> Dict:
        """
        Select an ask-bid strategy.

        Parameters
        ----------
        observation : dict
            Not used for this maker agent.

        Returns
        -------
        action : dict
            A dictionary containing:
            - 'ask_price': the ask price proposed by the agent.
            - 'bid_price': the bid price proposed by the agent.
        """
        pass