""" Meta Market Maker.
"""
import numpy as np

from typing import Callable

from ..maker import Maker



class MakerM3(Maker):
    """
    """

    def __init__(
        self,
        epsilon: float,
        scale_rewards: Callable[[float], float] = lambda r: r,
        ticksize: float = 0.2,
        low: float = 0.0,
        high: float = 1.0,
        eq: bool = True,
        prices: np.ndarray|None = None,
        action_space: np.ndarray|None = None,
        decimal_places: int = 2,
        name: str = 'exp3',
        seed: int|None = None
    ):
        """
        Parameters
        ----------
        epsilon : float
            Exploration parameter of Exp3.
        scale_rewards : callable[[float], float], default=lambda r: r
            Function to scale raw rewards into a normalized range suitable for Exp3.
            For example, to scale rewards into [0, 1], use a function like:
            `lambda r: (r - min_r) / (max_r - min_r)`.
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
        name : str, default='exp3'
            Name assigned to the agent.
        seed : int or None, default=None
            Seed for the internal random generator.
        """
        super().__init__(ticksize, low, high, eq, prices, action_space, decimal_places, name, seed)

        self.epsilon = epsilon
        """Exploration parameter."""
        self.scale_rewards = scale_rewards
        """Function to scale raw rewards."""
        return
    
    
    class EXP3:
        """
        """
        def __init__(self, n_arms: int, epsilon: float, seed: int|None = None):
            """
            Parameters
            ----------
            n_arms : int
                Number of arms.
            epsilon : float
                Exploration parameter.
            seed : int or None, default=None
                Seed for the internal random generator.
            """
            self.n_arms = n_arms
            self.epsilon = epsilon

            self.weights = np.zeros(self.n_arms, dtype=np.float64)
            """Current weights for each arm."""

            self.rng = np.random.default_rng(seed)
            """Internal random generator."""

            return
