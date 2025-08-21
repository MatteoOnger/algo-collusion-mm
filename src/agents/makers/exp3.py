import numpy as np

from typing import Dict, List

from ..agent import Agent



class MakerEXP3(Agent):
    """
    Market maker for the GM environment based on the Exp3.

    This agent maintains a discrete set of possible (ask, bid) strategies and selects one at
    each step according to a probability distribution computed by the Exp3 algorithm.
    The distribution balances exploitation of historically successful strategies and
    exploration of less tried strategies, controlled by the `gamma` parameter.

    Attributes
    ----------
    prices : np.ndarray
        Discrete set of possible prices from `low` to `high` with spacing `ticksize`.
    action_space : list of tuple of float
        All possible (ask_price, bid_price) pairs such that `bid_price <= ask_price`.
    n_arms : int
        Number of actions (arms) in the action space.
    weights : np.ndarray
        Current Exp3 weights for each arm, initially all ones.
    last_action : tuple or None
        Last chosen arm index and its associated selection probability.
    """

    def __init__(
        self,
        epsilon: float,
        ticksize: float = 0.05,
        low: float = 0.0,
        high: float = 1.0,
        name: str = 'maker',
        decimal_places: int = 2,
        seed: int | None = None
    ):
        """
        Parameters
        ----------
        epsilon : float
            Exploration parameter of Exp3, in the range (0, 1].
        ticksize : float, default=0.05
            Minimum increment for prices in the action space.
        low : float, default=0.0
            Minimum price allowed.
        high : float, default=1.0
            Maximum price allowed.
        name : str, default='maker'
            Name assigned to the agent.
        decimal_places : int, default=2
            Number of decimal places to which rewards are rounded.
        seed : int or None, default=None
            Seed for the internal random generator.
        """
        super().__init__(name)

        self.epsilon = epsilon
        self.ticksize = ticksize
        self.low = low
        self.high = high
        self.seed = seed

        self._rng = np.random.default_rng(seed)

        self.prices =  np.round(np.arange(self.low, self.high + self.ticksize, self.ticksize), decimal_places)
        self._action_space = np.array([(ask, bid) for ask in self.prices for bid in self.prices if bid <= ask])

        self.weights = np.ones(self.n_arms)
        self.last_action = None
        return


    @property
    def action_space(self) -> List:
        return self._action_space


    @property
    def probs(self) -> np.ndarray:
        """
        Compute the current probability distribution over actions.

        Returns
        -------
        probs : np.ndarray
            Array of shape (n_arms,) representing probability of selecting
            each arm according to the Exp3 formula.
        """
        return (1 - self.epsilon) * self.weights / np.sum(self.weights) + self.epsilon / self.n_arms


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
        arm_idx = self._rng.choice(self.n_arms, p=self.probs)

        strategy = self.action_space[arm_idx]
        self.last_action = (arm_idx, self.probs[arm_idx])

        self.history.record(strategy)
        return {
            'ask_price': strategy[0],
            'bid_price': strategy[1]
        }


    def update(self, reward: float) -> None:
        if self.last_action is None:
            return
        
        arm_idx, prob = self.last_action
        self.weights[arm_idx] = self.weights[arm_idx] * np.exp(self.epsilon * (reward/prob) / self.n_arms)

        self.last_action = None
        return


    def reset(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self.weights = np.ones(self.n_arms)
        self.last_action = None
        return
