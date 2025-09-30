import numpy as np

from typing import Dict

from .maker import Maker



class MakerEXP3(Maker):
    """
    Market maker for the GM environment based on the Exp3.

    This agent maintains a discrete set of possible (ask, bid) strategies and selects one at
    each step according to a probability distribution computed by the Exp3 algorithm.
    The distribution balances exploitation of historically successful strategies and
    exploration of less tried strategies, controlled by the `epsilon` parameter.

    Attributes
    ----------
    epsilon : float
        Exploration parameter of Exp3, in the range (0, 1].
    weights : np.ndarray
        Current Exp3 weights for each arm, initially all ones.
    probs : np.ndarray
        Current probability distribution over arms, computed from weights and epsilon.
    
    Notes
    -----
    Trader are stateless agents, so the history is only used to
    keep track of actions taken and the rewards received.
    """

    def __init__(
        self,
        epsilon: float,
        ticksize: float = 0.2,
        low: float = 0.0,
        high: float = 1.0,
        eq: bool = True,
        prices: np.ndarray|None = None,
        action_space: np.ndarray|None = None,
        decimal_places: int = 2,
        name: str = 'maker_exp3',
        seed: int|None = None
    ):
        """
        Parameters
        ----------
        epsilon : float
            Exploration parameter of Exp3, in the range (0, 1].
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
        """
        super().__init__(ticksize, low, high, eq, prices, action_space, decimal_places, name, seed)
        self.epsilon = epsilon
        self.weights = np.ones(self.n_arms)
        return


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


    @staticmethod
    def compute_epsilon(n_arms: int, n_episodes: int) -> float:
        """
        Compute the learning rate epsilon for the Exp3 algorithm.
        
        The formula used is derived from the theoretical guarantees of the Exp3
        algorithm to minimize regret.

        Parameters
        ----------
        n_arms : int
            The number of arms (actions) in the bandit problem.
        n_episodes : int
            The total number of episodes (rounds) to be played.

        Returns
        -------
        epsilon : float
            The calculated optimal learning rate, bounded between 0 and 1.
        """
        return min(1, np.sqrt((n_arms * np.log(n_arms)) / ((np.e - 1) * n_episodes)))


    def act(self, observation: Dict) -> Dict:
        arm_idx = self._rng.choice(self.n_arms, p=self.probs)

        strategy = self.action_space[arm_idx]
        self.last_action = (arm_idx, self.probs[arm_idx])

        self.history.record_action(strategy)
        return {
            'ask_price': strategy[0],
            'bid_price': strategy[1]
        }


    def update(self, reward: float, info: Dict) -> None:
        if self.last_action is None:
            return
        
        arm_idx, prob = self.last_action
        self.weights[arm_idx] = self.weights[arm_idx] * np.exp(self.epsilon * (reward/prob) / self.n_arms)

        self.history.record_reward(reward)
        self.last_action = None
        return


    def reset(self) -> None:
        super().reset()
        self.weights = np.ones(self.n_arms)
        return
