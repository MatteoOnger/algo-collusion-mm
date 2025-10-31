""" EXP3 maker.
"""
import numpy as np

from typing import Callable, Dict

from .maker import Maker



class MakerEXP3(Maker):
    """
    Market maker for the GM environment based on the Exp3 algorithm.

    This agent maintains a discrete set of possible (ask, bid) strategies and selects one at
    each step according to a probability distribution computed using Exp3. The algorithm 
    balances exploitation of historically high-reward strategies and exploration of others, 
    controlled by the `epsilon` parameter.

    The agent receives a reward at each step and uses a user-provided `scale_rewards` 
    function to normalize the reward into a suitable range (typically [0, 1]) as required 
    by Exp3. This allows flexible reward normalization depending on the environment.

    Attributes
    ----------
    epsilon : float
        Exploration parameter of Exp3.
    scale_rewards : callable[[float], float]
        A function that scales raw rewards into a normalized range (e.g., [0, 1]).
    weights : np.ndarray
        Current Exp3 weights for each arm.
    probs : np.ndarray
        Probability distribution over actions, computed from weights and epsilon.
    
    Notes
    -----
    - Code based on the adversarial bandit framework described in Lattimore & Szepesvári (2020).
    
    References
    ----------
    - Lattimore, T., & Szepesvári, C. (2020). Bandit algorithms. Cambridge University Press.
    - Auer, P., Cesa-Bianchi, N., Freund, Y., & Schapire, R. E. (2002).
    The nonstochastic multiarmed bandit problem. SIAM journal on computing, 32(1), 48-77.
    - Auer, P., Cesa-Bianchi, N., Freund, Y., & Schapire, R. E. (1995, October).
    Gambling in a rigged casino: The adversarial multi-armed bandit problem.
    In Proceedings of IEEE 36th annual foundations of computer science (pp. 322-331). IEEE.
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
            Exploration parameter of Exp3, in the range (0, 1].
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
        self.weights = np.zeros(self.n_arms, dtype=np.float64)
        """Current weights for each arm."""
        return


    @property
    def probs(self) -> np.ndarray:
        """Current probability distribution over actions."""
        x = self.epsilon * self.weights
        x_stable = x - np.max(x)
        return np.exp(x_stable) / np.sum(np.exp(x_stable))


    @staticmethod
    def compute_epsilon(n_arms: int, n_episodes: int) -> float:
        """
        Compute the ideal learning rate `epsilon` for the Exp3 algorithm.
        
        The formula used is derived from the theoretical guarantees of the Exp3
        algorithm to minimize regret.

        Parameters
        ----------
        n_arms : int
            The number of arms (actions) in the bandit problem.
        n_episodes : int
            Number of episodes.

        Returns
        -------
        epsilon : float
            The calculated optimal learning rate.
        """
        return np.sqrt(np.log(n_arms) / (n_arms * n_episodes))


    def act(self, observation: Dict) -> Dict[str, float]:
        arm_idx = self._rng.choice(self.n_arms, p=self.probs.astype(np.float64))

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

        scaled_reward = self.scale_rewards(reward)

        arm_idx, prob = self.last_action
        self.weights[arm_idx] += 1 - (1 - scaled_reward) / prob

        self.history.record_reward(reward)
        self.last_action = None
        return


    def reset(self) -> None:
        super().reset()
        self.weights = np.zeros(self.n_arms, dtype=np.float64)
        return
