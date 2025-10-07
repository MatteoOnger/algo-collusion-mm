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

    The agent receives a reward at each round and scales it into the [0, 1] interval
    using the `min_reward` and `max_reward` parameters. This normalization is required
    by the Exp3 algorithm to ensure correct weight updates.

    Attributes
    ----------
    epsilon : float
        Exploration parameter of Exp3, in the range (0, 1].
    min_reward : float
            Minimum possible reward that can be collected per round.
    max_reward : float
            Maximum possible reward that can be collected per round.
    weights : np.ndarray
        Current Exp3 weights for each arm, initially all zeros.
    probs : np.ndarray
        Current probability distribution over arms, computed from weights and epsilon.
    
    Notes
    -----
    - The reward is scaled using:

          scaled_reward = (raw_reward - min_reward) / (max_reward - min_reward)

      This assumes raw rewards lie within the [`min_reward`, `max_reward`] interval.
    - Agents are stateless, so past actions and rewards are not used for state updates,
      but only for computing learning signals.
    - This implementation follows the framework from Lattimore (2020).
    
    See Also
    --------
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
        min_reward: float,
        max_reward: float,
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
        min_reward : float
            Minimum possible reward that can be collected per round.
        max_reward : float
            Maximum possible reward that can be collected per round.
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
            If `min_reward` is not strictly less than `max_reward`.

        Notes
        -----
        The `min_reward` and `max_reward` parameters define the expected reward range
        and are used to normalize the reward into the [0, 1] interval for compatibility
        with the EXP3 algorithm. The scaled reward is computed as:

            scaled_reward = (raw_reward - min_reward) / (max_reward - min_reward)

        If the raw reward falls outside of [min_reward, max_reward], the scaling
        may produce values outside the [0, 1] range.
        """
        super().__init__(ticksize, low, high, eq, prices, action_space, decimal_places, name, seed)

        if min_reward >= max_reward:
                raise ValueError(f'`min_reward` ({min_reward}) must be strictly less than `max_reward` ({max_reward})')
        
        self.epsilon = epsilon
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.weights = np.zeros(self.n_arms, dtype=np.float64)
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
        return np.exp(self.weights * self.epsilon) / np.sum(np.exp(self.weights * self.epsilon))


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
            Number of episodes.

        Returns
        -------
        epsilon : float
            The calculated optimal learning rate, bounded between 0 and 1.
        """
        return np.sqrt(np.log(n_arms) / (n_arms * n_episodes))


    def act(self, observation: Dict) -> Dict:
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
        
        scaled_reward = (reward - self.min_reward) / (self.max_reward - self.min_reward)
        scaled_reward = np.clip(scaled_reward, 0.0, 1.0)

        arm_idx, prob = self.last_action
        self.weights[arm_idx] += 1 - (1 - scaled_reward) / prob

        self.history.record_reward(reward)
        self.last_action = None
        return


    def reset(self) -> None:
        super().reset()
        self.weights = np.zeros(self.n_arms)
        return
