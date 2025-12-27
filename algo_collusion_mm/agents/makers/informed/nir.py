""" No-Internal Regret Maker.
"""
import numpy as np

from typing import Dict, Callable

from ....enums import AgentType
from ..maker import Maker



class MakerNIR(Maker):
    """
    Market-making agent for the GM environment based on a
    No-Internal-Regret (NIR) learning algorithm, implemented via
    regret matching with full-information feedback.

    The agent maintains a matrix of cumulative conditional regrets.
    For each action i previously taken, and for each alternative action j,
    the entry regrets[i, j] represents the cumulative regret of not having
    played action j instead of i.

    At each decision step, the agent selects an action according to a
    probability distribution derived from the sum of positive conditional
    regrets for each action, following the regret-matching principle.
    Actions with higher accumulated positive regret are selected more
    frequently, while actions with non-positive regret receive little
    or no probability mass.

    The `epsilon` parameter scales the regret values before exponentiation,
    controlling the sharpness of the resulting probability distribution.

    After each step, the agent updates its regret matrix using
    full-information feedback: it compares the realized reward of the
    chosen action with the rewards that would have been obtained by
    playing every other action. Raw rewards are optionally normalized
    using a user-provided `scale_rewards` function to ensure numerical
    stability across environments.

    Attributes
    ----------
    epsilon : float
        Scaling parameter applied to cumulative positive regrets when
        computing the action probabilities.
    scale_rewards : callable[[np.ndarray], np.ndarray]
        Function that scales raw rewards into a suitable numerical range
        for regret updates.
    regrets : np.ndarray
        Cumulative conditional regret matrix of shape (n_arms, n_arms),
        where entry (i, j) represents the regret of not having played
        action j when action i was chosen.
    probs : np.ndarray
        Current probability distribution over actions, computed from
        cumulative positive regrets.
    """

    type: AgentType = AgentType.MAKER_I


    def __init__(
        self,
        epsilon: float,
        scale_rewards: Callable[[float], float] = lambda r: r,
        action_values_attr: str = 'probs',
        ticksize: float = 0.2,
        low: float = 0.0,
        high: float = 1.0,
        eq: bool = True,
        prices: np.ndarray|None = None,
        action_space: np.ndarray|None = None,
        decimal_places: int = 2,
        name: str = 'crm',
        seed: int|None = None
    ):
        """
        Parameters
        ----------
        epsilon : float
            Exploration parameter.
        scale_rewards : callable[[np.ndarray], np.ndarray], default=lambda r: r
            Function to scale raw rewards into a normalized range suitable
            for regret updates.
        action_values_attr : str, default='probs'
            Name of the property that provides the action value representation.
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
        name : str, default='crm'
            Name assigned to the agent.
        seed : int or None, default=None
            Seed for the internal random generator.
        """
        super().__init__(action_values_attr, ticksize, low, high, eq, prices, action_space, decimal_places, name, seed)

        self.epsilon = epsilon
        """Exploration parameter."""
        self.scale_rewards = scale_rewards
        """Function to scale raw rewards."""
        self.regrets = np.ones((self.n_arms, self.n_arms), dtype=np.float64)
        """Cumulative conditional regret matrix."""
        return
    

    @property
    def probs(self) -> np.ndarray:
        """Current probability distribution over actions."""
        x = self.epsilon * np.maximum(self.regrets, 0).sum(axis=0)
        x_stable = x - np.max(x)
        return np.exp(x_stable) / np.sum(np.exp(x_stable))


    def act(self, observation: Dict) -> Dict[str, float]:
        arm_idx = self._rng.choice(self.n_arms, p=self.probs.astype(np.float64))

        strategy = self.action_space[arm_idx]
        self.last_action = arm_idx

        self.history.record_action(strategy)
        return {
            'ask_price': strategy[0],
            'bid_price': strategy[1]
        }


    def update(self, reward: float, info: Dict) -> None:
        """
        Update the agent's internal state based on the reward received
        and additional information from the environment.

        Parameters
        ----------
        reward : float
            The reward assigned to the agent for the most recent action.
        info : dict of str
            A dictionary containing environment feedback, with keys:
                - 'rewards' (np.ndarray): Rewards corresponding to each action
                  in the action space of this agent.
        """
        if self.last_action is None:
            return

        rewards = info['rewards']

        scaled_reward = self.scale_rewards(reward)
        scaled_rewards = self.scale_rewards(rewards)

        self.regrets[self.last_action] += (scaled_rewards - scaled_reward)

        self.history.record_reward(reward)
        self.last_action = None
        return


    def reset(self) -> None:
        super().reset()
        self.weights = np.zeros(self.n_arms, dtype=np.float64)
        return
