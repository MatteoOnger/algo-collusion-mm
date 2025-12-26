""" Conditional Regret Matching Maker.
"""
import numpy as np

from typing import Dict, Callable

from ....enums import AgentType
from ..maker import Maker



class MakerCRM(Maker):
    """
    Market-making agent for the GM environment based on the
    Conditional Regret Matching (CRM) algorithm.

    This agent maintains a regret matrix that tracks, for each action taken,
    the cumulative regret of not having played any other action instead.
    At each step, the agent selects an action according to a probability
    distribution derived from positive conditional regrets, following
    the regret-matching principle.

    The CRM algorithm ensures that actions with higher positive regret
    are played more frequently over time, while actions with non-positive
    regret are suppressed. Exploration is controlled through the
    `epsilon` parameter, which scales the regret normalization.

    At each step, the agent receives a reward and updates its regret matrix
    using full-information feedback from the environment. Raw rewards are
    first normalized using a user-provided `scale_rewards` function to ensure
    stable regret updates across environments.

    Attributes
    ----------
    epsilon : float
        Exploration and normalization parameter used in regret matching.
    scale_rewards : callable[[np.ndarray], np.ndarray]
        Function that scales raw rewards into a suitable range.
    regrets : np.ndarray
        Matrix of cumulative conditional regrets with shape
        (n_arms, n_arms), where entry (i, j) represents the regret of
        not having played action j when action i was chosen.
    probs : np.ndarray
        Current probability distribution over actions, computed from
        positive conditional regrets.

    References
    ----------
    - Albrecht, S. V., Christianos, F., & Schäfer, L. (2024).
    Multi-agent reinforcement learning: Foundations and modern approaches. MIT Press.
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
            Exploration and normalization parameter for regret matching.
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

        self._t = 0
        """Rounds done."""

        self.epsilon = epsilon
        """Exploration parameter."""
        self.scale_rewards = scale_rewards
        """Function to scale raw rewards."""
        self.regrets = np.zeros((self.n_arms, self.n_arms), dtype=np.float64)
        """Cumulative conditional regret matrix."""
        self.probs = np.ones(self.n_arms, dtype=np.float64) / len(self.action_space)
        """Current probability distribution over actions."""
        return


    def act(self, observation: Dict) -> Dict[str, float]:
        arm_idx = self._rng.choice(self.n_arms, p=self.probs.astype(np.float64))

        strategy = self.action_space[arm_idx]
        self.last_action = arm_idx
        self._t += 1  

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

        self.probs = np.maximum(self.regrets[self.last_action], 0) / (self._t * self.epsilon)
        self.probs[self.last_action] = 1 - np.delete(self.probs, self.last_action).sum()

        self.history.record_reward(reward)
        self.last_action = None
        return


    def reset(self) -> None:
        super().reset()
        self.weights = np.zeros(self.n_arms, dtype=np.float64)
        return
