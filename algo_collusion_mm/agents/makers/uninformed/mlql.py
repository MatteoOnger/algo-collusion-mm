""" Uninformed Memoryless Q-learning makers.
"""
import numpy as np

from typing import Dict

from ....enums import AgentType
from ..maker import Maker



class MakerMLQL(Maker):
    """
    Market maker for the GM environment based on a memoryless Q-learning approach.

    This agent maintains a Q-table over a discrete set of possible (ask, bid) strategies,
    but updates action values without conditioning on the history of past actions, so it is a memoryless approach.
    In this sense, it behaves similarly to the Epsilon-Greedy algorithm, where each action is evaluated independently.
    The exploration-exploitation trade-off is managed through an epsilon-greedy policy or using the optimistic initialization.
    The learning rate `alpha` and the discount factor `gamma` control how quickly the agent updates its Q-values
    based on the rewards received from the environment.
    
    Attributes
    ----------
    alpha : float
        Learning rate for Q-value updates, in the range [0, 1].
    gamma : float
        Discount factor for future rewards, in the range [0, 1].
    epsilon_scheduler : str
        Name of the scheduler used to update the expolarion rate epsilon.
        Must be one of the ones in `MakerMLQL.scheduler`.
    epsilon_init : float
        Initial exploration rate for the epsilon-greedy policy, in the range [0, 1].
    epsilon_decay_rate : float
        Decay rate applied to `epsilon` after each step.
    q_init : float or np.ndarray
        Initial values of the Q-table.
    epsilon : float
        Current exploration rate for the epsilon-greedy policy, in the range [0, 1].
    Q : np.ndarray
        Current Q-value for each arm.
    """

    type: AgentType = AgentType.MAKER_U

    scheduler = {
        'constant': lambda eps, dr, t: eps,
        'exponential': lambda eps, dr, t: eps * np.e**(-t * dr),
        'linear': lambda eps, dr, t: eps - (t * dr)
    }


    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon_scheduler: str = 'constant',
        epsilon_init: float = 0.0,
        epsilon_decay_rate: float = 0.0,
        q_init: float|np.ndarray = 0.0,
        action_values_attr: str = 'Q',
        ticksize: float = 0.2,
        low: float = 0.0,
        high: float = 1.0,
        eq: bool = True,
        prices: np.ndarray|None = None,
        action_space: np.ndarray|None = None,
        decimal_places: int = 2,
        name: str = 'mlql',
        seed: int|None = None
    ):
        """
        Parameters
        ----------
        alpha : float, default=0.1
            Learning rate for Q-value updates, in the range [0, 1].
        gamma : float, default=0.9
            Discount factor for future rewards, in the range [0, 1].
        epsilon_scheduler : str, default='constant'
            Name of the scheduler used to update the expolarion rate epsilon.
            Must be one of the ones in `MakerMLQL.scheduler`.
        epsilon_init : float, default=0.0
            Initial exploration rate for the epsilon-greedy policy, in the range [0, 1].
        epsilon_decay_rate : float, default=0.0
            Decay rate applied to `epsilon` after each step.
        q_init : float or np.ndarray, default=0.0
            Initial values of the Q-table.
            - If an integer, all entries in the Q-table are initialized to this value.
            - If an array-like object, it must have the same shape as the Q-table, and each entry
            will be used to initialize the corresponding Q-value.
        action_values_attr : str, default='Q'
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
            Number of decimal places to which rewards and prices are rounded.
        name : str, default='mlql'
            Name assigned to the agent.
        seed : int or None, default=None
            Seed for the internal random generator.
        """
        super().__init__(action_values_attr, ticksize, low, high, eq, prices, action_space, decimal_places, name, seed)

        self.alpha = alpha
        """Learning rate."""
        self.gamma = gamma
        """Discount factor."""
        self.epsilon_scheduler = epsilon_scheduler
        """Name of the scheduler."""
        self.epsilon_init = epsilon_init
        """Initial exploration rate."""
        self.epsilon_decay_rate = epsilon_decay_rate
        """Decay rate."""
        self.q_init = q_init
        """Initial values of the Q-table."""
        
        self._t = 0
        """Rounds done."""
        self.scheduler = MakerMLQL.scheduler[epsilon_scheduler]
        """Epsilon scheduler."""

        self.epsilon = self.scheduler(self.epsilon_init, self.epsilon_decay_rate, self._t)
        """Current exploration rate."""
        self.Q = np.zeros(self.n_arms) + self.q_init
        """Current Q-values."""
        return


    def act(self, observation: Dict) -> Dict[str, float]:
        if self._rng.random() < self.epsilon:
            arm_idx = self._rng.integers(self.n_arms)
        else:
            best_actions = np.where(self.Q == self.Q.max())[0]
            arm_idx = self._rng.choice(best_actions)
        
        strategy = self.action_space[arm_idx]
        self.last_action = arm_idx
        self._t += 1

        self.history.record_action(strategy)
        return {
            'ask_price': strategy[0],
            'bid_price': strategy[1]
        }


    def update(self, reward: float, info: Dict) -> None:
        if self.last_action is None:
            return
        
        self.epsilon = self.scheduler(self.epsilon_init, self.epsilon_decay_rate, self._t)
        self.Q[self.last_action] += self.alpha * (reward + self.gamma * np.max(self.Q) - self.Q[self.last_action])

        self.history.record_reward(reward)
        self.last_action = None
        return


    def reset(self) -> None:
        super().reset()
        self._t = 0
        self.Q = np.zeros(self.n_arms) + self.q_init
        self.epsilon = self.scheduler(self.epsilon_init, self.epsilon_decay_rate, self._t)
        return
