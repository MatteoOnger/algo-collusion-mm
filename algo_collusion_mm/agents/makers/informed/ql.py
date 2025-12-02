""" Q-learning maker.
"""
import numpy as np

from typing import Dict

from ....enums import AgentType
from ..maker import Maker



class MakerIQL(Maker):
    """
    Market maker for the GM environment based on Q-learning.

    The agent's state is defined by the actions (ask and bid prices) taken by
    all market makers in the previous round. Each state is identified by a
    unique ID. This agent is considered "informed" because its decision-making
    process is directly influenced by the past actions of other market participants.

    The possible actions for this agent are pairs of (ask price, bid price),
    representing the prices at which it is willing to sell and buy an asset.
    The Q-learning algorithm is used to learn an optimal policy by updating a
    Q-table based on rewards received after each action, helping the agent
    maximize its long-term gains in the market.

    The agent's exploration policy is either epsilon-greedy, which involves
    a probability 'epsilon' of taking a random action to explore the state-action
    space, or optimistic initialization, where all Q-values are initialized
    to high values, encouraging the agent to explore all actions before settling
    on a policy.

    Attributes
    ----------
    n_agents : int
        Number of maker agents in the environment.
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
        Current Q-values for each arm.
    curr_state_idx : int
        Index of the current state of the agent.
    """

    type: AgentType = AgentType.MAKER_I

    scheduler = {
        'constant': lambda eps, dr, t: eps,
        'exponential': lambda eps, dr, t: eps * np.e**(-t * dr),
        'linear': lambda eps, dr, t: eps - (t * dr)
    }


    def __init__(
        self,
        n_agents: int,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon_scheduler: str = 'constant',
        epsilon_init: float = 0.0,
        epsilon_decay_rate: float = 0.0,
        q_init: float|np.ndarray = 0.0,
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
        n_agents : int
            Number of maker agents in the environment.
        alpha : float, default=0.1
            Learning rate for Q-value updates, in the range [0, 1].
        gamma : float, default=0.9
            Discount factor for future rewards, in the range [0, 1].
        epsilon_scheduler : str, default='constant'
            Name of the scheduler used to update the expolarion rate epsilon.
            Must be one of the ones in `MakerMLQL._scheduler`.
        epsilon_init : float, default=0.0
            Initial exploration rate for the epsilon-greedy policy, in the range [0, 1].
        epsilon_decay_rate : float, default=0.0
            Decay rate applied to `epsilon` after each step.
        q_init : float or np.ndarray, default=0.0
            Initial values of the Q-table.
            - If an integer, all entries in the Q-table are initialized to this value.
            - If an array-like object, it must have the same shape as the Q-table, and each entry
            will be used to initialize the corresponding Q-value.
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
            Number of decimal places to which rewards are rounded.
        name : str, default='maker'
            Name assigned to the agent.
        seed : int or None, default=None
            Seed for the internal random generator.
        """
        super().__init__(ticksize, low, high, eq, prices, action_space, decimal_places, name, seed)

        self.n_agents = n_agents
        """Number of agents."""
        self.alpha = alpha
        """Learning rate."""
        self.gamma = gamma
        """Discount factor."""
        self.epsilon_scheduler = epsilon_scheduler
        """Name of the scheduler."""
        self.epsilon_init = epsilon_init
        """ Initial exploration rate."""
        self.epsilon_decay_rate = epsilon_decay_rate
        """ Decay rate."""
        self.q_init = q_init
        """Initial values of the Q-table."""

        self._t = 0
        """Rounds done."""
        self.scheduler = MakerIQL.scheduler[epsilon_scheduler]
        """Epsilon scheduler."""

        self.epsilon = self.scheduler(self.epsilon_init, self.epsilon_decay_rate, self._t)
        """Current exploration rate."""
        self.Q = np.zeros(((self.n_arms ** self.n_agents) + 1, self.n_arms)) + self.q_init
        """Current Q-values."""
        self.curr_state_idx = 0
        """Index of the current state."""
        return


    def act(self, observation: Dict) -> Dict[str, float]:
        if self._rng.random() < self.epsilon:
            arm_idx = self._rng.integers(self.n_arms)
        else:
            best_actions = np.where(self.Q[self.curr_state_idx] == self.Q[self.curr_state_idx].max())[0]
            arm_idx = self._rng.choice(best_actions)
        
        strategy = self.action_space[arm_idx]
        self.last_action = arm_idx
        self._t += 1

        self.history.record_state(self.curr_state_idx)
        self.history.record_action(strategy)
        return {
            'ask_price': strategy[0],
            'bid_price': strategy[1]
        }


    def update(self, reward: float, info: Dict[str, np.ndarray]) -> None:
        """
        Update the agent's internal state based on the reward received
        and additional information from the enivironment.

        Parameters
        ----------
        reward : float
            The reward assigned to the agent for the most recent action.
        info : dict of str
            A dictionary containing:
            - 'actions' (np.ndarray): array of actions played by all makers in the round just ended.
                It represents the new state.
        """
        if self.last_action is None:
            return
        
        next_state_idx = self._action_to_state_idx(info['actions'])
        self.Q[self.curr_state_idx, self.last_action] += self.alpha * (
            reward + self.gamma * np.max(self.Q[next_state_idx]) - self.Q[self.curr_state_idx, self.last_action]
        )

        self.epsilon = self.scheduler(self.epsilon_init, self.epsilon_decay_rate, self._t)
        self.curr_state_idx = next_state_idx
        self.history.record_reward(reward)
        self.last_action = None
        return


    def reset(self) -> None:
        super().reset()
        self._t = 0
        self.curr_state_idx = 0
        self.Q = np.zeros(self.n_arms) + self.q_init
        self.epsilon = self.scheduler(self.epsilon_init, self.epsilon_decay_rate, self._t)
        return
    

    def _action_to_state_idx(self, actions: np.ndarray) -> int:
        """
        Convert a list of actions into a unique state index.

        Parameters
        ----------
        actions : np.ndarray
            Array of shape (n_agents, 2) representing the (ask, bid) prices chosen by the agents.

        Returns
        -------
        state_idx : int
            Unique index corresponding to the joint action of all agents.
        """
        actions_idx = self.action_to_index(actions)
        return np.sum(actions_idx * (self.n_arms ** np.arange(len(actions_idx) -1, -1, -1))) + 1
