import numpy as np

from typing import Dict, List

from ..agent import Agent



class MakerQL(Agent):

    _scheduler = {
        'constant': lambda eps, dr, t: eps,
        'exponential': lambda eps, dr, t: eps * np.e**(-t * dr),
        'linear': lambda eps, dr, t: eps - (t * dr)
    }


    def __init__(
        self,
        n_agents: int,
        alpha: float,
        gamma: float,
        epsilon_init: float = 0.0,
        decay_rate: float = 0.0,
        epsilon_scheduler: str = 'constant',
        q_init: float|np.ndarray = 0.0,
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
        n_agents : int
            Number of agents in the environment.
        alpha : float
            Learning rate for Q-value updates, in the range (0, 1].
        gamma : float
            Discount factor for future rewards, in the range [0, 1].
        epsilon_init : float, default=0.0
            Initial exploration rate for the epsilon-greedy policy, in the range [0, 1].
        decay_rate : float, default=0.0
            Decay rate applied to `epsilon` after each step.
        epsilon_scheduler : str, default='constant'
            Name of the scheduler used to update the expolarion rate epsilon.
        q_init : float or np.ndarray, default=0.0
            Initial values of the Q-table.
            - If an integer, all entries in the Q-table are initialized to this value.
            - If an array-like object, it must have the same shape as the Q-table, and each entry
            will be used to initialize the corresponding Q-value.
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

        self.n_agents = n_agents
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_init = epsilon_init
        self.decay_rate = decay_rate
        self.epsilon_scheduler = epsilon_scheduler
        self.q_init = q_init
        self.ticksize = ticksize
        self.low = low
        self.high = high
        self.decimal_places = decimal_places
        self.seed = seed

        self._rng = np.random.default_rng(seed)
        self._scheduler = MakerQL._scheduler[epsilon_scheduler]

        self.t = 0
        self.epsilon = self._scheduler(self.epsilon_init, self.decay_rate, self.t)

        self.prices =  np.round(np.arange(self.low, self.high + self.ticksize, self.ticksize), decimal_places)
        self._action_space = np.array([(ask, bid) for ask in self.prices for bid in self.prices if bid <= ask])

        self.Q = np.zeros(((self.n_arms ** self.n_agents) + 1, self.n_arms)) + self.q_init
        self.last_action_idx = None
        self.state_idx = 0
        return


    @property
    def action_space(self) -> List:
        return self._action_space


    def act(self, observation: Dict) -> Dict:
        if self._rng.random() < self.epsilon:
            arm_idx = self._rng.integers(self.n_arms)
        else:
            best_actions = np.where(self.Q[self.state_idx] == self.Q[self.state_idx].max())[0]
            arm_idx = self._rng.choice(best_actions)
        
        strategy = self.action_space[arm_idx]
        self.last_action_idx = arm_idx
        self.t += 1

        self.history.record_action(strategy)
        return {
            'ask_price': strategy[0],
            'bid_price': strategy[1]
        }


    def update(self, reward: float, info: Dict) -> None:
        if self.last_action_idx is None:
            return
        
        next_state_idx = self._actions_to_space_state_idx(info['actions'])
        self.Q[self.state_idx, self.last_action_idx] += self.alpha * (
            reward + self.gamma * np.max(self.Q[next_state_idx]) - self.Q[self.state_idx, self.last_action_idx]
        )

        self.last_action_idx = None
        self.state_idx = next_state_idx
        self.history.record_reward(reward)
        self.epsilon = self._scheduler(self.epsilon_init, self.decay_rate, self.t)
        return


    def reset(self) -> None:
        super().reset()
        self._rng = np.random.default_rng(self.seed)
        self.epsilon = self._scheduler(self.epsilon_init, self.decay_rate, 0)
        self.Q = np.zeros(self.n_arms) + self.q_init
        self.last_action_idx = None
        self.state_idx = 0
        self.t = 0
        return
    

    def _actions_to_space_state_idx(self, actions: np.ndarray) -> int:
        actions = (np.round(actions / self.ticksize, 0)).astype(int)
        actions_idx =  ((actions[:, 0] * (actions[:, 0] + 1) / 2) + actions[:, 1]).astype(int)
        return np.sum(actions_idx * (self.n_arms ** np.arange(len(actions_idx) -1, -1, -1))) + 1
