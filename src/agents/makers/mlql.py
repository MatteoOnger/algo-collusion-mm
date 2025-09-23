import numpy as np

from typing import Dict, List

from ..agent import Agent
from ...envs import GMEnv



class MakerMLQL(Agent):
    """
    Market maker for the GM environment based on a memoryless Q-learning approach.

    This agent maintains a Q-table over a discrete set of possible (ask, bid) strategies,
    but updates action values without conditioning on the history of past actions.
    In this sense, it behaves similarly to the Epsilon-Greedy algorithm, where each action
    is evaluated independently.
    At each step it selects an action using an epsilon-greedy (or an optimistic initial value) exploration strategy,
    with the exploration rate `epsilon` decaying linearly towards `epsilon_min`.
    The value of each action is updated through temporal-difference learning with learning
    rate `alpha` and discount factor `gamma`.

    Attributes
    ----------
    prices : np.ndarray
        Discrete set of possible prices from `low` to `high` with spacing `ticksize`.
    action_space : list of tuple of float
        All possible (ask_price, bid_price) pairs such that `bid_price <= ask_price`.
    n_arms : int
        Number of actions (arms) in the action space.
    epsilon : float
        Current exploration rate for the epsilon-greedy policy, in the range [0, 1].
    Q : np.ndarray
        Current Q-values for each arm, initialized to zeros.
    last_action : int or None
        Index of the last chosen action in the action space.
    t : int
        Current number of steps done.
    """

    _scheduler = {
        'constant': lambda eps, dr, t: eps,
        'exponential': lambda eps, dr, t: eps * np.e**(-t * dr),
        'linear': lambda eps, dr, t: eps - (t * dr)
    }


    def __init__(
        self,
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
        self._scheduler = MakerMLQL._scheduler[epsilon_scheduler]

        self.t = 0
        self.epsilon = self._scheduler(self.epsilon_init, self.decay_rate, self.t)

        self.prices =  np.round(np.arange(self.low, self.high + self.ticksize, self.ticksize), decimal_places)
        self._action_space = np.array([(ask, bid) for ask in self.prices for bid in self.prices if bid <= ask])

        self.Q = np.zeros(self.n_arms) + self.q_init
        self.last_action_idx = None
        return


    @property
    def action_space(self) -> List:
        return self._action_space


    def act(self, observation: Dict) -> Dict:
        """
        Select an ask-bid strategy using an exploration policy.

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
        if self._rng.random() < self.epsilon:
            arm_idx = self._rng.integers(self.n_arms)
        else:
            best_actions = np.where(self.Q == self.Q.max())[0]
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
        
        self.epsilon = self._scheduler(self.epsilon_init, self.decay_rate, self.t)
        self.Q[self.last_action_idx] += self.alpha * (reward + self.gamma * np.max(self.Q) - self.Q[self.last_action_idx])

        self.last_action_idx = None
        self.history.record_reward(reward)
        return


    def reset(self) -> None:
        super().reset()
        self._rng = np.random.default_rng(self.seed)
        self.Q = np.zeros(self.n_arms) + self.q_init
        self.last_action_idx = None
        self.t = 0
        return



class MakerInformedMLQL(MakerMLQL):
    """
    Market maker for the GM environment based on an informed memoryless Q-learning approach.

    This agent is similar to the `MakerMLQL` agent, but, after each action, it updates all 
    its Q-values using additional information from the environment provided in the `info` dictionary
    to estimate the reward it would have received if it had taken a different action.
    This allows the agent to learn more efficiently from each interaction with the environment.

    Attributes
    ----------
    prices : np.ndarray
        Discrete set of possible prices from `low` to `high` with spacing `ticksize`.
    action_space : list of tuple of float
        All possible (ask_price, bid_price) pairs such that `bid_price <= ask_price`.
    n_arms : int
        Number of actions (arms) in the action space.
    epsilon : float
        Current exploration rate for the epsilon-greedy policy, in the range [0, 1].
    Q : np.ndarray
        Current Q-values for each arm, initialized to zeros.
    last_action : int or None
        Index of the last chosen action in the action space.
    t : int
        Current number of steps done.
    """

    def update(self, reward: float, info: Dict) -> None:
        if self.last_action_idx is None:
            return

        true_value = info['true_value']
        min_ask_price = info['min_ask_price']
        max_bid_price = info['max_bid_price']
        trader_action = info['trader_action']
        maker_reward = info['maker_reward']
        trader_reward = info['trader_reward']
        
        n_selected_makers = abs(trader_reward) // abs(maker_reward) if maker_reward != 0 else 0

        for idx, action in enumerate(self.action_space):
            ask_price = action[0]
            bid_price = action[1]

            ask_reward = ask_price - true_value
            bid_reward = true_value - bid_price

            if idx == self.last_action_idx:
                rwd = reward
            elif trader_action == GMEnv.TraderAction.PASS:
                rwd = 0
            elif ask_price > min_ask_price and bid_price < max_bid_price:
                rwd = 0
            elif ask_price < min_ask_price or bid_price > max_bid_price:
                rwd = min(ask_reward, bid_reward)
            else:
                if trader_action == GMEnv.TraderAction.BUY and ask_price == min_ask_price:
                    rwd = ask_reward / (n_selected_makers + 1 if reward == 0 else n_selected_makers)
                elif trader_action == GMEnv.TraderAction.SELL and bid_price == max_bid_price:
                    rwd = bid_reward / (n_selected_makers + 1 if reward == 0 else n_selected_makers)
                else:
                    rwd = 0
            
            rwd = round(rwd, self.decimal_places)
            self.Q[idx] += self.alpha * (rwd + self.gamma * np.max(self.Q) - self.Q[idx])

        self.epsilon = self._scheduler(self.epsilon_init, self.decay_rate, self.t)
        self.history.record_reward(reward)
        self.last_action_idx = None
        return
