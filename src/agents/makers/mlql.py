import numpy as np

from typing import Dict

from .maker import Maker
from ...envs import GMEnv



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
    alpha : float, default=0.1
        Learning rate for Q-value updates, in the range (0, 1].
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
    epsilon : float
        Current exploration rate for the epsilon-greedy policy, in the range [0, 1].
    Q : np.ndarray
        Current Q-values for each arm.
    """

    _scheduler = {
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
        ticksize: float = 0.2,
        low: float = 0.0,
        high: float = 1.0,
        eq: bool = True,
        prices: np.ndarray|None = None,
        action_space: np.ndarray|None = None,
        decimal_places: int = 2,
        name: str = 'maker_mlql',
        seed: int|None = None
    ):
        """
        Parameters
        ----------
        alpha : float, default=0.1
            Learning rate for Q-value updates, in the range (0, 1].
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

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_scheduler = epsilon_scheduler
        self.epsilon_init = epsilon_init
        self.epsilon_decay_rate = epsilon_decay_rate
        self.q_init = q_init

        self._t = 0
        self._scheduler = MakerMLQL._scheduler[epsilon_scheduler]

        self.epsilon = self._scheduler(self.epsilon_init, self.epsilon_decay_rate, self._t)
        self.Q = np.zeros(self.n_arms) + self.q_init
        return


    def act(self, observation: Dict) -> Dict:
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
        
        self.epsilon = self._scheduler(self.epsilon_init, self.epsilon_decay_rate, self._t)
        self.Q[self.last_action] += self.alpha * (reward + self.gamma * np.max(self.Q) - self.Q[self.last_action])

        self.history.record_reward(reward)
        self.last_action = None
        return


    def reset(self) -> None:
        super().reset()
        self._t = 0
        self.Q = np.zeros(self.n_arms) + self.q_init
        self.epsilon = self._scheduler(self.epsilon_init, self.epsilon_decay_rate, self._t)
        return



class MakerInformedMLQL(MakerMLQL):
    """
    Market maker for the GM environment based on an informed memoryless Q-learning approach.

    This agent is similar to the `MakerMLQL` agent, but, after each action, it updates all 
    its Q-values using additional information from the environment provided in the `info` dictionary
    to estimate the reward it would have received if it had taken a different action.
    """

    def update(self, reward: float, info: Dict) -> None:
        """
        Updates the agent's internal state based on the reward received
        and additional information from the enivironment.

        Parameters
        ----------
        reward : float
            The reward assigned to the agent for the most recent action.
        info : dict
            A dictionary containing environment feedback, with keys:
                - 'true_value': The true value of the traded asset.
                - 'min_ask_price': The minimum ask price among all makers.
                - 'max_bid_price': The maximum bid price among all makers.
                - 'trader_action': The action taken by the trader.
                - 'maker_reward': The reward assigned to makers.
                - 'trader_reward': The reward assigned to the trader.
        """
        if self.last_action is None:
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

            if idx == self.last_action:
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

        self.epsilon = self._scheduler(self.epsilon_init, self.epsilon_decay_rate, self._t)
        self.history.record_reward(reward)
        self.last_action = None
        return
