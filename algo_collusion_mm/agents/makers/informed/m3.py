""" Meta Market Maker.
"""
import numpy as np

from typing import Callable, Dict

from ..maker import Maker



class MakerM3(Maker):
    """
    """

    is_informed = True

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
        name: str = 'm3',
        seed: int|None = None
    ):
        """
        """
        super().__init__(ticksize, low, high, eq, prices, action_space, decimal_places, name, seed)
        self.epsilon = epsilon
        self.scale_rewards = scale_rewards

        #TODO: warning se action_space != price X price

        self._isswapped = None
        self._subagents = 2 * [MakerM3.EXP3(self, len(self.prices), self.epsilon),]
        return
    

    def act(self, observation: Dict) -> Dict[str, float]:
        strategy = self.prices[[self._subagents[0].act(), self._subagents[1].act()]]
        if strategy[0] < strategy[1]:
            strategy = strategy[::-1]
            self._isswapped = True

        self.last_action = strategy
        self.history.record_action(strategy)
        return {
            'ask_price': strategy[0],
            'bid_price': strategy[1]
        }


    def update(self, reward: float, info: Dict) -> None:
        """
        """
        if self.last_action is None:
            return

        operation = info['op_done'] 
        scaled_reward = self.scale_rewards(reward)

        if (self._isswapped and operation == 'sell') or (not self._isswapped and operation ==  'buy'):
            self._subagents[0].update(scaled_reward)
            self._subagents[1].update(0.0)
        elif (self._isswapped and operation ==  'buy') or (not self._isswapped and operation ==  'sell'):
            self._subagents[0].update(0.0)
            self._subagents[1].update(scaled_reward)

        self.history.record_reward(reward)
        self.last_action = None
        return


    def reset(self) -> None:
        super().reset()
        self.weights = np.zeros(self.n_arms, dtype=np.float64)
        for subagent in self._subagents:
            subagent.reset()
        return


    class EXP3():
        """
        """

        def __init__(self, agent: 'MakerM3', n_arms :int, epsilon: float):
            """
            """
            self.agent = agent
            self.epsilon = epsilon
            self.n_arms = n_arms

            self.last_action = None
            self.weights = np.zeros(self.n_arms, dtype=np.float64)
            return


        @property
        def probs(self) -> np.ndarray:
            """Current probability distribution over actions."""
            x = self.epsilon * self.weights
            x_stable = x - np.max(x)
            return np.exp(x_stable) / np.sum(np.exp(x_stable))
        

        def act(self) -> int:
            """
            """
            arm_idx = self.agent._rng.choice(self.n_arms, p=self.probs.astype(np.float64))
            self.last_action = (arm_idx, self.probs[arm_idx])
            return arm_idx


        def update(self, reward: float) -> None:
            """
            """
            if self.last_action is None:
                return

            arm_idx, prob = self.last_action
            self.weights[arm_idx] += 1 - (1 - reward) / prob

            self.last_action = None
            return


        def reset(self) -> None:
            """
            """
            self.weights = np.zeros(self.n_arms, dtype=np.float64)
            self.last_action = None
            return
