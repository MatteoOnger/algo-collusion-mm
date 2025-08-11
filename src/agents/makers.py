import numpy as np

from typing import Dict

from .agent import Agent



class MakerEXP3(Agent):
    """
    """

    def __init__(
        self,
        gamma: float,
        ticksize: float,
        low: float = 0.0,
        high: float = 1.0,
        name: str = 'agent',
        seed: int | None = None
    ):
        """
        """
        super().__init__(name)
        self.gamma = gamma
        self.ticksize = ticksize
        self.low = low
        self.high = high

        self._rng = np.random.default_rng(seed)

        self.prices =  np.arange(self.low, self.high + self.ticksize, self.ticksize)
        self.action_space = [(float(ask), float(bid)) for ask in self.prices for bid in self.prices if bid <= ask]
        self.n_arms = len(self.action_space)

        self.weights = np.ones(self.n_arms)
        self.cumulative_reward = 0.0
        self.last_action = None
        return


    @property
    def probs(self) -> np.ndarray:
        return (1 - self.gamma) * self.weights / np.sum(self.weights) + self.gamma / self.n_arms


    def act(self, observation: Dict) -> Dict:
        """
        Select an ask-bid strategy.

        Parameters
        ----------
        observation : dict
            The current observation for the agent (not used for this maker agent).

        Returns
        -------
        action : dict
            A dictionary containing:
            - 'ask_price': the ask price proposed by the agent.
            - 'bid_price': the bid price proposed by the agent.
        """
        arm_idx = self._rng.choice(self.n_arms, p=self.probs)

        strategy = self.action_space[arm_idx]
        self.last_action = (arm_idx, self.probs[arm_idx])

        return {
            "ask_price": strategy[0],
            "bid_price": strategy[1]
        }


    def update(self, reward: float) -> None:
        if self.last_action is None:
            return
        
        arm_idx, prob = self.last_action
        self.weights[arm_idx] = self.weights[arm_idx] * np.exp(self.gamma * (reward/prob) / self.n_arms)

        self.last_action = None
        return


    def reset(self) -> None:
        self.weights = np.ones(self.n_arms)
        self.cumulative_reward = 0.0
        self.last_action = None
        return
