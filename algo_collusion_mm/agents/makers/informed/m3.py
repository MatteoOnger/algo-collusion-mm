""" Meta Market Maker (M3).
"""
import numpy as np

from typing import Callable, Dict

from ....enums import AgentType
from ..maker import Maker



class MakerM3(Maker):
    """
    Market maker for the GM environment based on M3, which in turn is based on 
    Exp3.

    This agent factorizes the market-making problem into two independent subproblems:
    selecting the ask price and selecting the bid price. Each subproblem is handled by a
    dedicated sub-agent implementing a lightweight Exp3 algorithm. The joint policy over
    (ask, bid) pairs is obtained by combining the probability distributions of the two
    sub-agents.

    Rewards are decomposed and routed to the appropriate sub-agent depending on whether
    a buy or sell operation was executed.

    The agent supports user-defined reward scaling to allow normalization into a range
    suitable for Exp3 updates, typically [0, 1].

    Attributes
    ----------
    epsilon : float
        Exploration parameter shared by the two internal Exp3 sub-agents.
    scale_rewards : callable[[float], float]
        Function used to normalize raw rewards.
    probs : np.ndarray
        Probability distribution over actions, computed from the current weights and `epsilon`.
    
    Notes
    -----
    - Each sub-agent operates on the discrete set of available prices.
    - Swapping ensures valid market making quotes even when independent sampling violates
      price ordering.

    References
    ----------
    - Cesa-Bianchi, N., Cesari, T., Colomboni, R., Foscari, L., & Pathak, V. (2024).
    Market Making without Regret. arXiv preprint arXiv:2411.13993.
    """

    type: AgentType = AgentType.MAKER_I

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
        Parameters
        ----------
        epsilon : float
            Exploration parameter of the internal Exp3 sub-agents.
        scale_rewards : callable[[float], float], default=lambda r: r
            Function to scale raw rewards into a normalized range suitable for Exp3.
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
        name : str, default='m3'
            Name assigned to the agent.
        seed : int or None, default=None
            Seed for the internal random generator.
        """
        super().__init__(ticksize, low, high, eq, prices, action_space, decimal_places, name, seed)
        self.epsilon = epsilon
        self.scale_rewards = scale_rewards

        self._isswapped = None
        self._subagents = [MakerM3.EXP3(self, len(self.prices), self.epsilon) for _ in range(2)]
        return
    
    @property
    def probs(self) -> np.ndarray:
        """Current probability distribution over actions."""
        indexes = self.price_to_index(self.action_space)
        ask_probs = self._subagents[0].probs[indexes[:, 0]]
        bid_probs = self._subagents[1].probs[indexes[:, 1]]
        return ask_probs * bid_probs


    @staticmethod
    def compute_epsilon(n_arms: int, n_rounds: int) -> float:
        """
        Compute the ideal learning rate `epsilon` for the M3 algorithm.
        
        The formula used is derived from the theoretical guarantees of the M3
        algorithm to minimize regret.

        Parameters
        ----------
        n_arms : int
            The number of arms (actions) in the bandit problem.
        n_rounds : int
            Number of rounds.

        Returns
        -------
        epsilon : float
            The calculated optimal learning rate.
        """
        return np.sqrt(np.log(n_arms) / (n_arms * n_rounds))


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
        Update the sub-agents based on the executed operation and realized reward.

        Rewards from the environment are decomposed according to the executed operation
        (`buy` or `sell`). Only the sub-agent corresponding to the active side receives
        a non-zero reward, while the other receives zero. Rewards are first scaled using
        the user-provided normalization function.

        Parameters
        ----------
        reward : float
            Raw reward received at the current step.
        info : dict
            Additional information from the environment containing:
            - 'op_done' (str): either 'buy' or 'sell'.
            - 'rewards' (np.ndarray): Rewards corresponding to each action
                in the action space of this agent.
        """
        if self.last_action is None:
            return

        rewards = info['rewards']
        operations = info['op_done']

        if self._isswapped:
            self.last_action = self.last_action[::-1]

        arm_idx = self.action_to_index(self.last_action)
        scaled_reward = self.scale_rewards(rewards[arm_idx])

        # operation: 0 = buy, 1 = sell
        if operations[arm_idx] ==  0:
            self._subagents[0].update(scaled_reward)
            self._subagents[1].update(0.0)
        elif operations[arm_idx] ==  1:
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
        Lightweight Exp3 learner used as a sub-agent inside MakerM3.

        This class implements a minimal adversarial bandit algorithm maintaining a
        probability distribution over price levels. It is used to independently select
        the ask price and bid price in the hierarchical M3 architecture.

        Attributes
        ----------
        agent : MakerM3
            Reference to the parent agent, providing the random generator.
        epsilon : float
            Exploration parameter controlling the balance between exploration and 
            exploitation.
        n_arms : int
            Number of available price levels.
        weights : np.ndarray
            Current Exp3 weights associated with each arm.
        last_action : tuple[int, float] or None
            Stores the last action index and its probability for updating.
        """

        def __init__(self, agent: 'MakerM3', n_arms :int, epsilon: float):
            """
            Parameters
            ----------
            agent : MakerM3
                Parent agent providing shared utilities such as the RNG.
            n_arms : int
                Number of possible price levels considered by the sub-agent.
            epsilon : float
                Exploration parameter of Exp3.
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
            Select an arm (price index) according to the Exp3 probability distribution.

            The chosen arm index and its associated probability are stored for the
            subsequent update step.

            Returns
            -------
            arm_idx : int
                Index of the sampled price.
            """
            arm_idx = self.agent._rng.choice(self.n_arms, p=self.probs.astype(np.float64))
            self.last_action = (arm_idx, self.probs[arm_idx])
            return arm_idx


        def update(self, reward: float) -> None:
            """
            Update the Exp3 weight of the last selected arm.

            The update uses an importance-weighted estimator based on the received reward
            and the probability with which the arm was selected. If no action was taken
            previously, the update is skipped.

            Parameters
            ----------
            reward : float
                Scaled reward in the range expected by Exp3 (typically [0, 1]).
            """

            if self.last_action is None:
                return

            arm_idx, prob = self.last_action
            self.weights[arm_idx] += 1 - (1 - reward) / prob

            self.last_action = None
            return


        def reset(self) -> None:
            """
            Reset the internal state of the agent.
            """
            self.weights = np.zeros(self.n_arms, dtype=np.float64)
            self.last_action = None
            return
