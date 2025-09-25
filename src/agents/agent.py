import numpy as np

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Union



class Agent(ABC):
    """
    Interface for agents in the GM environment.

    Attributes
    ----------
    name : str
        Unique identifier for the agent.
    action_space : np.ndarray
        All possible available actions.
    n_arms : int
        Number of actions (arms) in the action space.
    history : Agent.History
        Internal tracker of actions taken.
    """


    def __init__(self, name: str = 'agent', seed: int|None = None):
        """
        Parameters
        ----------
        name : str, default='agent'
            Unique identifier for the agent.
        seed : int or None, default=None
            Seed for the internal random generator.
        """
        super().__init__()
        self._seed = seed
        self._rng = np.random.default_rng(self._seed)

        self.name = name
        self.history = Agent.History()
        return


    @property
    def n_arms(self) -> int:
        return len(self.action_space)
    

    @property
    @abstractmethod
    def action_space(self) -> np.ndarray:
        pass


    def reset(self) -> None:
        """
        Reset the internal state of the agent.
        """
        self._rng = np.random.default_rng(self._seed)
        self.history = Agent.History()
        return


    @abstractmethod
    def act(self, observation: Dict) -> Dict:
        """
        Return an action given the current observation.

        Parameters
        ----------
        observation : dict
            The current observation for the agent.

        Returns
        -------
        action : dict
            The action chosen by the agent.
        """
        pass

    
    @abstractmethod
    def update(self, reward: float, info: Dict) -> None:
        """
        Updates the agent's internal state based on the reward received and
        any additional information (if available).

        Parameters
        ----------
        reward : float
            The reward assigned to the agent for the most recent action.
        info : dict
            Empty dictionary. Not used for this agent.
        """
        pass


    class History():
        """
        Class for tracking the sequence of actions, rewards and states of the agent.
        """

        def __init__(self):
            """
            Initialize a new instance of History to track the actions done.
            """
            self._actions = list()
            self._rewards = list()
            self._states = list()
            return


        def compute_freqs(self, key: Union[int, slice, Tuple[Union[int, slice], ...]] = slice(None)) -> np.ndarray:
            """
            Count the frequency of each action.

            Parameters
            ----------
            key : int, slice, or tuple of ints/slices, default=[:]
                Index, slice, or tuple specifying episodes.

            Returns
            -------
            unique_actions : np.ndarray
                Array containing unique actions.
            counts : np.ndarray
                Array containing the count of unique action.
            """
            unique_actions, counts = np.unique(self.get_actions(key), axis=0, return_counts=True)
            return unique_actions, counts


        def get_actions(self, key: Union[int, slice, Tuple[Union[int, slice], ...]] = slice(None)) -> np.ndarray:
            """
            Retrieve a subset of the action history.

            Parameters
            ----------
            key : int, slice, or tuple of ints/slices, default=[:]
                Index, slice, or tuple specifying episodes.

            Returns
            -------
            : np.ndarray
                Subset of the action history specified by the key.
            """
            return np.array(self._actions)[key]


        def get_rewards(self, key: Union[int, slice, Tuple[Union[int, slice], ...]] = slice(None)) -> np.ndarray:
            """
            Retrieve a subset of the reward history.

            Parameters
            ----------
            key : int, slice, or tuple of ints/slices, default=[:]
                Index, slice, or tuple specifying episodes.

            Returns
            -------
            : np.ndarray
                Subset of the reward history specified by the key.
            """
            return np.array(self._rewards)[key]


        def get_statess(self, key: Union[int, slice, Tuple[Union[int, slice], ...]] = slice(None)) -> np.ndarray:
            """
            Retrieve a subset of the state history.

            Parameters
            ----------
            key : int, slice, or tuple of ints/slices, default=[:]
                Index, slice, or tuple specifying episodes.

            Returns
            -------
            : np.ndarray
                Subset of the state history specified by the key.
            """
            return np.array(self._states)[key]


        def record_action(self, action: Any) -> None:
            """
            Record a new action in the history.

            Parameters
            ----------
            action : any
                The strategy chosen by the agent, must be part of `action_space`.
            """
            self._actions.append(action)
            return


        def record_reward(self, reward: float) -> None:
            """
            Record a new reward in the history.

            Parameters
            ----------
            reward : float
                The reward obtained by the agent.
            """
            self._rewards.append(reward)
            return


        def record_state(self, state: Any) -> None:
            """
            Record a new state in the history.

            Parameters
            ----------
            state : any
                The state of the agent.
            """
            self._states.append(state)
            return


        def __len__(self) -> int:
            return len(self._actions)
