import numpy as np

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union



class Agent(ABC):
    """
    Interface for agents in the GM environment.

    Attributes
    ----------
    action_space : list
        All possible available actions.
    n_arms : int
        Number of actions (arms) in the action space.
    history : Agent.History
        Internal tracker of actions taken.
    """


    def __init__(self, name: str = 'agent'):
        """
        Parameters
        ----------
        name : str, default='agent'
            Unique identifier for the agent.
        """
        super().__init__()
        self.name = name
        self.history = Agent.History()
        return


    @property
    def n_arms(self) -> int:
        return len(self.action_space)


    def reset(self) -> None:
        """
        Reset the internal state of the agent.
        """
        self.history = Agent.History()
        return


    @property
    @abstractmethod
    def action_space(self) -> List:
        pass


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
    def update(self, reward: float) -> None:
        """
        Update the internal state of the agent based on the received reward.

        Parameters
        ----------
        reward : float
            The reward assigned to the agent for the most recent action.
        """
        pass


    class History():
        """
        Class for tracking the sequence of actions taken by the agent.
        """
        
        def __init__(self):
            """
            Initialize a new instance of History to track the actions done.
            """
            self._actions = list()
            self._rewards = list()
            return


        def get_actions(self, key: Union[int, slice, Tuple[Union[int, slice], ...]] = slice(None)) -> np.ndarray:
            """
            Retrieve a subset of the action history.

            Parameters
            ----------
            key : int, slice, or tuple of ints/slices, default=[:]
                Index, slice, or tuple specifying episodes (rows) and actions (columns).

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
                Index, slice, or tuple specifying episodes (rows) and actions (columns).

            Returns
            -------
            : np.ndarray
                Subset of the reward history specified by the key.
            """
            return np.array(self._rewards)[key]


        def get_freqs(self, key: Union[int, slice, Tuple[Union[int, slice], ...]] = slice(None)) -> np.ndarray:
            """
            Count the frequency of each action.

            Parameters
            ----------
            key : int, slice, or tuple of ints/slices, default=[:]
                Index, slice, or tuple specifying episodes (rows) and actions (columns).

            Returns
            -------
            unique_pairs : np.ndarray
                Array containing unique actions.
            counts : np.ndarray
                Array containing the count of unique action.
            """
            unique_pairs, counts = np.unique(self.get_actions(key), axis=0, return_counts=True)
            return unique_pairs, counts


        def record_action(self, action: Any) -> None:
            """
            Record a new action in the history.

            Parameters
            ----------
            action : Any
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


        def __len__(self) -> int:
            return len(self._actions)
