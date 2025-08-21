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


    @abstractmethod
    def reset(self) -> None:
        """
        Reset the internal state of the agent, if any.
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
            self._history = list()
            return


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
            unique_pairs, counts = np.unique(self.get_history(key), axis=0, return_counts=True)
            return unique_pairs, counts


        def get_history(self, key: Union[int, slice, Tuple[Union[int, slice], ...]] = slice(None)) -> np.ndarray:
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
            return np.array(self._history)[key]


        def record(self, strategy: Any) -> None:
            """
            Record a new action in the history.

            Parameters
            ----------
            action : Any
                The strategy chosen by the agent, must be part of `action_space`.
            """
            self._history.append(strategy)
            return
        

        def __len__(self) -> int:
            return len(self._history)
