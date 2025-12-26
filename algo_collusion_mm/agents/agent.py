""" Abstract agent.
"""
import numpy as np

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Union

from ..enums import AgentType



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
        Internal tracker of actions taken, rewards and other infos.
    type : AgentType
        Role of the agent, used to distinguish
        between different agent implementations or behaviors
    """

    type: AgentType = AgentType.ABSTRACT
    """Agent's type."""


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
        """Seed of the PRNG."""
        self._rng = np.random.default_rng(self._seed)
        """PRNG."""

        if self._seed is None:
            self._seed = self._rng.bit_generator.seed_seq.entropy

        self.name = name
        """Name of the agent."""
        self.history = Agent.History(self)
        """History of the agent."""
        return


    @property
    def n_arms(self) -> int:
        """Number of actions."""
        return len(self.action_space)
    

    @property
    @abstractmethod
    def action_space(self) -> np.ndarray:
        """Action space."""
        pass


    def action_to_index(self, actions :np.ndarray) -> np.ndarray:
        """
        Convert an array of actions to their corresponding indices based on the action space.

        This method takes a NumPy array of actions and maps each action to an index based on
        its position in the action space of the agent.

        Parameters
        -----------
        actions : np.ndarray
            A NumPy array representing the actions, last axis corresponds to a single action.

        Returns
        --------
        : np.ndarray
            The function returns a NumPy array of indices corresponding to each action.

        Raises
        -------
        ValueError
            If the shape of the actions does not match the shape of the action space.
        """
        shape = (-1)

        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.ndim > 2:
            shape = actions.shape[:-1] + (1,)
            actions = actions.reshape((-1, 2))

        if actions.shape[-1] != self.action_space.shape[-1]:
            raise ValueError(f'Shape mismatch: {actions.shape} vs {self.action_space}')
        
        indexes = np.where(
            (
                np.broadcast_to(self.action_space, (len(actions),) + self.action_space.shape) == actions[:, None, :]
            ).all(axis=2)
        )[1]
        return indexes.reshape(shape)


    def update_seed(self, seed: int|None = None) -> None:
        """
        Update the internal seed and reinitialize the random number generator (PRNG).

        This method updates the agent's internal seed. If a specific seed is provided, it is used to
        initialize the random number generator. If no seed is provided (i.e., seed is None), a new 
        random seed is generated from entropy.

        Parameters
        ----------
        seed : int or None, default=None
            New seed to set for the agent's random number generator. If None, a random seed is generated.
        """
        self._seed = seed
        self._rng = np.random.default_rng(self._seed)

        if self._seed is None:
            self._seed = self._rng.bit_generator.seed_seq.entropy
        return


    def reset(self) -> None:
        """
        Reset the internal state of the agent.
        """
        self._rng = np.random.default_rng(self._seed)
        self.history = Agent.History(self)
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
        Update the agent's internal state based on the reward received and
        any additional information (if available/needed).

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

        def __init__(self, agent: 'Agent'):
            """
            Parameters
            ----------
            agent : Agent
                The agent whose actions and experiences will be tracked by this History instance.
            """
            self._agent = agent
            """Agent being tracked."""

            self._actions = list()
            """List of actions taken by the agent."""
            self._extras = list()
            """List of additional info associated with each action (optional)."""
            self._rewards = list()
            """List of rewards received after each action."""
            self._states = list()
            """List of states observed at each step (optional)."""
            return


        def compute_freqs(
            self,
            key: Union[int, slice, Tuple[Union[int, slice], ...]] = slice(None),
            return_unique: bool = False
        ) -> np.ndarray|Tuple[np.ndarray, np.ndarray]:
            """
            Compute the frequency of each action taken over selected rounds.

            Parameters
            ----------
            key : int, slice, or tuple of ints/slices, default=[:]
                Index, slice, or tuple specifying rounds.
            return_unique : bool, default=False
                If True, returns the unique actions and their counts directly.
                If False, returns an array with counts aligned to the agent's action space.

            Returns
            -------
            : np.ndarray or tuple[np.ndarray, np.ndarray]
                If return_unique is False:
                    A 1D array of shape `(n_arms,)` containing the counts
                    of actions taken for each arm.
                If return_unique is True:
                    A tuple (unique_actions, counts), where:
                        - unique_actions: array of unique actions taken.
                        - counts: array of corresponding action counts.
            """
            unique_actions, counts = np.unique(self.get_actions(key), axis=0, return_counts=True)

            arm_counts = np.zeros(self._agent.n_arms, dtype=int)
            arm_counts[self._agent.action_to_index(unique_actions)] = counts

            if return_unique:
                return unique_actions, counts
            return arm_counts


        def compute_most_common(
            self,
            key: Union[int, slice, Tuple[Union[int, slice], ...]] = slice(None)
        ) -> Tuple[np.ndarray, int]:
            """
            Return the most frequent action.

            Parameters
            ----------
            key : int, slice, or tuple of ints/slices, default=[:]
                Index, slice, or tuple specifying rounds.

            Returns
            -------
            most_common_action : np.ndarray
                The most frequently occurring action.
            frq : int
                Frequency of the most common action.
            """
            unique_actions, counts = np.unique(self.get_actions(key), axis=0, return_counts=True)
            return unique_actions[np.argmax(counts)], int(np.max(counts))


        def get_actions(
            self,
            key: Union[int, slice, Tuple[Union[int, slice], ...]] = slice(None),
            return_index: bool = False
        ) -> np.ndarray:
            """
            Retrieve a subset of the agent's action history.

            This method allows selecting specific rounds or ranges of rounds from the
            action history. Optionally, it can return the indices of the actions rather than
            the actions themselves.

            Parameters
            ----------
            key : int, slice, or tuple of ints/slices, default=slice(None)
                Index, slice, or tuple specifying rounds.
            return_index : bool, default=False
                If True, return the index representation of the actions using the agent's
                `action_to_index` method. If False, return the original actions.

            Returns
            -------
            : np.ndarray
                The selected subset of actions, or their corresponding indices if
                `return_index=True`.
            """
            if return_index:
                return self._agent.action_to_index(np.array(self._actions)[key])
            return np.array(self._actions)[key]


        def get_extras(self, key: Union[int, slice, Tuple[Union[int, slice], ...]] = slice(None)) -> np.ndarray:
            """
            Retrieve all the extra infos collected.

            Parameters
            ----------
            key : int, slice, or tuple of ints/slices, default=[:]
                Index, slice, or tuple specifying rounds.

            Returns
            -------
            : np.ndarray
                Subset of the extra info collected.
            """
            return np.array(self._extras)[key]


        def get_rewards(self, key: Union[int, slice, Tuple[Union[int, slice], ...]] = slice(None)) -> np.ndarray:
            """
            Retrieve a subset of the reward history.

            Parameters
            ----------
            key : int, slice, or tuple of ints/slices, default=[:]
                Index, slice, or tuple specifying rounds.

            Returns
            -------
            : np.ndarray
                Subset of the reward history specified by the key.
            """
            return np.array(self._rewards)[key]


        def get_states(self, key: Union[int, slice, Tuple[Union[int, slice], ...]] = slice(None)) -> np.ndarray:
            """
            Retrieve a subset of the state history.

            Parameters
            ----------
            key : int, slice, or tuple of ints/slices, default=[:]
                Index, slice, or tuple specifying rounds.

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


        def record_extra(self, extra: Any) -> None:
            """
            Record extra info.

            Parameters
            ----------
            extra : any
                The new info to save.
            """
            self._extras.append(extra)
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
            return max(len(self._actions), len(self._rewards), len(self._states), len(self._extras))
