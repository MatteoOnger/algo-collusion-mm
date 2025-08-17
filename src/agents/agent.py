from abc import ABC, abstractmethod
from typing import Dict, List



class Agent(ABC):
    """
    Interface for agents in the GM environment.

    Attributes
    ----------
    action_space : list
        All possible available actions.
    n_arms : int
        Number of actions (arms) in the action space.
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
