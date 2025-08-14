from abc import ABC, abstractmethod
from typing import Dict



class Agent(ABC):
    """
    Interface for agents in the GM environment.
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
