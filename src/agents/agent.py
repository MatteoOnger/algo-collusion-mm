from abc import ABC, abstractmethod
from typing import Any, Dict, List



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
        self.history = Agent.History(self.action_space)
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
        Container for tracking the sequence of actions taken by the agent.

        Attributes
        ----------
        history : list
            Chronological list of actions taken.
        freqs : dict
            Mapping from action to the number of times it has been chosen.
        """
        
        def __init__(self, action_space: List):
            """
            Parameters
            ----------
            action_space : list
                The set of possible actions available to the agent.
            """
            self.history = list()
            self.freqs = {action:0 for action in action_space}
            return


        def record(self, strategy: Any) -> None:
            """
            Record a new action in the history.

            Parameters
            ----------
            action : Any
                The strategy chosen by the agent, must be part of `action_space`.
            """
            self.history.append(strategy)
            self.freqs[strategy] += 1
            return
        

        def __len__(self) -> int:
            return len(self.history)