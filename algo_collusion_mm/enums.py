""""""
from enum import Enum



class AgentType(Enum):
    """
    Agent types in the GM environment.
    """

    ABSTRACT = 0
    MAKER_U = 1
    MAKER_I = 2
    TRADER = 3


    def is_informed(self) -> bool:
        """
        Check if the agent type is an informed maker.

        Returns
        -------
        : bool
            True if the agent type is an informed maker, False otherwise.
        """
        return self == AgentType.MAKER_I


    def is_maker(self) -> bool:
        """
        Check if the agent type is a maker.

        Returns
        -------
        : bool
            True if the agent type is a maker, False otherwise.
        """
        return self == AgentType.MAKER_U or self == AgentType.MAKER_I


    def is_trader(self) -> bool:
        """
        Check if the agent type is a trader.

        Returns
        -------
        : bool
            True if the agent type is a trader, False otherwise.
        """
        return self == AgentType.TRADER



class TraderAction(Enum):
    """
    Possible trader actions.
    """
    
    BUY = 0
    SELL = 1
    PASS = 2
