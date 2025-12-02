""" Glosten-Milgrom environment.
"""
import functools
import gymnasium as gym
import numpy as np
import pettingzoo as ptz

from copy import deepcopy
from pettingzoo.utils import AgentSelector
from typing import Callable, Dict, List, Literal, Tuple

from .agents.agent import Agent
from .agents.makers.maker import Maker
from .agents.traders.trader import Trader
from .enums import TraderAction



class CGMEnv(ptz.AECEnv):
    """
    Competitive Glosten-Milgrom (CGM) environment.

    This environment is inspired by the microstructure model introduced
    by Glosten and Milgrom (1985), which analyzes how bid and ask quotes are set
    in the presence of asymmetric information.

    In this implementation, multiple market makers (some informed) interact with traders
    who can BUY, SELL, or PASS based on a private signal, usually equal to the true asset value.
    
    Attributes
    ----------
    generate_vt : callable[[], float]
        Function generating the true asset value at the start of each round.
    n_rounds : int
        Total number of simulation rounds.
    makers : list of Maker
        Market-maker agents (may be informed or uninformed).
    traders : list of Trader
        Trader agents.
    low : float
        Minimum possible price or asset value.
    high : float
        Maximum possible price or asset value.
    decimal_places : int
        Number of decimal places used when rounding rewards.
    info_level : {'full', 'partial'}
        Defines what information informed makers receive:
        - ``'full'``: counterfactual rewards under alternative actions,
          while the trader is allowed to react accordingly.
        - ``'partial'``: counterfactual rewards assuming the trader's
          realized action remains fixed.
    render_mode : {'ascii', 'human'}
        Rendering mode.
    name : str
        Name of the environment (from metadata).
    n_makers : int
        Number of market makers.
    possible_agents : list of Agent
        All agents in the environment (makers + traders).
    agent_name_mapping : dict[str, int]
        Maps agent names to indices.
    round : int
        Current round number.
    true_value : float
        True asset value for the current round.
    trader : Trader
        Trader selected for the current round.
    agents : list of Agent
        Agents active in the current round (makers + selected trader).
    agent_names : list of str
        Names of active agents.
    agent_selection : str
        Name of the agent whose turn it is to act.
    ask_prices : np.ndarray
        Ask prices quoted by each maker.
    bid_prices : np.ndarray
        Bid prices quoted by each maker.
    min_ask_price : float
        Minimum ask price in the current step.
    max_bid_price : float
        Maximum bid price in the current step.
    trader_op : TraderAction or None
        Trader's action for the current step.
    observations : dict[str, dict]
        Latest observation for each agent.
    infos : dict[str, dict]
        Auxiliary per-agent information (e.g., counterfactual rewards).
    rewards : dict[str, float]
        Rewards from the latest completed step.
    cumulative_rewards : dict[str, float]
        Accumulated rewards for each agent.
    terminations : dict[str, bool]
        Whether each agent has terminated.
    truncations : dict[str, bool]
        Whether each agent was truncated due to round limits.

    References
    ----------
    - Glosten, L. R., & Milgrom, P. R. (1985). Bid, ask and transaction prices
    in a specialist market with heterogeneously informed traders. 
    Journal of Financial Economics, 14(1), 71-100.
    https://doi.org/10.1016/0304-405X(85)90044-3
    """

    metadata = {
        'name': 'glosten-milgrom_environment',
        'render_modes': ['ascii', 'human'],
        'information_levels': ['full', 'partial'],
        'valid_operations': list(TraderAction)
    }


    def __init__(
        self,
        generate_vt: Callable[[], float],
        n_rounds: int,
        makers: List[Maker],
        traders: List[Trader],
        low: float = 0.0,
        high: float = 1.0,
        decimal_places: int = 2,
        info_level: Literal['full', 'partial'] = 'partial',
        render_mode: Literal['ascii', 'human'] = 'ascii'
    ):
        """
        Parameters
        ----------
        generate_vt : callable[[], float]
            Function that returns the true value for the asset at the beginning of each round.
        n_rounds : int
            Total number of rounds to simulate.
        makers: list of Maker
            List of market maker agents.
        traders: list of Trader
            List of trader agents.
        low : float, default=0.0
            Minimum possible price or asset value.
        high : float, default=1.0
            Maximum possible price or asset value.
        decimal_places : int, default=2
            Number of decimal places to which rewards are rounded.
        info_level: {'full', 'partial'}, default='partial'
            Level of information provided to informed market makers.
            - 'full': informed makers receive information about the rewards they would have received
                by taking a different action, but the trader is free to change their action accordingly.
            - 'partial': informed makers receve information about the rewards they would have received
                by taking a different action, but the trader's action is fixed.
        render_mode : {'ascii', 'human'}, default='ascii'
            Mode for rendering the environment.
            
        Raises
        ------
        ValueError
            If the number of makers, traders or rounds is smaller than one.
        """
        super().__init__()

        if n_rounds < 1:
            raise ValueError('n_rounds < 1')
        if len(makers) < 1:
            raise ValueError('At least one maker needed')
        if len(traders) < 1:
            raise ValueError('At least one trader needed')

        self.generate_vt = generate_vt
        """Function that returns the true value for the asset at the beginning of each round."""
        self.n_rounds = n_rounds
        """Total number of rounds to simulate."""
        self.makers = makers
        """List of market maker agents."""
        self.traders = traders
        """List of trader agents."""
        self.low = low
        """Minimum possible price."""
        self.high = high
        """Maximum possible price."""
        self.decimal_places = decimal_places
        """Number of decimal places."""
        self.info_level = info_level
        """Level of information provided to informed market makers."""
        self.render_mode = render_mode
        """Rendering mode."""

        self.name = CGMEnv.metadata['name']
        """Name of the environment."""
        self.n_makers = len(self.makers)
        """Total number of market makers."""
        
        self.possible_agents: List[Agent] = self.makers + self.traders
        """List of all agents in the environment."""
        self.agent_name_mapping = {agent.name: idx for idx, agent in enumerate(self.possible_agents)}
        """Mapping from agent name to index."""
    
        self._np_random: np.random.Generator
        """PRNG."""
        self._np_random_seed: int
        """Seed of the PRNG."""
        self._agent_selector: AgentSelector
        """Agent selector."""

        self.round: int
        """Current round number."""
        self.true_value: float
        """Current true asset value."""

        self.trader: Trader
        """Currently selected trader."""
        self.agents: List[Agent]
        """Agents active in the current round."""
        self.agent_names: List[str]
        """Names of agents active in the current round."""
        self.agent_selection: str
        """Name of the agent whose turn it is to act."""

        self.ask_prices: np.ndarray
        """Ask prices quoted by each market maker."""
        self.bid_prices: np.ndarray
        """Bid prices quoted by each market maker."""

        self.min_ask_price: float
        """Minimum ask price offered by any maker in the current step."""
        self.max_bid_price: float
        """Maximum bid price offered by any maker in the current step."""
        self.trader_op: TraderAction|None
        """Action taken by the trader in the current step."""

        self.observations: Dict[str, Dict]
        """Observation available per agent."""
        self.infos: Dict[str, Dict]
        """Information available per agent."""

        self.rewards: Dict[str, float]
        """Rewards assigned to each agent."""
        self.cumulative_rewards: Dict[str, float]
        """Sum of the rewards assigned to each agent."""
        self._cumulative_rewards: Dict[str, float]
        """Alias of `self.cumulative_rewards`, kept for compatibility."""

        self.terminations: Dict[str, bool]
        """Indicates whether each agent has terminated."""
        self.truncations: Dict[str, bool]
        """Indicates whether each agent was truncated."""
        return


    @staticmethod
    def _partial_reset(func: Callable) -> Callable:
        """
        Decorator to reset certain environment variables at the start of each new round.
        
        It resets maker prices, min/max prices, trader action, observations, and infos
        if the round has finished.

        Parameters
        ----------
        func : callable
            The step function to be wrapped.
        
        Returns
        -------
        : callable
            The wrapped step function with partial reset functionality.
        """
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs) -> None:
            result = func(self, *args, **kwargs)

            if result[4]['round_finished']:
                self.ask_prices = np.zeros(self.n_makers, dtype=np.float64)
                self.bid_prices = np.zeros(self.n_makers, dtype=np.float64)
        
                self.min_ask_price = self.high
                self.max_bid_price = self.low
                self.trader_action = None

                self.observations = {agent.name: self.observe(agent) for agent in self.possible_agents}
                self.infos = {'round_finished': False} | {agent.name: {} for agent in self.possible_agents}
            return result
        return wrapper


    def action_space(self, agent: str|Agent, seed: int|None = None) -> gym.Space:
        """
        Return the action space associated with a given agent.

        Makers choose both ask and bid prices, while traders select a single action
        from the set of valid operations.

        Parameters
        ----------
        agent : str or Agent
            The target agent, either as an identifier (string) or an Agent instance.
        seed : int or None, default=None
            Random seed to initialize the PRNG that is used to sample from the space.
            If `None`, a random seed is used.

        Returns
        -------
        : gym.Space
            The action space available to the specified agent.
        """
        agent = self._get_agent(agent) if isinstance(agent, str) else agent

        if agent.type.is_maker():
            space = gym.spaces.Dict({
                'ask_price': gym.spaces.Box(self.low, self.high, dtype=np.float64, seed=seed),
                'bid_price': gym.spaces.Box(self.low, self.high, dtype=np.float64, seed=seed)
            })
        else:
            space = gym.spaces.Dict({
                'operation': gym.spaces.Discrete(len(CGMEnv.metadata['valid_operations']), seed=seed)
            })
        return space


    def observation_space(self, agent: str|Agent, seed: int|None = None) -> gym.Space:
        """
        Return the observation space for a given agent.

        Traders observe the true asset value, the current minimum ask price, and the
        current maximum bid price. Market makers receive an empty observation.

        Parameters
        ----------
        agent : str or Agent
            The target agent, either as an identifier (string) or an Agent instance.
        seed : int or None, default=None
            Random seed to initialize the PRNG that is used to sample from the space.
            If `None`, a random seed is used.

        Returns
        -------
        : gym.Space
            A Gym space object describing the structure of the agent's observation.
            For traders, this is a `Dict` space with the following keys:
                - 'true_value' (Box): The true value of the asset.
                - 'min_ask_price' (Box): The lowest current ask price.
                - 'max_bid_price' (Box): The highest current bid price.
            For market makers, this is an empty `Dict` space.
        """
        agent = self._get_agent(agent) if isinstance(agent, str) else agent

        if agent.type.is_maker():
            space = gym.spaces.Dict({})
        else:
            space = gym.spaces.Dict({
                'true_value': gym.spaces.Box(self.low, self.high, dtype=np.float64, seed=seed),
                'min_ask_price': gym.spaces.Box(self.low, self.high, dtype=np.float64, seed=seed),
                'max_bid_price': gym.spaces.Box(self.low, self.high, dtype=np.float64, seed=seed)
            })
        return space


    def observe(self, agent: str|Agent) -> Dict:
        """
        Return the current observation for the specified agent.

        Parameters
        ----------
        agent : str or Agent
            The target agent, either as an identifier (string) or an Agent instance.

        Returns
        -------
        : dict
            A dictionary containing the agent's current observation.
            For traders, the dictionary contains:
                - 'true_value' (float): The true value of the asset.
                - 'min_ask_price' (float): The lowest ask price quoted by any maker.
                - 'max_bid_price' (float): The highest bid price quoted by any maker.
            For market makers, an empty dictionary is returned.
        """
        agent = self._get_agent(agent) if isinstance(agent, str) else agent

        if agent.type.is_maker():
            observation = {}
        else:
            observation = {
                'true_value': self.true_value,
                'min_ask_price': self.min_ask_price,
                'max_bid_price': self.max_bid_price
            }
        return observation


    def inform(self, agent: str|Agent) -> Dict:
        """
        Provide additional information to the given agent based on their role and knowledge.

        Informed market makers receive:
        - Their agent index.
        - The true asset value.
        - The list of current (ask, bid) prices.
        - The rewards corresponding to each possible action in their action space.
        Uninformed agents (e.g., traders or uninformed makers) receive an empty dictionary.

        Parameters
        ----------
        agent : str or Agent
            The target agent, either as an identifier (string) or an Agent instance.

        Returns
        -------
        info : dict of str
            A dictionary containing additional information if the agent is an informed maker;
            otherwise, an empty dictionary.
            Keys (for informed makers) include:
                - 'agent_index' (int): Index of the agent in `self.possible_agents`.
                - 'true_value' (float): The true value of the asset.
                - 'actions' (np.ndarray): Array of (ask, bid) action pairs.
                - 'trader_op' (np.ndarray or int): The index of the operation performed by the trader.
                - 'rewards' (np.ndarray): Rewards corresponding to each action.
        """
        agent = self._get_agent(agent) if isinstance(agent, str) else agent
        agent_idx = self._get_agent_idx(agent)

        if agent.type.is_maker() and agent.type.is_informed():
            action_space = agent.action_space
            
            ask_prices = np.tile(self.ask_prices, (len(action_space), 1))
            bid_prices = np.tile(self.bid_prices, (len(action_space), 1))
            ask_prices[:, agent_idx] = action_space[:, 0]
            bid_prices[:, agent_idx] = action_space[:, 1]

            if self.info_level == 'full':
                trader = deepcopy(self.trader)
                trader.reset()
                rewards = self._compute_rewards(trader=trader, ask_prices=ask_prices, bid_prices=bid_prices)[:, agent_idx] 
                trader_op = trader.history.get_actions()
            elif self.info_level == 'partial':
                rewards = self._compute_rewards(ask_prices=ask_prices, bid_prices=bid_prices)[:, agent_idx]
                trader_op = self.trader_op
            else:
                raise ValueError(f'Unknown info level {self.info_level}')

            info = {
                'agent_index': agent_idx,
                'true_value': self.true_value,
                'actions': np.array(list(zip(self.ask_prices, self.bid_prices)), dtype=np.float64),
                'trader_op': trader_op,
                'rewards': rewards
            }
        else:
            info = {}
        return info


    def state(self) -> Dict:
        """
        Return the global state of the environment.

        It includes current prices, round count, rewards, and agent status.

        Returns
        -------
        : dict of str
            A dictionary with the full internal state.
        """
        state = {
            'round': self.round,
            'possible_agents': [agent.name for agent in self.possible_agents],
            'agents': [agent.name for agent in self.agents],
            'true_value': self.true_value,
            'ask_prices': self.ask_prices,
            'bid_prices': self.bid_prices,
            'min_ask_price': self.min_ask_price,
            'max_bid_price': self.max_bid_price,
            'trader_action': self.trader_op,
            'rewards': np.array(list(self.rewards.values()), dtype=np.float64),
            'cumulative_rewards': np.array(list(self.cumulative_rewards.values()), dtype=np.float64),
            'terminations': np.array(list(self.terminations.values()), dtype=np.bool),
            'truncations': np.array(list(self.truncations.values()), dtype=np.bool)
        }
        return state


    def reset(self, seed: int|None = None, options: Dict|None = None) -> Tuple[Dict, Dict]:
        """
        Reset the environment at the beginning of a new sequence of rounds.

        Randomly selects one trader and initializes maker prices and true value.

        Parameters
        ----------
        seed : int or None, default=None
            Random seed for reproducibility. If None, a random seed wil be used.
        options : dict or None, default=None
            Additional options (not used).

        Returns
        -------
        observations : dict of str
            Initial observations for all agents.
        infos : dict of str
            Additional information for each agent.
        """
        self._np_random, self._np_random_seed = gym.utils.seeding.np_random(seed)

        self.round = 0
        self.true_value = self.generate_vt()

        self.trader = self._np_random.choice(self.traders)
        self.agents = self.makers + [self.trader]
        self.agent_names = [agent.name for agent in self.agents]

        self._agent_selector = AgentSelector(self.agent_names)
        self.agent_selection = self._agent_selector.next()

        self.ask_prices = np.zeros(self.n_makers, dtype=np.float64)
        self.bid_prices = np.zeros(self.n_makers, dtype=np.float64)
        
        self.min_ask_price = self.high
        self.max_bid_price = self.low
        self.trader_op = None

        self.rewards = {agent.name: 0.0 for agent in self.possible_agents}
        self.cumulative_rewards = {agent.name: 0.0 for agent in self.possible_agents}
        self._cumulative_rewards = self.cumulative_rewards

        self.observations = {agent.name: self.observe(agent) for agent in self.possible_agents}
        self.infos = {'round_finished': False} | {agent.name: {} for agent in self.possible_agents}

        self.terminations = {agent.name: not agent in self.agents for agent in self.possible_agents}
        self.truncations = {agent.name: False for agent in self.possible_agents}
        return self.observations, self.infos


    @_partial_reset
    def step(self, action: Dict[str, TraderAction|int|float|np.ndarray]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        Execute a step in the environment with the provided action.

        Each maker sets prices; the trader decides to buy, sell or pass.
        Rewards are computed based on the resulting transaction and the true value.

        Parameters
        ----------
        action : dict of str
            Dictionary containing the action for the current agent.

        Returns
        -------
        observations : dict
            New observations after this step.
        rewards : dict of str to float
            Rewards for each agent.
        terminations : dict of str to bool
            Boolean flags indicating if agents have terminated.
        truncations : dict of str to bool
            Boolean flags indicating if agents were truncated (round limit reached).
        infos : dict
            Additional info.

        Raises
        ------
        ValueError
            If the maximum number of rounds has been reached or an invalid action
            has been performed given the current agent.

        Notes
        -----
        For market makers, the action is a dictionary containing 'ask_price' and 'bid_price' 
        values. These values can be passed either as scalar floats (e.g., 0.8) or 
        as NumPy arrays with a single float entry (e.g., np.array([0.8])) — both 
        formats are accepted.
        """
        curr_agent_name = self.agent_selection
        curr_agent_idx = self.agent_name_mapping[curr_agent_name]
        curr_agent = self.possible_agents[curr_agent_idx]
        
        action, ok = self._assert_and_format_action(curr_agent_name, action)

        # Update state given current action
        if self.round >= self.n_rounds:
            raise ValueError('Maximum number of rounds reached')
        if not ok:
            raise ValueError(f'Action {action} is not valid for agent {curr_agent_name}')

        if curr_agent.type.is_maker():
            self.ask_prices[curr_agent_idx] = action['ask_price']
            self.bid_prices[curr_agent_idx] = action['bid_price']

            self.min_ask_price = action['ask_price'] if self.min_ask_price > action['ask_price'] else self.min_ask_price
            self.max_bid_price = action['bid_price'] if self.max_bid_price < action['bid_price'] else self.max_bid_price
        else:
            self.trader_op = action['operation']
        
        # Compute rewards
        if self._agent_selector.is_last():
            rewards = self._compute_rewards()
            for idx, agent in enumerate(self.possible_agents):
                self.rewards[agent.name] = rewards[idx]
                self.cumulative_rewards[agent.name] = round(self.cumulative_rewards[agent.name] + self.rewards[agent.name], self.decimal_places)

        # Update observations
        # Infos, truncations and terminations will be updated only if the round ends
        self.observations = {agent.name: self.observe(agent) for agent in self.possible_agents}

        # Render the environment
        if self.render_mode == 'human':
            self.render()

        # Update for next round and upadate infos, terminations and truncations
        if self._agent_selector.is_last():
            self.infos = {'round_finished': True} | {agent.name: self.inform(agent) for agent in self.possible_agents}            
            
            self.round += 1
            self._true_value = self.generate_vt()

            self.trader = self._np_random.choice(self.traders)
            self.agents = self.makers + [self.trader] if self.round < self.n_rounds else []
            self.agent_names = [agent.name for agent in self.agents] if self.round < self.n_rounds else []

            self.terminations = {agent.name: not agent in self.agents for agent in self.possible_agents}
            self.truncations = {agent.name: self.round >= self.n_rounds for agent in self.possible_agents}

            # Variables reset will be handled by the decorator
        self.agent_selection = self._agent_selector.next()
        return self.observations, self.rewards, self.terminations, self.truncations, self.infos


    def render(self) -> str|None:
        """
        Render the environment state according to the selected mode.

        Returns
        -------
        : str or None
            ASCII string representation if render mode is 'ascii', otherwise None.

        Raises
        ------
        ValueError
            If called without setting a valid `render_mode`.

        Notes
        -----
        The environment supports two render modes:
        - 'human': prints directly to standard output.
        - 'ascii': returns the string representation of the environment.
        """
        if self.render_mode is None:
            raise ValueError('Calling render method without specifying any render mode')

        if self.render_mode == 'human':
            print(self)
        elif self.render_mode == 'ascii':
            return self.__str__()
        return None


    def _assert_and_format_action(
        self,
        agent: str|Agent,            
        action: Dict[str, TraderAction|int|float|np.ndarray]
    ) -> Tuple[Dict[str, TraderAction|float], bool]:
        """
        Validate and format the agent's action.

        It converts action values to proper types and
        checks if they are within the allowed space.

        Parameters
        ----------
        agent : str or Agent
            The target agent, either as an identifier (string) or an Agent instance.
        action : dict
            Raw action provided.

        Returns
        -------
        formatted_action : dict of str of TraderAction or float
            Cleaned and validated action.
        ok : bool
            Whether the action is valid.
        """
        try:
            for k, v in action.items():
                action[k] = np.array([v], dtype=np.float64) if isinstance(v, float) else v
                action[k] = v.value if isinstance(v, TraderAction) else action[k]

            ok = self.action_space(agent).contains(action)

            for k, v in action.items():
                action[k] = round(float(v[0]), self.decimal_places) if isinstance(v, np.ndarray) else v
                action[k] = TraderAction(v) if isinstance(v, int) else action[k]
        except Exception :
            return None, False
        return action, ok


    def _compute_rewards(
        self,
        true_value: float|None = None,
        trader: Trader|str|None = None,
        trader_op: TraderAction|None = None,
        ask_prices: np.ndarray|None = None,
        bid_prices: np.ndarray|None = None
    ) -> np.ndarray:
        """
        Compute rewards for all agents based on market prices and a trader's action.

        Rewards depend on the trader's operation (BUY, SELL, PASS) and the market state 
        (ask and bid prices). If multiple samples of prices are provided, rewards are 
        computed separately for each sample.

        If `trader_op` is None and a trader is specified, the trader's policy will be used
        to determine the action for each sample of prices.

        Parameters
        ----------
        true_value : float or None, default=None
            The true value of the asset. Defaults to `self.true_value` if None.
        trader : Trader, str or None, default=None
            Trader, or its name, whose action is considered. Defaults to `self.trader` if None.
        trader_op : TraderAction or None, default=None
            Trader's operation (BUY, SELL, PASS). If None, determined by the trader's policy.
        ask_prices : np.ndarray or None, default=None
            Current ask prices from market makers. Defaults to `self.ask_prices` if None.
            Can be 1D (single sample) or 2D (multiple samples).
        bid_prices : np.ndarray or None, default=None
            Current bid prices from market makers. Defaults to `self.bid_prices` if None.
            Can be 1D or 2D.

        Returns
        -------
        : np.ndarray
            Rewards for each agent. Shape is `(n_samples, max_num_agents)` for multiple
            samples, or `(max_num_agents,)` for a single sample. Rewards are rounded
            to `self.decimal_places`.

        Raises
        ------
        ValueError
            If `trader_op` is not a valid `TraderAction`.

        Notes
        -----
        - PASS: all rewards are zero.
        - BUY: trader gains `true_value - min_ask_price`; makers selling at `min_ask_price` lose proportionally.
        - SELL: trader gains `max_bid_price - true_value`; makers buying at `max_bid_price` lose proportionally.
        - If multiple samples are provided, computation is performed for each sample independently.
        - If `trader_op` is None but a trader is provided, that trader will determine the action for each sample.
        """
        if true_value is None:
            true_value = self.true_value

        if ask_prices is None or bid_prices is None:
            ask_prices = self.ask_prices
            bid_prices = self.bid_prices

            min_ask_price = np.array([self.min_ask_price])
            max_bid_price = np.array([self.max_bid_price])
        else:
            min_ask_price = np.min(ask_prices, axis=-1, keepdims=(ask_prices.ndim == 1))
            max_bid_price = np.max(bid_prices, axis=-1, keepdims=(bid_prices.ndim == 1))

        shape = ((1,) if ask_prices.ndim == 1 else ask_prices.shape[:-1]) + (self.max_num_agents,)
        rewards = np.zeros(shape, dtype=np.float64)     

        if trader_op is None:
            if trader is None:
                trader_op = self.trader_op
            else:
                trader: Trader = deepcopy(self._get_agent(trader)) if isinstance(trader, str) else trader
                if len(min_ask_price) == 1:
                    trader_op = trader.act({
                            'true_value': true_value,
                            'min_ask_price': min_ask_price[0],
                            'max_bid_price': max_bid_price[0]
                        })['operation']
                else:
                    for i in range(len(min_ask_price)):
                        trader_op = trader.act({
                            'true_value': true_value,
                            'min_ask_price': min_ask_price[i],
                            'max_bid_price': max_bid_price[i]
                        })['operation']
                        rewards[i, :] = self._compute_rewards(
                            true_value = true_value,
                            trader_op = trader_op,
                            ask_prices = ask_prices[i],
                            bid_prices = bid_prices[i]
                        )
                    return np.round(rewards.squeeze(), self.decimal_places)
        
        if trader is None:
            trader = self.trader

        if trader_op == TraderAction.PASS:
            return np.round(rewards.squeeze(), self.decimal_places)
        elif trader_op == TraderAction.BUY:
            reward = self.true_value - min_ask_price
            selected_makers_idx = np.where(ask_prices == min_ask_price[:, None])
            count_selected_makers = np.sum(ask_prices == min_ask_price[:, None], axis=-1)
        elif trader_op == TraderAction.SELL:
            reward = max_bid_price - self.true_value
            selected_makers_idx = np.where(bid_prices == max_bid_price[:, None])
            count_selected_makers = np.sum(bid_prices == max_bid_price[:, None], axis=-1)
        else:
            raise ValueError(f'Invalid trader_action: {trader_op}')

        rewards[:, self._get_agent_idx(trader)] = reward
        rewards[selected_makers_idx[0], selected_makers_idx[1]] = (- reward / count_selected_makers)[selected_makers_idx[0]]
        return np.round(rewards.squeeze(), self.decimal_places)


    def _get_agent(self, agent_name: str) -> Agent:
        """
        Retrieve an Agent instance by its name.

        Parameters
        ----------
        agent_name : str
            The unique identifier of the agent.

        Returns
        -------
        : Agent
            The corresponding Agent object from the environment.
        """
        return self.possible_agents[self.agent_name_mapping[agent_name]]


    def _get_agent_idx(self, agent: str|Agent) -> int:
        """
        Get the index of an agent in the environment.

        This function accepts either an Agent instance or the agent's name
        as a string and returns its corresponding index in the environment.

        Parameters
        ----------
        agent : str or Agent
            The agent's name or an Agent instance.

        Returns
        -------
        : int
            The index of the agent in `self.possible_agents`.
        """
        name = agent.name if isinstance(agent, Agent) else agent
        return self.agent_name_mapping[name]


    def __str__(self) -> str:
        s = f'round {self.round}/{self.n_rounds}:\n' + \
            f' - true value -> {self.true_value}\n' + \
            f' - curr. agent -> {self.agent_selection}\n' + \
            f' - min ask price -> {self.min_ask_price}\n' + \
            f' - max bid price -> {self.max_bid_price}\n' + \
            f' - trader action -> {self.trader_op}\n' + \
            f' - total rewards -> {self.cumulative_rewards}\n'
        return s
