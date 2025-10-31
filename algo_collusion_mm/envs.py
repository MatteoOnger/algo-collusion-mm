""" Glosten-Milgrom environment.
"""
import functools
import gymnasium as gym
import numpy as np
import pettingzoo as ptz

from enum import Enum
from pettingzoo.utils import AgentSelector
from typing import Callable, Dict, List, Literal, Tuple



class GMEnv(ptz.AECEnv):
    """
    Glosten-Milgrom (GM) environment for simulating a single-asset dealer market.

    This environment is inspired by the microstructure model introduced
    by Glosten and Milgrom (1985), which analyzes how bid and ask quotes are set
    in the presence of asymmetric information.

    In this implementation, multiple market makers (some informed) interact with traders
    who can BUY, SELL, or PASS based on a private signal, usually equal to the true asset value.
    
    Attributes
    ----------
    generate_vt : callable[[], float]
        Function that returns the true value for the asset at the beginning of each episode.
    n_episodes : int
        Total number of episodes to simulate.
    n_makers_u : int
        Number of uninformed market makers.
    n_makers_i : int
        Number of informed market makers.
    n_traders : int
        Number of traders.
    low : float
        Minimum possible price or asset value.
    high : float
        Maximum possible price or asset value.
    agents_action_space: dict of str to np.ndarray
        Mapping from agent name to its action space (used to inform informed agents).
    decimal_places : int
        Number of decimal places to which rewards are rounded.
    render_mode : literal['ascii', 'human'] 
        Mode used for rendering.
    name : str
        Name of the environment (from metadata).
    n_makers : int
        Total number of market makers (informed + uninformed).
    makers_u : list of str
        Names of uninformed maker agents.
    makers_i : list of str
        Names of informed maker agents.
    makers : list of str
        Names of all market maker agents (makers_u + makers_i).
    traders : list of str
        Names of all trader agents.
    possible_agents : list of str
        Name of all agents in the environment (makers + traders).
    agent_name_mapping : dict of str to int
        Mapping from agent name to index.
    agents_type : list of GM2Env.AgentType
        List of agent types (MAKER_U, MAKER_I, or TRADER).
    episode : int
        Current episode number. Starts at 0 and increments after each complete step.
    trader : int
        Name of the trader in the current episode.
    agents : list of str
        Name of agents active in the current episode (all makers + one trader).
    agent_selection : str
        Name of the agent whose turn it is to act.
    rewards : dict of str to float
        Latest rewards assigned to each agent.
    cumulative_rewards : dict of str to float
        Sum of the rewards assigned to each agent.
    observations : dict of str to dict
        Most recent observation available to each agent.
    infos : dict of str to dict
        Auxiliary information returned to agents (e.g., whether episode has finished).
    terminations : dict of str to bool
        Indicates whether each agent has terminated (e.g., episode ended).
    truncations : dict of str to bool
        Indicates whether each agent was truncated due to episode length.
    true_value : float
        The true underlying value of the asset for the current episode.
    min_ask_price : float
        Minimum ask price offered by any maker in the current step.
    max_bid_price : float
        Maximum bid price offered by any maker in the current step.
    trader_action : GM2Env.TraderAction or None
        Action taken by the trader in the current step (None, BUY, SELL, or PASS).

    References
    ----------
    - Glosten, L. R., & Milgrom, P. R. (1985). Bid, ask and transaction prices
    in a specialist market with heterogeneously informed traders. 
    Journal of Financial Economics, 14(1), 71-100.
    https://doi.org/10.1016/0304-405X(85)90044-3
    """

    metadata = {
        'name': 'glosten-milgrom_environment',
        'render_modes': ['ascii', 'human']
    }


    class AgentType(Enum):
        """
        Agent types in the GM environment.
        """
        MAKER_U = 0
        MAKER_I = 1
        TRADER = 2


    class TraderAction(Enum):
        """
        Possible trader actions.
        """
        BUY = 0
        SELL = 1
        PASS = 2


    def __init__(
        self,
        generate_vt: Callable[[], float],
        n_episodes: int,
        n_makers_u: int,
        n_makers_i: int,
        n_traders: int,
        low: float = 0.0,
        high: float = 1.0,
        agents_action_space: Dict[str, np.ndarray] = dict(), 
        decimal_places: int = 2,
        render_mode: Literal['ascii', 'human'] = 'ascii'
    ):
        """
        Parameters
        ----------
        generate_vt : callable[[], float]
            Function that returns the true value for the asset at the beginning of each episode.
        n_episodes : int
            Total number of episodes to simulate.
        n_makers_u : int
            Number of uninformed market makers.
        n_makers_i : int
            Number of informed market makers.
        n_traders : int
            Number of traders.
        low : float, default=0.0
            Minimum possible price or asset value.
        high : float, default=1.0
            Maximum possible price or asset value.
        agents_action_space: dict of str to np.ndarray, default={}
            Mapping from agent names to their respective action spaces. Each action space must be 
            a subset or a discretization of the environment's action space (i.e., of `self`). 
            This parameter is used only to inform informed agents.
        decimal_places : int, default=2
            Number of decimal places to which rewards are rounded.
        render_mode : {'ascii', 'human'}, default='ascii'
            Mode for rendering the environment.
            
        Raises
        ------
        ValueError
            If the number of makers, traders or episodes is smaller than one.
        """
        super().__init__()

        if n_makers_i + n_makers_u < 1:
            raise ValueError('n_makers < 1')
        if n_traders < 1:
            raise ValueError('n_traders < 1')
        if n_episodes < 1:
            raise ValueError('n_episodes < 1')

        self.generate_vt = generate_vt
        """Function that returns the true value for the asset at the beginning of each episode."""
        self.n_episodes = n_episodes
        """Total number of episodes to simulate."""
        self.n_makers_u = n_makers_u
        """Number of uninformed market makers."""
        self.n_makers_i = n_makers_i
        """Number of informed market makers."""
        self.n_traders = n_traders
        """Number of traders."""
        self.low = low
        """Minimum possible price."""
        self.high = high
        """Maximum possible price."""
        self.agents_action_space = agents_action_space
        """Mapping from agent names to their respective action spaces."""
        self.decimal_places = decimal_places
        """Number of decimal places."""
        self.render_mode = render_mode
        """Rendering mode."""

        self.name = GMEnv.metadata['name']
        """Name of the environment."""
        self.n_makers = n_makers_u + n_makers_i
        """Total number of market makers."""

        self.makers_u = [f'maker_u_{idx}' for idx in range(n_makers_u)]
        """Names of uninformed maker agents."""
        self.makers_i = [f'maker_i_{idx}' for idx in range(n_makers_i)]
        """Names of informed maker agents."""
        self.makers = self.makers_u + self.makers_i
        """Names of all market maker agents."""
        self.traders = [f'trader_{idx}' for idx in range(n_traders)]
        """Names of all trader agents."""

        self.possible_agents = self.makers + self.traders
        """Name of all agents in the environment."""
        self.agent_name_mapping = {agent:idx for idx, agent in enumerate(self.possible_agents)}
        """Mapping from agent name to index."""
        self.agents_type = (
            [GMEnv.AgentType.MAKER_U] * n_makers_u +
            [GMEnv.AgentType.MAKER_I] * n_makers_i +
            [GMEnv.AgentType.TRADER] * n_traders
        )
        """List of agent types."""

        self._np_random: np.random.Generator
        """PRNG."""
        self._np_random_seed: int
        """Seed of the PRNG."""
        self._agent_selector: AgentSelector
        """Agent selector."""

        self.episode: int
        """Current episode number."""
        self.true_value: float
        """Current true asset value."""

        self.trader: str
        """Name of the currently selected trader."""
        self.agents: List[str]
        """Name of agents active in the current episode."""
        self.agent_selection: str
        """Name of the agent whose turn it is to act."""

        self.min_ask_price: float
        """Minimum ask price offered by any maker in the current step."""
        self.max_bid_price: float
        """Maximum bid price offered by any maker in the current step."""
        self.trader_action: GMEnv.TraderAction|None
        """Action taken by the trader in the current step."""

        self._ask_prices: np.ndarray
        """Ask prices quoted by each market maker."""
        self._bid_prices: np.ndarray
        """Bid prices quoted by each market maker."""

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
        Decorator to reset certain environment variables at the start of each new episode.
        
        It resets maker prices, min/max prices, trader action, observations, and infos
        if the episode has finished.

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
            """
            """
            result = func(self, *args, **kwargs)

            if result[4]['episode_finished']:
                self._ask_prices = np.zeros(self.n_makers, dtype=np.float64)
                self._bid_prices = np.zeros(self.n_makers, dtype=np.float64)
        
                self.min_ask_price = self.high
                self.max_bid_price = self.low
                self.trader_action = None

                self.observations = {agent: self.observe(agent) for agent in self.possible_agents}
                self.infos = {'episode_finished': False} | {agent: {} for agent in self.possible_agents}
            return result
        return wrapper


    def action_space(self, agent: str, seed: int|None = None) -> gym.Space:
        """
        Return the action space for a given agent.

        Makers choose ask and bid prices. Traders choose whether to buy, sell or pass.

        Parameters
        ----------
        agent : str
            Name of the agent.
        seed : int or None, default=None
            Random seed. Default is None, so a random seed will be used.

        Returns
        -------
        : gym.Space
            The agent's action space.
        """
        if self._ismaker(agent):
            space = gym.spaces.Dict({
                'ask_price': gym.spaces.Box(self.low, self.high, dtype=np.float64, seed=seed),
                'bid_price': gym.spaces.Box(self.low, self.high, dtype=np.float64, seed=seed)
            })
        else:
            space = gym.spaces.Dict({
                'operation': gym.spaces.Discrete(len(GMEnv.TraderAction), seed=seed)
            })
        return space


    def observation_space(self, agent: str, seed: int|None = None) -> gym.Space:
        """
        Return the observation space for a given agent.

        Traders observe the true value, the minimum ask, and the maximum bid. 
        Makers receive an empty observation.

        Parameters
        ----------
        agent : str
            Name of the agent.
        seed : int or None, default=None
            Random seed. Default is None, so a random seed will be used.

        Returns
        -------
        : gym.Space
            A Gym space object describing the structure of the observation for the agent.
            For traders: a Dict space with keys:
                - 'true_value' (Box): True value of the asset.
                - 'min_ask_price' (Box): Lowest current ask price.
                - 'max_bid_price' (Box): Highest current bid price.
            For market makers: an empty Dict space.
        """
        if self._ismaker(agent):
            space = gym.spaces.Dict({})
        else:
            space = gym.spaces.Dict({
                'true_value': gym.spaces.Box(self.low, self.high, dtype=np.float64, seed=seed),
                'min_ask_price': gym.spaces.Box(self.low, self.high, dtype=np.float64, seed=seed),
                'max_bid_price': gym.spaces.Box(self.low, self.high, dtype=np.float64, seed=seed)
            })
        return space


    def observe(self, agent: str) -> Dict:
        """
        Return the current observation for the given agent.

        Parameters
        ----------
        agent : str
            Name of the agent.

        Returns
        -------
        : dict
            A dictionary containing the current observation for the specified agent.
            Keys (for traders) include:
                - 'true_value' (float): The true value of the asset.
                - 'min_ask_price' (float): The lowest ask price quoted by any maker.
                - 'max_bid_price' (float): The highest bid price quoted by any maker.
            For market makers, an empty dictionary is returned.
        """
        if self._ismaker(agent):
            observation = {}
        else:
            observation = {
                'true_value': self.true_value,
                'min_ask_price': self.min_ask_price,
                'max_bid_price': self.max_bid_price
            }
        return observation
    

    def inform(self, agent: str) -> Dict:
        """
        Provides additional information to the given agent based on their role and knowledge.

        Informed market makers receive:
        - Their agent index.
        - The true asset value.
        - The list of current (ask, bid) prices.
        - The rewards corresponding to each possible action in their action space.

        Uninformed agents (e.g., traders or uninformed makers) receive an empty dictionary.

        Parameters
        ----------
        agent : str
            Name of the agent requesting information.

        Returns
        -------
        info : dict
            A dictionary containing additional information if the agent is an informed maker;
            otherwise, an empty dictionary.
            Keys (for informed makers) include:
                - 'agent_index' (int): Index of the agent in `self.possible_agents`.
                - 'true_value' (float): The true value of the asset.
                - 'actions' (np.ndarray): Array of (ask, bid) action pairs.
                - 'rewards' (np.ndarray): Rewards corresponding to each action.
        """
        i = self.agent_name_mapping[agent]

        if self._ismaker(agent) and self._isinformed(agent):
            if agent in self.agents_action_space.keys():
                action_space = self.agents_action_space[agent]
                
                ask_prices = np.tile(self._ask_prices, (len(action_space), 1))
                bid_prices = np.tile(self._bid_prices, (len(action_space), 1))
                ask_prices[:, i] = action_space[:, 0]
                bid_prices[:, i] = action_space[:, 1]

                rewards = self._compute_rewards(self.trader_action, ask_prices, bid_prices)[:, i] 
            else:
                rewards = None

            info = {
                'agent_index': i,
                'true_value': self.true_value,
                'actions': np.array(list(zip(self._ask_prices, self._bid_prices)), dtype=np.float64),
                'rewards': rewards
            }
        else:
            info = {}
        return info


    def state(self) -> Dict:
        """
        Return the global state of the environment.

        It includes current prices, episode count, rewards, and agent status.

        Returns
        -------
        : dict of str
            A dictionary with the full internal state.
        """
        state = {
            'episode': self.episode,
            'possible_agents': self.possible_agents,
            'agents': self.agents,
            'true_value': self.true_value,
            'ask_prices': self._ask_prices,
            'bid_prices': self._bid_prices,
            'min_ask_price': self.min_ask_price,
            'max_bid_price': self.max_bid_price,
            'trader_action': self.trader_action,
            'rewards': np.array(list(self.rewards.values()), dtype=np.float64),
            'cumulative_rewards': np.array(list(self.cumulative_rewards.values()), dtype=np.float64),
            'terminations': np.array(list(self.terminations.values()), dtype=np.bool),
            'truncations': np.array(list(self.truncations.values()), dtype=np.bool)
        }
        return state


    def reset(self, seed: int|None = None, options: Dict|None = None) -> Tuple[Dict, Dict]:
        """
        Reset the environment to the beginning of a new sequence of episodes.

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

        self.episode = 0
        self.true_value = self.generate_vt()

        self.trader = str(self._np_random.choice(self.traders))
        self.agents = self.makers + [self.trader]
        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.next()

        self._ask_prices = np.zeros(self.n_makers, dtype=np.float64)
        self._bid_prices = np.zeros(self.n_makers, dtype=np.float64)
        
        self.min_ask_price = self.high
        self.max_bid_price = self.low
        self.trader_action = None

        self.rewards = {agent: 0 for agent in self.possible_agents}
        self.cumulative_rewards = {agent: 0 for agent in self.possible_agents}
        self._cumulative_rewards = self.cumulative_rewards

        self.observations = {agent: self.observe(agent) for agent in self.possible_agents}
        self.infos = {'episode_finished': False} | {agent: {} for agent in self.possible_agents}

        self.terminations = {agent: not agent in self.agents for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        return self.observations, self.infos


    @_partial_reset
    def step(self, action: 'Dict[str, GMEnv.TraderAction|int|float|np.ndarray]') -> Tuple[Dict, Dict, Dict, Dict, Dict]:
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
            Boolean flags indicating if agents were truncated (episode limit reached).
        infos : dict
            Additional info.

        Raises
        ------
        ValueError
            If the maximum number of episodes has been reached or an invalid action
            has been performed given the current agent.

        Notes
        -----
        For market makers, the action is a dictionary containing 'ask_price' and 'bid_price' 
        values. These values can be passed either as scalar floats (e.g., 0.8) or 
        as NumPy arrays with a single float entry (e.g., np.array([0.8])) — both 
        formats are accepted.
        """
        curr_agent = self.agent_selection
        curr_agent_idx = self.agent_name_mapping[curr_agent]
        
        action, ok = self._assert_and_format_action(curr_agent, action)

        # Update state given current action
        if self.episode >= self.n_episodes:
            raise ValueError('Maximum number of episodes reached')
        if not ok:
            raise ValueError(f'Action `{action}` is not valid for agent {curr_agent}')

        if self._ismaker(curr_agent):
            self._ask_prices[curr_agent_idx] = action['ask_price']
            self._bid_prices[curr_agent_idx] = action['bid_price']

            self.min_ask_price = action['ask_price'] if self.min_ask_price > action['ask_price'] else self.min_ask_price
            self.max_bid_price = action['bid_price'] if self.max_bid_price < action['bid_price'] else self.max_bid_price
        else:
            self.trader_action = action['operation']
        
        # Compute rewards
        if self._agent_selector.is_last():
            rewards = self._compute_rewards()
            for idx, agent in enumerate(self.possible_agents):
                self.rewards[agent] = rewards[idx]
                self.cumulative_rewards[agent] = round(self.cumulative_rewards[agent] + self.rewards[agent], self.decimal_places)

        # Update observations
        # Infos, truncations and terminations will be updated only if the episode ends
        self.observations = {agent: self.observe(agent) for agent in self.possible_agents}

        # Render the environment
        if self.render_mode == 'human':
            self.render()

        # Update for next episode and upadate infos, terminations and truncations
        if self._agent_selector.is_last():
            self.episode += 1
            self._true_value = self.generate_vt()

            self.trader = str(self._np_random.choice(self.traders))
            self.agents = self.makers + [self.trader] if self.episode < self.n_episodes else []

            self.infos = {'episode_finished': True} | {agent: self.inform(agent) for agent in self.possible_agents}
            self.terminations = {agent: not agent in self.agents for agent in self.possible_agents}
            self.truncations = {agent: self.episode >= self.n_episodes for agent in self.possible_agents}

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
            raise ValueError('calling render method without specifying any render mode')

        if self.render_mode == 'human':
            print(self)
        elif self.render_mode == 'ascii':
            return self.__str__()
        return None


    def _assert_and_format_action(
        self,
        agent: str,                          
        action: 'Dict[str, GMEnv.TraderAction|int|float|np.ndarray]'
    ) -> Tuple[Dict[str, TraderAction|float], bool]:
        """
        Validate and format the agent's action.

        It converts action values to proper types and
        checks if they are within the allowed space.

        Parameters
        ----------
        agent : str
            Name of the agent.
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
                action[k] = v.value if isinstance(v, GMEnv.TraderAction) else action[k]
            
            ok = self.action_space(agent).contains(action)

            for k, v in action.items():
                action[k] = round(float(v[0]), self.decimal_places) if isinstance(v, np.ndarray) else v
                action[k] = GMEnv.TraderAction(v) if isinstance(v, int) else action[k]
        except Exception :
            return None, False
        return action, ok
    
    
    def _compute_rewards(
        self,
        trader_action: 'GMEnv.TraderAction|None' = None,
        ask_prices: np.ndarray|None = None,
        bid_prices: np.ndarray|None = None
    ) -> np.ndarray:
        """
        Compute the rewards for the trader and selected market makers based on the action taken.

        Depending on the `trader_action`, the reward is computed either as the difference 
        between the true value and the best available price (min ask or max bid), or zero 
        if the trader passes. The reward is then distributed between the trader and the 
        selected makers who offered the best price.

        If arguments are not provided, internal state is used to fetch the latest
        prices and trader action.

        Parameters
        ----------
        trader_action : GMEnv.TraderAction or None, defualt=None
            The action taken by the trader. If None, the internal `self.trader_action` is used.
        ask_prices : np.ndarray or None, defualt=None
            Array of ask prices. If None, uses `self._ask_prices`.
        bid_prices : np.ndarray or None, defualt=None
            Array of bid prices. If None, uses `self._bid_prices`.

        Returns
        -------
        rewards : np.ndarray
            Array of rewards assigned to each agent (trader and market makers). 
            Shape is (..., len(self.possible_agents)) and squeezed before returning.
        
        Raises
        ------
        ValueError
            If `trader_action` and `self.trader_action` are None.
        """
        if trader_action is None or ask_prices is None or bid_prices is None:
            trader_action = self.trader_action
            ask_prices = self._ask_prices
            bid_prices = self._bid_prices

            min_ask_price = np.array([self.min_ask_price])
            max_bid_price = np.array([self.max_bid_price])
        else:
            min_ask_price = np.min(ask_prices, axis=-1, keepdims=(ask_prices.ndim == 1))
            max_bid_price = np.max(bid_prices, axis=-1, keepdims=(ask_prices.ndim == 1))

        shape = ((1,) if ask_prices.ndim == 1 else ask_prices.shape[:-1]) + (self.max_num_agents,)
        rewards = np.zeros(shape, dtype=np.float64)       

        if trader_action is None:
            raise ValueError('trader_action is None')
        elif trader_action == GMEnv.TraderAction.PASS:
            return rewards    
        elif trader_action == GMEnv.TraderAction.BUY:
            reward = self.true_value - min_ask_price
            selected_makers_idx = np.where(ask_prices == min_ask_price[:, None])
            count_selected_makers = np.sum(ask_prices == min_ask_price[:, None], axis=-1)
        else:
            reward = max_bid_price - self.true_value
            selected_makers_idx = np.where(bid_prices == max_bid_price[:, None])
            count_selected_makers = np.sum(bid_prices == max_bid_price[:, None], axis=-1)

        rewards[:, self.agent_name_mapping[self.trader]] = reward
        rewards[selected_makers_idx[0], selected_makers_idx[1]] = (- reward / count_selected_makers)[selected_makers_idx[0]]
        return np.round(rewards.squeeze(), self.decimal_places)


    def _ismaker(self, agent: str) -> bool:
        """
        Check if the given agent is a maker.

        Parameters
        ----------
        agent : str
            Agent name.

        Returns
        -------
        : bool
            True if the agent is a maker, False otherwise.
        """
        return self.agents_type[self.agent_name_mapping[agent]] == GMEnv.AgentType.MAKER_U or \
               self.agents_type[self.agent_name_mapping[agent]] == GMEnv.AgentType.MAKER_I


    def _istrader(self, agent: str) -> bool:
        """
        Check if the given agent is a trader.

        Parameters
        ----------
        agent : str
            Agent name.

        Returns
        -------
        : bool
            True if the agent is a trader, False otherwise.
        """
        return self.agents_type[self.agent_name_mapping[agent]] == GMEnv.AgentType.TRADER


    def _isinformed(self, agent: str) -> bool:
        """
        Check if the given agent is informed.

        Parameters
        ----------
        agent : str
            Agent name.

        Returns
        -------
        : bool
            True if the agent is informed, False otherwise.
        """
        return self.agents_type[self.agent_name_mapping[agent]] == GMEnv.AgentType.MAKER_I


    def __str__(self) -> str:
        s = f'Episode {self.episode}/{self.n_episodes}:\n' + \
            f' - true value -> {self.true_value}\n' + \
            f' - curr. agent -> {self.agent_selection}\n' + \
            f' - min ask price -> {self.min_ask_price}\n' + \
            f' - max bid price -> {self.max_bid_price}\n' + \
            f' - trader action -> {self.trader_action}\n' + \
            f' - total rewards -> {self.cumulative_rewards}\n'
        return s
