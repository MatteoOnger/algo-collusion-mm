import gym
import numpy as np
import pettingzoo as ptz

from copy import deepcopy
from enum import Enum
from gym.spaces import Box, Discrete
from pettingzoo.utils import agent_selector as AgentSelector
from typing import Callable, Dict, Literal, Tuple



class GMEnv(ptz.AECEnv):
    """
    Glosten-Milgrom (GM) environment for simulating a single-asset dealer market.

    This environment is inspired by the microstructure model introduced
    by Glosten and Milgrom (1985), which analyzes how bid and ask quotes are set
    in the presence of asymmetric information.

    In this version, multiple market makers quote bid and ask prices for a single asset.
    Traders may choose to BUY, SELL, or PASS. For each trade, if they decide to transact,
    the trader selects the dealer offering the most advantageous price given
    their private signal about the asset's true value.

    Parameters
    ----------
    generate_vt : Callable[[], float]
        Function that returns the true value for the asset at the beginning of each episode.
    n_episodes : int
        Total number of episodes to simulate.
    n_makers : int
        Number of market makers.
    n_traders : int
        Number of traders.
    low : float, default=0.0
        Minimum possible price or asset value.
    high : float, default=1.0
        Maximum possible price or asset value.
    render_mode : {'ascii', 'human'}, default='ascii'
        Mode for rendering the environment.
    
    Attributes
    ----------      
    makers : list of str
        List of maker agent names.
    traders : list of str
        List of trader agent names.
    possible_agents : list of str
        Full list of all agents in the environment (makers + traders).
    agent_name_mapping : dict
        Mapping from agent name to index.
    agents_type : list of GM2Env.AgentType
        Agent type for each agent (MAKER or TRADER).
    episode : int
        Current episode number (incremented after each trader action).
    agents : list of str
        Agents currently active in the environment (all makers + one trader).
    agent_selection : str
        Name of the agent whose turn it is to act.
    rewards : dict
        Latest rewards assigned to each agent after a step.
    cumulative_rewards : dict
        Sum of the rewards assigned to each agent.
    observations : dict
        Latest observations available to each agent.
    infos : dict
        Additional information returned per agent (empty by default).
    terminations : dict
        Flags indicating whether an agent has terminated.
    truncations : dict
        Flags indicating whether an agent was truncated (due to episode limit).
    true_value : float
        Holds the true underlying value of the asset for the current episode.
    min_ask_price : float
        Lowest ask price among all makers in the current step.
    max_bid_price : float
        Highest bid price among all makers in the current step.
    trader_action : GM2Env.TraderAction or None
        The last action taken by the trader (BUY or SELL).

    See Also
    --------
    - Glosten, L. R., & Milgrom, P. R. (1985). Bid, ask and transaction prices
    in a specialist market with heterogeneously informed traders. 
    Journal of Financial Economics, 14(1), 71–100.
    https://doi.org/10.1016/0304-405X(85)90044-3
    """

    metadata = {
        'name': 'Glosten-Milgrom_environment',
        'render_modes': ['ascii', 'human']
    }


    class AgentType(Enum):
        """
        Agent types in the GM environment.
        """
        MAKER = 0
        TRADER = 1


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
        n_makers: int,
        n_traders: int,
        low: float = 0.0,
        high: float = 1.0,
        render_mode: Literal['ascii', 'human'] = 'ascii'
    ):
        """
        Parameters
        ----------
        See class-level docstring for full parameter descriptions.
            
        Raises
        ------
        ValueError
            If the number of makers or traders is smaller than one.
        """
        super().__init__()

        if n_makers < 1:
            raise ValueError('n_makers < 1')
        if n_traders < 1:
            raise ValueError('n_traders < 1')

        self.generate_vt = generate_vt
        self.n_episodes = n_episodes
        self.n_makers = n_makers
        self.n_traders = n_traders
        self.low = low
        self.high = high
        self.render_mode = render_mode

        self.makers = [f'maker_{idx}' for idx in range(n_makers)]
        self.traders = [f'trader_{idx}' for idx in range(n_traders)]

        self.possible_agents = self.makers + self.traders
        self.agent_name_mapping = {agent:idx for idx, agent in enumerate(self.possible_agents)}
        self.agents_type = [GMEnv.AgentType.MAKER] * n_makers + [GMEnv.AgentType.TRADER] * n_traders
        return


    def action_space(self, agent: str, seed: int|None = None) -> gym.Space:
        """
        Returns the action space for a given agent.

        Makers choose ask and bid prices. Traders choose whether to buy, sell or pass.

        Parameters
        ----------
        agent : str
            Name of the agent.
        seed : int, optional
            Random seed. Default is None, so a random seed will be used.

        Returns
        -------
        : gym.Space
            The agent's action space.
        """
        if self._ismaker(agent):
            space = gym.spaces.Dict({
                'ask_price': Box(self.low, self.high, dtype=np.float64, seed=seed),
                'bid_price': Box(self.low, self.high, dtype=np.float64, seed=seed)
            })
        else:
            space = gym.spaces.Dict({
                'operation': Discrete(len(GMEnv.TraderAction), seed=seed)
            })
        return space


    def observation_space(self, agent: str, seed: int|None = None) -> gym.Space:
        """
        Returns the observation space for a given agent.

        Traders observe the true value, the minimum ask, and the maximum bid. 
        Makers receive an empty observation.

        Parameters
        ----------
        agent : str
            Name of the agent.
        seed : int, optional
            Random seed. Default is None, so a random seed will be used.

        Returns
        -------
        : gym.Space
            The agent's observation space.
        """
        if self._ismaker(agent):
            space = gym.spaces.Dict({})
        else:
            space = Dict({
                'true_value': Box(self.low, self.high, dtype=np.float64, seed=seed),
                'min_ask_price': Box(self.low, self.high, dtype=np.float64, seed=seed),
                'max_bid_price': Box(self.low, self.high, dtype=np.float64, seed=seed)
            })
        return space


    def observe(self, agent: str) -> Dict:
        """
        Returns the current observation for the given agent.

        Parameters
        ----------
        agent : str
            Agent to observe.

        Returns
        -------
        : dict
            The observation dictionary for the agent.
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


    def state(self) -> Dict:
        """
        Returns the global state of the environment.

        Includes current prices, episode count, rewards, and agent status.

        Returns
        -------
        : dict
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
        Resets the environment to the beginning of a new sequence of episodes.

        Randomly selects one trader and initializes maker prices and true value.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.
            Default is None, so a random seed wil be used.
        options : dict, optional
            Additional options (not used).

        Returns
        -------
        observations : dict
            Initial observations for all agents.
        infos : dict
            Additional information for each agent (empty).
        """
        self._np_random, self._np_random_seed = gym.utils.seeding.np_random(seed)

        self.episode = 0
        self.true_value = self.generate_vt()

        self.agents = self.makers + [str(self._np_random.choice(self.traders))]
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
        self.infos = {agent: {} for agent in self.possible_agents}

        self.terminations = {agent: not agent in self.agents for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        return self.observations, self.infos


    def step(self, action: 'Dict[str, GMEnv.TraderAction|int|float|np.ndarray]') -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        Executes a step in the environment with the provided action.

        Each maker sets prices; the trader decides to buy, sell or pass.
        Rewards are computed based on the resulting transaction and the true value.

        Parameters
        ----------
        action : dict
            Dictionary containing the action for the current agent.

        Returns
        -------
        observations : dict
            New observations after the step.
        rewards : dict
            Rewards for each agent.
        terminations : dict
            Boolean flags indicating if agents have terminated.
        truncations : dict
            Boolean flags indicating if agents were truncated (episode limit reached).
        infos : dict
            Additional info (empty).

        Raises
        ------
        ValueError
            If the maximum number of episodes has been reached or an invalid action
            has been performed given the current agent.

        Notes
        -----
        For market makers, the action is a dictionary containing 'ask' and 'bid' 
        values. These values can be passed either as scalar floats (e.g., 0.8) or 
        as NumPy arrays with a single float entry (e.g., np.array([0.8])) — both 
        formats are accepted.
        """
        curr_agent = self.agent_selection
        curr_agent_idx = self.agent_name_mapping[curr_agent]
        
        action, ok = self._assert_and_format_action(curr_agent, action)

        # Update state given current action
        if self.episode >= self.n_episodes:
            raise ValueError('maximum number of episodes reached')
        if not ok:
            raise ValueError(f'invalid action for agent {curr_agent}')

        if self._ismaker(curr_agent):
            self._ask_prices[curr_agent_idx] = action['ask_price']
            self._bid_prices[curr_agent_idx] = action['bid_price']

            self.min_ask_price = action['ask_price'] if self.min_ask_price > action['ask_price'] else self.min_ask_price
            self.max_bid_price = action['bid_price'] if self.max_bid_price < action['bid_price'] else self.max_bid_price
        else:
            self.trader_action = action['operation']
        
        # Compute rewards
        if self._agent_selector.is_last() and self.trader_action != GMEnv.TraderAction.PASS:
            if self.trader_action == GMEnv.TraderAction.BUY:
                reward = self.true_value - self.min_ask_price
                selected_makers_idx = np.where(self._ask_prices == self.min_ask_price)[0]
            elif self.trader_action == GMEnv.TraderAction.SELL:
                reward = self.max_bid_price - self.true_value
                selected_makers_idx = np.where(self._bid_prices == self.max_bid_price)[0]
            
            for idx, agent in enumerate(self.possible_agents):
                if self._istrader(agent) and agent in self.agents:
                    self.rewards[agent] = reward
                elif self._ismaker(agent) and idx in selected_makers_idx:
                    self.rewards[agent] = - reward / len(selected_makers_idx)
                else:
                    self.rewards[agent] = 0
                self.cumulative_rewards[agent] += self.rewards[agent]
        else:
            self.rewards = {agent: 0 for agent in self.possible_agents}
            
        observations = {agent: self.observe(agent) for agent in self.possible_agents}
        infos = {agent: {} for agent in self.possible_agents}
        rewards = deepcopy(self.rewards)
        terminations = deepcopy(self.terminations)
        truncations = deepcopy(self.truncations)

        # Render the environment
        if self.render_mode == 'human':
            self.render()

        # Perform a partial reset at the end of an episode
        if self._agent_selector.is_last():
            self.episode += 1
            #self._true_value = self.val_gen()

            self.agents = self.makers + [str(self._np_random.choice(self.traders))] if self.episode < self.n_episodes else []

            self._ask_prices = np.zeros(self.n_makers, dtype=np.float64)
            self._bid_prices = np.zeros(self.n_makers, dtype=np.float64)
            
            self.min_ask_price = self.high
            self.max_bid_price = self.low
            self.trader_action = None
            
            self.rewards = {agent: 0 for agent in self.possible_agents}

            self.observations = {agent: self.observe(agent) for agent in self.possible_agents}
            self.infos = {agent: {} for agent in self.possible_agents}

            self.terminations = {agent: not agent in self.agents for agent in self.possible_agents}
            self.truncations = {agent: self.episode >= self.n_episodes for agent in self.possible_agents}
        self.agent_selection = self._agent_selector.next()
        return observations, rewards, terminations, truncations, infos


    def render(self) -> str|None:
        """
        Renders the environment state according to the selected mode.

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
        Validates and formats the agent's action.

        Converts action values to proper types and checks if they are within the 
        allowed space.

        Parameters
        ----------
        agent : str
            Name of the agent.
        action : dict
            Raw action provided.

        Returns
        -------
        formatted_action : dict
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
                action[k] =float(v[0]) if isinstance(v, np.ndarray) else v
                action[k] = GMEnv.TraderAction(v) if isinstance(v, int) else action[k]
        except Exception :
            return None, False
        return action, ok


    def _ismaker(self, agent: str) -> bool:
        """
        Checks if the given agent is a maker.

        Parameters
        ----------
        agent : str
            Agent name.

        Returns
        -------
        : bool
            True if the agent is a maker, False otherwise.
        """
        return self.agents_type[self.agent_name_mapping[agent]] == GMEnv.AgentType.MAKER


    def _istrader(self, agent: str) -> bool:
        """
        Checks if the given agent is a trader.

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


    def __str__(self) -> str:
        s = f'Episode {self.episode}/{self.n_episodes}:\n' + \
            f' - true value -> {self.true_value}\n' + \
            f' - curr. agent -> {self.agent_selection}\n' + \
            f' - min ask price -> {self.min_ask_price}\n' + \
            f' - max bid price -> {self.max_bid_price}\n' + \
            f' - trader action -> {self.trader_action}\n' + \
            f' - total rewards -> {self._cumulative_rewards}\n'
        return s
