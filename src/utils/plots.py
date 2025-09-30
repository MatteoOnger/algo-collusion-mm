import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from typing import Dict,List

from ..agents.makers.exp3 import Maker



DECIMAL_PLACES_AXES = 2
"""
Number of decimal places to display for axis labels.
"""
DECIMAL_PLACES_VALUES = 4
"""
Number of decimal places to display for numeric values.
"""



def plot_all(
    window_size: int,
    makers: Dict[str, Maker],
    cci: np.ndarray,
    makers_belif: Dict[str, np.ndarray]|None = None,
    nash_reward: float|None = None,
    coll_reward: float|None = None,
    title: str = 'Makers Summary Plots'
) -> plt.Figure:
    """
    """
    n_makers = len(makers)

    # Determine how many plot rows we need
    rows = 4  # actions, rewards, freq heatmap, CCI
    if makers_belif is not None:
        rows += 1  # Add Q-table row
    if n_makers == 2:
        rows += 1  # Add combined action plots

    fig = plt.figure(figsize=(6 * n_makers, 4 * rows), constrained_layout=True)
    fig.suptitle(title, fontsize=20)

    gs = gridspec.GridSpec(rows, n_makers, figure=fig)

    # Per-agent plots
    for i, maker in enumerate(makers.keys()):
        agent = makers[maker]

        # Row 0: Action history
        ax_actions = fig.add_subplot(gs[0, i])
        plot_maker_actions(
            agent.history.get_actions(),
            agent_name = maker,
            ax = ax_actions
        )

        # Row 1: Rewards
        ax_rewards = fig.add_subplot(gs[1, i])
        plot_maker_rewards(
            agent.history.get_rewards(),
            nash_reward = nash_reward / n_makers,
            coll_reward = coll_reward / n_makers,
            agent_name = maker,
            ax = ax_rewards
        )

        # Row 2: Frequency heatmap
        actions, freqs = agent.history.compute_freqs()

        ax_freq = fig.add_subplot(gs[2, i])
        plot_maker_heatmap(
            indexes = agent.price_to_index(actions),
            labels = agent.prices,
            values = freqs,
            title = f'Actions Absolute Frequency',
            agent_name = maker,
            ax = ax_freq
        )
        ax_freq.set_aspect('equal', adjustable='box')

        # Row 3 (optional): Q-table heatmap
        if makers_belif is not None:
            ax_q = fig.add_subplot(gs[3, i])
            plot_maker_heatmap(
                indexes = agent.price_to_index(agent.action_space),
                labels = agent.prices,
                values = makers_belif[maker],
                title = f'Final Q-Table',
                agent_name = maker,
                ax = ax_q
            )
            ax_q.set_aspect('equal', adjustable='box')

    # CCI row is always last row before combined actions (if present)
    cci_row = 4 if makers_belif is not None else 3
    ax_cci = fig.add_subplot(gs[cci_row, :])
    plot_makers_cci(window_size, cci, makers.keys(), ax=ax_cci)

    # Optional: combined action plots (only if 2 agents)
    if n_makers == 2:
        comb_row = cci_row + 1
        maker1, maker2 = list(makers.keys())[:2]
        agent1, agent2 = list(makers.values())[:2]

        # First window (initial steps)
        ax_comb_1 = fig.add_subplot(gs[comb_row, 0])
        plot_makers_comb_actions(
            labels = agent1.action_space,
            actions_idx_m1 = agent1.action_to_index(agent1.history.get_actions(slice(window_size)))[:, None],
            actions_idx_m2 = agent2.action_to_index(agent2.history.get_actions(slice(window_size)))[:, None],
            title = 'Makers Actions - First Window',
            agent_1_name = maker1,
            agent_2_name = maker2,
            ax = ax_comb_1
        )
        ax_comb_1.set_aspect('equal', adjustable='box')

        # Second window (final steps)
        ax_comb_2 = fig.add_subplot(gs[comb_row, 1])
        plot_makers_comb_actions(
            labels = agent1.action_space,
            actions_idx_m1 = agent1.action_to_index(agent1.history.get_actions(slice(-window_size, None)))[:, None],
            actions_idx_m2 = agent2.action_to_index(agent2.history.get_actions(slice(-window_size, None)))[:, None],
            title = 'Makers Actions - Last Window',
            agent_1_name = maker1,
            agent_2_name = maker2,
            ax = ax_comb_2
        )
        ax_comb_2.set_aspect('equal', adjustable='box')

    plt.close()
    return fig


def plot_maker_actions(
    actions: np.ndarray,
    agent_name: str = 'Unknown',
    ax: plt.Axes|None = None
) -> plt.Axes:
    """
    Plot the evolution of ask and bid prices for a single maker.

    Parameters
    ----------
    actions : np.ndarray
        Array of shape (n_episodes, 2) where column 0 is ask price and column 1 is bid price.
    agent_name : str, default='Unknown'
        Name of the agent for labeling the plot.
    ax : matplotlib.axes.Axes, optional
        Axis on which to plot. If None, a new axis is created.

    Returns
    -------
    : matplotlib.axes.Axes
        The axis containing the plot.
    """
    if ax is None:
        _, ax = plt.subplots()

    x = np.arange(len(actions))

    ax.plot(x, actions[:, 0], label='Ask', color='red', alpha=0.6, marker='*')
    ax.plot(x, actions[:, 1], label='Bid', color='green', alpha=0.5, marker='*')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Price')
    ax.set_title(f'Ask/Bid Price Trend - {agent_name.capitalize()}')
    ax.legend()
    ax.grid(True)
    return ax


def plot_maker_heatmap(
    indexes: np.ndarray,
    labels: np.ndarray,
    values: np.ndarray,
    title: str = 'Heatmap',
    agent_name: str = 'unknown',
    ax: plt.Axes|None = None
) -> plt.Axes:
    """
    Plots a heatmap of values corresponding to bid/ask price combinations.

    This function visualizes a set of values on a 2D grid using a heatmap, where each cell 
    represents a specific bid/ask price pair. The values are placed into a matrix according 
    to their corresponding (x, y) index positions.

    Parameters
    ----------
    indexes : np.ndarray
        Array of index pairs indicating the (x, y) positions of each value in the heatmap matrix.
    labels : np.ndarray
        Array of strings or numbers (prices) used as tick labels for both the x and y axes.
    values : np.ndarray
        Array of values to populate in the heatmap, corresponding to each index in `indexes`.
    title : str, default='Heatmap'
        Title of the plot.
    agent_name : str, default='unknown'
        Name of the agent to include in the plot title.
    ax : matplotlib.axes.Axes or None, optional
        The matplotlib axis on which to draw the heatmap. If None, a new axis is created.

    Returns
    -------
    : matplotlib.axes.Axes
        The axis object containing the generated heatmap.
    """
    is_int = np.all(np.mod(values, 1) == 0)

    matrix = np.full((len(labels), len(labels)), None, dtype=np.float32)
    matrix[indexes[:, 1], indexes[:, 0]] = values 

    if ax is None:
        _, ax = plt.subplots()

    sns.heatmap(
        matrix,
        annot = True,
        fmt = f'.{0 if is_int else DECIMAL_PLACES_VALUES}f',
        cmap = 'viridis',
        cbar = True,
        xticklabels = labels,
        yticklabels = labels,
        ax = ax
    )
    
    ax.set_xlabel('Ask Price')
    ax.set_ylabel('Bid Price')
    ax.set_title(title + ' - ' + agent_name.capitalize())
    return ax


def plot_maker_rewards(
    rewards: np.ndarray,
    nash_reward: float|None = None,
    coll_reward: float|None = None,
    agent_name: str = 'unknown',
    ax: plt.Axes|None = None
) -> plt.Axes:
    """
    Plot the reward trend of a single maker across episodes.

    Optionally includes constant reference lines for Nash equilibrium
    reward and collusion reward.

    Parameters
    ----------
    rewards : np.ndarray
        Array of shape (n_episodes,) with reward values per episode.
    nash_reward : float, default=None
        Reference value for the Nash equilibrium reward. Default is None.
    coll_reward : float, default=None
        Reference value for the collusion reward. Default is None.
    agent_name : str, default='unknown'
        Name of the agent for labeling the plot.
    ax : matplotlib.axes.Axes, default=None
        Axis on which to plot. If None, a new axis is created.

    Returns
    -------
    : matplotlib.axes.Axes
        The axis containing the plot.
    """
    if ax is None:
        _, ax = plt.subplots()

    x = np.arange(len(rewards))

    cmap = plt.get_cmap("tab10")
    color = cmap(int(agent_name.split("_")[-1]) % 10)

    ax.plot(x, rewards, label=agent_name.capitalize(), color=color, alpha=1.0, marker='*')
    if nash_reward is not None:
        ax.axhline(nash_reward, label='Nash', color='blue', ls='--')
    if coll_reward is not None:
        ax.axhline(coll_reward, label='Collusion', color='red', ls='--')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title(f'Rewards Trend - {agent_name.capitalize()}')
    ax.legend()
    ax.grid(True)
    return ax


def plot_makers_best_actions(
    true_value: float,
    actions: np.ndarray,
    agents_name: List[str]|None = None,
    ax: plt.Axes|None = None
) -> plt.Axes:
    """
    Plot the best price (closest to the true value) for each maker at each episode.
    Bid and ask prices are shown with different markers, and each agent has a different color.

    Parameters
    ----------
    true_value : float
        The true value of the asset to compare prices against.
    actions : np.ndarray
       Array of shape (n_agents, n_episodes, 2) containing ask and bid prices for each agent and episode.
    agents_name : list of str or None, default=None
        Names of the agents for labeling.
    ax : matplotlib.axes.Axes, default=None
        Axis on which to plot. If None, a new axis is created.

    Returns
    -------
    : matplotlib.axes.Axes
        The axis containing the plot.
    """
    n_agents = actions.shape[0]
    n_episodes = actions.shape[1]

    if ax is None:
        _, ax = plt.subplots()
    if agents_name is None:
        agents_name = [f'unknown_{i}' for i in range(n_agents)]

    x = np.arange(n_episodes)

    cmap = plt.get_cmap("tab10")
    markers = {'ask': 'o', 'bid': '*'}

    for idx, (action, name) in enumerate(zip(actions, agents_name)):
        color = cmap(idx % 10)

        ask_prices = action[:, 0]
        bid_prices = action[:, 1]

        best_is_ask = (true_value - ask_prices) >= (bid_prices - true_value)
        best_is_bid = (true_value - ask_prices) <= (bid_prices - true_value)

        ax.scatter(x[best_is_ask], ask_prices[best_is_ask], label=f'{name.capitalize()} - Ask', color=color, marker=markers['ask'], alpha=1.0)
        ax.scatter(x[best_is_bid], bid_prices[best_is_bid], label=f'{name.capitalize()} - Bid', color=color, marker=markers['bid'], alpha=1.0)

    min_ask_prices = np.min(actions[:, :, 0], axis=0)
    max_bid_prices = np.max(actions[:, :, 1], axis=0)

    best_is_ask = (true_value - min_ask_prices) >= (max_bid_prices - true_value)
    best_is_bid = (true_value - min_ask_prices) <= (max_bid_prices - true_value)

    ax.scatter(x[best_is_ask], min_ask_prices[best_is_ask], s=200, facecolors='none', edgecolors='red')
    ax.scatter(x[best_is_bid], max_bid_prices[best_is_bid], s=200, facecolors='none', edgecolors='red')

    ax.axhline(true_value, color='black', ls=':', label='True Value')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Price')
    ax.set_title('Best Price per Maker vs True Value')
    ax.legend(loc='upper center', ncol=n_agents+1)
    ax.grid(True)
    return ax


def plot_makers_cci(
    episodes_per_window: int,
    cci: np.ndarray,
    agents_name: List[str]|None = None,
    ax: plt.Axes|None = None,
) -> plt.Axes:
    """
    Plot the Calvano Collusion Index (CCI) for multiple makers.

    Each agent's CCI time series is plotted as a separate line. 
    The average CCI across agents is also included as a dashed red line.

    Parameters
    ----------
    episodes_per_window : int
        Number of episodes grouped in each window.
    cci : np.ndarray
        Array of shape (n_agents, n_windows) containing the CCI per agent and per window.
    agents_name : list of str or None
        Names of the agents, used as labels in the legend.
    ax : matplotlib.axes.Axes, default=None
        Axis on which to plot. If None, a new axis is created.

    Returns
    -------
    : matplotlib.axes.Axes
        The axis containing the plot.
    
    See Also
    --------
    - Calvano, E., Calzolari, G., Denicolò, V., & Pastorello, S. (2020).  
    Artificial intelligence, algorithmic pricing, and collusion.  
    *American Economic Review, 110*(10), 3267–3297.  
    https://doi.org/10.1257/aer.20190623
    """
    if ax is None:
        _, ax = plt.subplots()
    if agents_name is None:
        agents_name = [f'unknown_{i}' for i in range(cci.size)]

    x = (np.arange(cci.shape[1]) + 1 ) * episodes_per_window

    cmap = plt.get_cmap("tab10")

    for idx, (agent, series) in enumerate(zip(agents_name, cci)):
        ax.plot(x, series, label=agent.capitalize(), color=cmap(idx % 10), marker='*')
    
    ax.plot(x, cci.mean(axis=0), label='Average', color='red', ls='--')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Collusion Index')
    ax.set_title(f'Calvano Collusion Index (CCI) per Agent')
    ax.legend()
    ax.grid(True)
    return ax


def plot_makers_comb_actions(
    labels: np.ndarray,
    actions_idx_m1: np.ndarray,
    actions_idx_m2: np.ndarray,
    title: str = 'Heatmap',
    agent_1_name: str = 'First Maker',
    agent_2_name: str = 'Second Maker',
    ax: plt.Axes|None = None
) -> plt.Axes:
    """
    Plot the joint distribution of actions taken by two market makers.

    Parameters
    ----------
    labels : np.ndarray
        Array of strings or numbers used as tick labels for both the x and y axes.
    actions_idx_m1 : np.ndarray
        Array of shape (n_episodes, 2) containing ask and bid prices of the first maker.
    actions_idx_m2 : np.ndarray
        Array of shape (n_episodes, 2) containing ask and bid prices of the second maker.
    title : str, default='Heatmap'
        Title of the plot.
    agent_1_name : str, default='First Maker'
        Name of the first agent for axis labeling.
    agent_2_name : str, default='Second Maker'
        Name of the second agent for axis labeling.
    ax : matplotlib.axes.Axes, optional
        Axis on which to plot. If None, a new axis is created.

    Returns
    -------
    : matplotlib.axes.Axes
        The axis containing the heatmap of action combinations.
    """
    actions_comb = np.concat([actions_idx_m1, actions_idx_m2], axis=1)
    unique_actions_comb, freqs = np.unique(actions_comb, return_counts=True, axis=0)

    matrix = np.full((len(labels), len(labels)), 0, dtype=np.int32)
    matrix[unique_actions_comb[:, 1], unique_actions_comb[:, 0]] = freqs

    if ax is None:
        _, ax = plt.subplots()

    sns.heatmap(
        matrix,
        annot = True,
        fmt = 'd',
        cmap = 'viridis',
        cbar = True,
        xticklabels = labels,
        yticklabels = labels,
        ax = ax
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_xlabel(agent_1_name.capitalize())
    ax.set_ylabel(agent_2_name.capitalize())
    ax.set_title(title)
    return
