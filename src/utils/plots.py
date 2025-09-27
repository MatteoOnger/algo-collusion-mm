import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from typing import List



DECIMAL_PLACES_AXES = 2
"""
Number of decimal places to display for axis labels.
"""
DECIMAL_PLACES_VALUES = 4
"""
Number of decimal places to display for numeric values.
"""



def plot_cci(
    agents_name: List[str],
    cci: np.ndarray,
    episodes_per_window: int,
    ax: plt.Axes|None = None,
) -> plt.Axes:
    """
    Plot the Calvano Collusion Index (CCI) for multiple agents.

    Each agent's CCI time series is plotted as a separate line. 
    The average CCI across agents is also included as a dashed red line.

    Parameters
    ----------
    agents_name : list of str
        Names of the agents, used as labels in the legend.
    cci : np.ndarray
        Array of shape (n_agents, n_windows) containing the CCI per agent and per window.
    episodes_per_window : int
        Number of episodes grouped in each window.
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

    x = (np.arange(cci.shape[1]) + 1 ) * episodes_per_window

    cmap = plt.get_cmap("tab10")

    for idx, (agent, series) in enumerate(zip(agents_name, cci)):
        ax.plot(x, series, label=agent.capitalize(), color=cmap(idx % 10), marker='*')

    avg_cci = cci.mean(axis=0)
    ax.plot(x, avg_cci, label='Average', color='red', ls='--')

    ax.set_xlim(xmin=0)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Collusion Index')
    ax.set_title(f'Calvano Collusion Index (CCI) per Agent')
    ax.legend()
    ax.grid(True)
    return ax


def plot_maker_heatmap(
    ticksize: float,
    labels: np.ndarray,
    indexes: np.ndarray,
    values: np.ndarray,
    title: str = 'Heatmap',
    ax: plt.Axes|None = None
) -> plt.Axes:
    """
    Plot a heatmap of values over bid/ask price combinations.

    Parameters
    ----------
    ticksize : float
        Tick size used to discretize price indexes.
    labels : np.ndarray
        Array of labels for the x and y axes (price ticks).
    indexes : np.ndarray
        Array of shape (n, 2), with bid and ask indexes for each value.
    values : np.ndarray
        Array of shape (n,) with values corresponding to each (ask, bid) pair.
    title : str, default='Heatmap'
        Title of the plot.
    ax : matplotlib.axes.Axes, default=None
        Axis on which to plot. If None, a new axis is created.

    Returns
    -------
    : matplotlib.axes.Axes
        The axis containing the heatmap.
    """
    matrix = np.full((len(labels), len(labels)), None, dtype=np.float32)
    indexes = (np.round(indexes / ticksize, 0)).astype(int)
    matrix[indexes[:, 1], indexes[:, 0]] = values 

    if ax is None:
        _, ax = plt.subplots()

    sns.heatmap(
        matrix,
        annot = True,
        fmt = f'.{DECIMAL_PLACES_VALUES}f',
        cmap = 'viridis',
        cbar = True,
        xticklabels = labels,
        yticklabels = labels,
        ax = ax
    )
    
    ax.set_xlabel('Ask Price')
    ax.set_ylabel('Bid Price')
    ax.set_title(title)
    return ax


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


def plot_maker_rewards(
    rewards: np.ndarray,
    nash_reward: float|None = None,
    coll_reward: float|None = None,
    agent_name: str = 'Unknown',
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
    agent_name : str, default='Unknown'
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
    agents_name: List[str],
    ax: plt.Axes|None = None
) -> plt.Axes:
    """
    Plot the best price (closest to the true value) for each maker at each episode.
    Bid and ask prices are shown with different markers, and each agent has a different color.

    Parameters
    ----------
    actions : np.ndarray
       Array of shape (n_agents, n_episodes, 2) containing ask and bid prices for each agent and episode.
    true_value : float
        The true value of the asset to compare prices against.
    agents_name : list of str, optional
        Names of the agents for labeling.
    ax : matplotlib.axes.Axes, optional
        Axis on which to plot. If None, a new axis is created.

    Returns
    -------
    : matplotlib.axes.Axes
        The axis containing the plot.
    """
    if ax is None:
        _, ax = plt.subplots()
    
    n_agents = actions.shape[0]
    n_episodes = actions.shape[1]

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
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=n_agents+1)
    ax.grid(True)
    return ax


def plot_makers_comb_actions(
    ticksize: float,
    labels: np.ndarray,
    actions_m1: np.ndarray,
    actions_m2: np.ndarray,
    title: str = 'Heatmap',
    agent_1_name: str = 'First Maker',
    agnet_2_name: str = 'Second Maker',
    ax: plt.Axes|None = None
) -> plt.Axes:
    """
    Plot the joint distribution of actions taken by two market makers.

    Parameters
    ----------
    ticksize : float
        Tick size used to discretize price indexes.
    labels : np.ndarray
        Array of labels for tick values used as axis ticks in the heatmap.
    actions_m1 : np.ndarray
        Array of shape (n_episodes, 2) containing ask and bid prices of the first maker.
    actions_m2 : np.ndarray
        Array of shape (n_episodes, 2) containing ask and bid prices of the second maker.
    title : str, default='Heatmap'
        Title of the plot.
    agent_1_name : str, default='First Maker'
        Name of the first agent for axis labeling.
    agnet_2_name : str, default='Second Maker'
        Name of the second agent for axis labeling.
    ax : matplotlib.axes.Axes, optional
        Axis on which to plot. If None, a new axis is created.

    Returns
    -------
    matplotlib.axes.Axes
        The axis containing the heatmap of action combinations.
    """
    prices = np.concat([actions_m1, actions_m2], axis=1)
    unique_prices, freqs = np.unique(prices, return_counts=True, axis=0)

    indexes = (np.round(unique_prices / ticksize, 0)).astype(int)
    index_m1 = ((indexes[:, 0] * (indexes[:, 0] + 1) / 2) + indexes[:, 1]).astype(int)
    index_m2 = ((indexes[:, 2] * (indexes[:, 2] + 1) / 2) + indexes[:, 3]).astype(int)

    matrix = np.full((len(labels), len(labels)), 0, dtype=np.int32)
    matrix[index_m1, index_m2] = freqs

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

    ax.set_xlabel(agent_1_name.capitalize())
    ax.set_ylabel(agnet_2_name.capitalize())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title(title)
    return
