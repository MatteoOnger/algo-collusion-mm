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
        agents: List[str],
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
    agents : list of str
        Names of the agents, used as labels in the legend.
    cci : np.ndarray
        Array of shape (n_agents, n_windows) containing the CCI per agent and per window.
    episodes_per_window : int
        Number of episodes grouped in each window on the x-axis.
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
    colors = [cmap(i % 10) for i in range(cci.shape[0])]

    for agent, series, color in zip(agents, cci, colors):
        ax.plot(x, series, label=agent.capitalize(), color=color, marker='*')

    avg_cci = cci.mean(axis=0)
    ax.plot(x, avg_cci, label='Average', color='red', ls='--')

    ax.set_xlim(xmin=0)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Collusion Index')
    ax.set_title(f'Calvano Collusion Index (CCI) per Agent')
    ax.legend()
    ax.grid(True)
    return ax


def plot_heatmap(
        ticksize: float,
        labels: np.ndarray,
        indexes: np.ndarray,
        values: np.ndarray,
        title: str = 'Heatmap',
        xlabel: str = 'Ask price',
        ylabel: str = 'Bid price',
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
    xlabel : str, default='Ask price'
        Label for the x-axis.
    ylabel : str, default='Bid price'
        Label for the y-axis.
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
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return ax


def plot_maker_prices(
        prices: np.ndarray,
        agent_name: str = 'Unknown',
        ax: plt.Axes|None = None
    ) -> plt.Axes:
    """
    Plot the evolution of ask and bid prices for a single maker.

    Parameters
    ----------
    prices : np.ndarray
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

    x = np.arange(len(prices))

    ax.plot(x, prices[:, 0], label='Ask', color='red', alpha=0.6, marker='*')
    ax.plot(x, prices[:, 1], label='Bid', color='green', alpha=0.5, marker='*')

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
