""" Functions for creating and customizing charts to visualize the results of tests.
"""
import ipywidgets as widgets
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import re
import seaborn as sns

from IPython.display import display
from typing import List, Literal, Tuple

from ..agents.makers.maker import Maker
from ..utils.stats import OnlineVectorStats



DECIMAL_PLACES_AXES = 2
"""
Number of decimal places to display for axis labels.
"""
DECIMAL_PLACES_VALUES = 3
"""
Number of decimal places to display for numeric values.
"""
JITTER_STRENGTH = 0.
"""
Controls the amount of jitter applied.
"""



def plot_all(
    window_size: int,
    nash_reward: float,
    coll_reward: float,
    makers: List[Maker],
    makers_belief_name: List[str]|str,
    cci: np.ndarray,
    annot: Literal['all', 'none', 'above_mean'] = 'above_mean',
    title: str = 'Makers Summary Plot'
) -> plt.Figure:
    """
    Create a comprehensive summary of multiple maker agents' behavior over time.

    This figure summarizes multiple aspects of the makers' behavior during training or simulation, including:
    - **Action histories** - plots showing the sequence of actions taken by each maker over time.
    - **Individual reward time series** - per-agent reward progression across rounds or windows.
    - **Absolute action frequency heatmaps** - visualizations of how often each action was chosen by each maker.
    - **Final agents' beliefs** - representation of each maker's belief at the end of the simulation.
    - **Calvano Collusion Index (CCI) over time** - tracks the level of collusion or cooperation between agents.
    - **Joint action heatmaps** - shows joint action distributions in the first and last windows.

    Parameters
    ----------
    window_size : int
        Number of rounds per window.
    nash_reward : float
        Reference reward level for Nash equilibrium, shown as a horizontal line in reward plots.
    coll_reward : float
        Reference reward level for full collusion, shown as a horizontal line in reward plots.
    makers : List[Maker]
        List of `Maker` objects representing the agents.
    makers_belief_name : List[str] or str
        List of names of the maker variable representing the belief of each agent.
    cci : np.ndarray
        Array of shape (n_agents, n_windows) containing the CCI values per agent.
    annot : {'all', 'none', 'above_mean'}, default='above_mean'
        Controls whether and how to annotate heatmap cells:
        - `'all'` — annotate all cells.
        - `'none'` — no annotations.
        - `'above_mean'` — annotate only cells with values above the matrix mean.
    title : str, default='Makers Summary Plot'
        Title of the generated figure.

    Returns
    -------
    : matplotlib.figure.Figure
        The figure containing all generated subplots for makers' actions, rewards, frequencies,
        beliefs, CCI and joint action heatmaps.

    Notes
    -----
    - The figure is closed (`plt.close()`) before being returned to prevent duplicate display in notebooks.
    """
    n_makers = len(makers)
    
    if isinstance(makers_belief_name, str):
        makers_belief_name = n_makers * [makers_belief_name]

    # Determine how many plot rows we need
    n_rows = 6  # actions, rewards, freq heatmap, CCI, belief, joint actions

    fig = plt.figure(figsize=(8 * n_makers, 6 * n_rows)) 
    fig.suptitle(title, fontsize=20)

    gs = gridspec.GridSpec(n_rows, n_makers, figure=fig)

    # Per-agent plots
    for i, maker in enumerate(makers):
        # Row 0: action history
        ax_actions = fig.add_subplot(gs[0, i])
        plot_maker_actions(
            maker = maker,
            ax = ax_actions
        )

        # Row 1: rewards
        ax_rewards = fig.add_subplot(gs[1, i])
        plot_maker_rewards(
            maker = maker,
            nash_reward = nash_reward / n_makers,
            coll_reward = coll_reward / n_makers,
            ax = ax_rewards
        )

        # Row 2: action frequency
        ax_freq = fig.add_subplot(gs[2, i])
        plot_maker_actions_freq(
            maker = maker,
            annot = annot,
            title = 'Global Actions Frequency',
            ax = ax_freq
        )
        ax_freq.set_aspect('equal', adjustable='box')

        # Row 3: belief heatmap
        ax_belief = fig.add_subplot(gs[3, i])
        plot_maker_belief(
            maker = maker,
            belief_name = makers_belief_name[i],
            annot = annot,
            title = 'Final Belief',
            ax = ax_belief
        )
        ax_belief.set_aspect('equal', adjustable='box')

    # Row 4: collusion index
    ax_cci = fig.add_subplot(gs[4, :])
    plot_makers_cci(
        xlabel = 'Round',
        x = window_size * np.arange(cci.shape[1]),
        cci = cci,
        makers_name = [maker.name for maker in makers],
        ax = ax_cci
    )

    # Joint actions frequency
    sub_gs = gridspec.GridSpecFromSubplotSpec(
        nrows = 1, 
        ncols = 2,
        subplot_spec = gs[5, :]
    )
    ax_comb_1 = fig.add_subplot(sub_gs[0, 0])
    ax_comb_2 = fig.add_subplot(sub_gs[0, 1])

    if n_makers == 2:
        # First window
        plot_makers_joint_actions_freq_hm(
            makers = makers,
            round_range = slice(window_size),
            annot = annot,
            title = 'Relative Joint Actions Frequency - First Window',
            ax = ax_comb_1
        )
        ax_comb_1.set_aspect('equal', adjustable='box')

        # Last window
        plot_makers_joint_actions_freq_hm(
            makers = makers,
            round_range = slice(-window_size, None),
            annot = annot,
            title = 'Relative Joint Actions Frequency - Last Window',
            ax = ax_comb_2
        )
        ax_comb_2.set_aspect('equal', adjustable='box')
    else:
        # First window
        plot_makers_joint_actions_freq_pc(
            makers = makers,
            round_range = slice(window_size),
            threshold = 'above_mean_1_std',
            title = 'Relative Joint Actions Frequency - First Window',
            pad = 0.09,
            ax = ax_comb_1
        )

        # Last window
        plot_makers_joint_actions_freq_pc(
            makers = makers,
            round_range = slice(-window_size, None),
            threshold = 'above_mean_1_std',
            title = 'Relative Joint Actions Frequency - Last Window',
            location = 'left',
            pad = 0.09,
            ax = ax_comb_2
        )
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    plt.close()
    return fig


def plot_all_stats(
    window_size: int,
    makers: List[Maker],
    stats_cci: OnlineVectorStats|None = None,
    stats_sorted_cci: OnlineVectorStats|None = None,
    stats_actions_freq: OnlineVectorStats|None = None,
    stats_joint_actions_freq: OnlineVectorStats|None = None,
    stats_belief: OnlineVectorStats|None = None,
    annot: Literal['all', 'none', 'above_mean'] = 'above_mean',
    title: str = 'Makers Statistics Summary Plot',
) -> plt.Figure:
    """
    Create a comprehensive summary figure of statistical measures for multiple maker agents.

    This function compiles several plots summarizing the evolution and variability of 
    market-making agents' behavior across simulation windows. It visualizes measures 
    such as the Calvano Collusion Index (CCI), per-maker action frequencies, joint 
    action frequencies, and belief matrices.

    Depending on the provided statistical trackers, the figure may include up to 
    five main sections (stacked vertically):

    1. **Mean Calvano Collusion Index (CCI)** — shows per-maker mean, standard deviation,
       minimum, and maximum CCI values over time.
    2. **Mean Sorted CCI** — displays the same statistics after sorting makers by CCI.
    3. **Mean Relative Action Frequencies** — one heatmap per maker, showing average 
       action frequency matrices for the first and last simulation windows.
    4. **Mean Joint Action Frequencies** — shown the joint action distributions
       for the first and last windows.
    5. **Mean Final Belief Matrices** — optional section showing each maker's belief matrix 
       mean and standard deviation across windows.

    Parameters
    ----------
    window_size : int
        Number of rounds per averaging window; defines the x-axis scale for time-based plots.
    makers : list of Maker
        List of maker agents whose statistics are being summarized.
    stats_cci : OnlineVectorStats or None, default=None
        Online statistics tracker for Calvano Collusion Index (CCI) values per maker 
        over time. Generates a time-series subplot if provided.
    stats_sorted_cci : OnlineVectorStats or None, default=None
        Online statistics tracker for CCI values after sorting makers by their CCI.
        Generates an additional subplot if provided.
    stats_actions_freq : OnlineVectorStats or None, default=None
        Online tracker for relative action frequency matrices per maker. 
        Generates two rows of heatmaps (first and last window means) if provided.
    stats_joint_actions_freq : OnlineVectorStats or None, default=None
        Online tracker for joint action frequency matrices between makers.
        Generates two full-width plots (first and last window means) if provided.
    stats_belief : OnlineVectorStats or None, default=None
        Online tracker for belief matrices per maker across simulation rounds.
        Adds one row of heatmaps (mean ± std) if provided.
    annot : {'all', 'none', 'above_mean'}, default='above_mean'
        Controls how heatmap cells are annotated:
        - `'all'` — annotate all cells.
        - `'none'` — no annotations.
        - `'above_mean'` — annotate only cells above the matrix mean.
    title : str, default='Makers Statistics Summary Plot'
        Overall title of the figure.

    Returns
    -------
    : matplotlib.figure.Figure
        The complete figure containing all generated subplots.

    Notes
    -----
    - Each section's presence depends on which `OnlineVectorStats` objects are provided.
    - Action and belief heatmaps use maker-specific price grids.
    - The function closes the figure before returning it (`plt.close()`) to avoid 
      automatic display in notebooks.
    """
    n_makers = len(makers)
    n_rows = (stats_cci is not None) +\
        (stats_sorted_cci is not None) +\
        2 * (stats_actions_freq is not None) +\
        2 * (stats_joint_actions_freq is not None) +\
        (stats_belief is not None)

    row = 0

    fig = plt.figure(figsize=(8 * n_makers, 6 * n_rows), constrained_layout=True)
    fig.suptitle(title, fontsize=20)

    gs = gridspec.GridSpec(n_rows, n_makers, figure=fig)

    # Row 1: mean cci
    if stats_cci is not None:
        plot_makers_cci(
            xlabel = 'Round',
            x = window_size * np.arange(stats_cci.dim[1]),
            cci = stats_cci.get_mean(),
            std = stats_cci.get_std(sample=False),
            min = stats_cci.get_min(),
            max = stats_cci.get_max(),
            makers_name = [maker.name for maker in makers],
            title = 'Mean Calvano Collusion Index (CCI)',
            ax = fig.add_subplot(gs[row, :])
        )
        row += 1

    # Row 2: mean sorted cci
    if stats_sorted_cci is not None:
        plot_makers_cci(
            xlabel = 'Round',
            x = window_size * np.arange(stats_cci.dim[1]),
            cci = stats_sorted_cci.get_mean(),
            std = stats_sorted_cci.get_std(sample=False),
            min = stats_sorted_cci.get_min(),
            max = stats_sorted_cci.get_max(),
            makers_name = [maker.name for maker in makers],
            title = 'Mean Sorted Calvano Collusion Index (CCI)',
            ax = fig.add_subplot(gs[row, :])
        )
        row += 1

    # Row 3 and 4: mean relative actions frequency
    if stats_actions_freq is not None:
        for i, maker in enumerate(makers):
            matrix_mean = np.full(2 * (len(maker.prices),), np.nan)
            matrix_mean[*maker.price_to_index(maker.action_space).T] = stats_actions_freq.get_mean()[i, 0, :]
            
            matrix_std = np.full(2 * (len(maker.prices),), np.nan)
            matrix_std[*maker.price_to_index(maker.action_space).T] = stats_actions_freq.get_std(sample=False)[i, 0, :]

            subfig = fig.add_subfigure(gs[row, i])
            ax = subfig.add_subplot(111)
            plot_maker_actions_freq(
                maker = maker,
                matrix = matrix_mean,
                matrix_stdev = matrix_std,
                annot = 'all',
                title = 'Mean Rel. Actions Freq. - First Window',
                ax = ax
            )
        row += 1

        for i, maker in enumerate(makers):
            matrix_mean = np.full(2 * (len(maker.prices),), np.nan)
            matrix_mean[*maker.price_to_index(maker.action_space).T] = stats_actions_freq.get_mean()[i, 1, :]
            
            matrix_std = np.full(2 * (len(maker.prices),), np.nan)
            matrix_std[*maker.price_to_index(maker.action_space).T] = stats_actions_freq.get_std(sample=False)[i, 1, :]

            subfig = fig.add_subfigure(gs[row, i])
            ax = subfig.add_subplot(111)
            plot_maker_actions_freq(
                maker = maker,
                matrix = matrix_mean,
                matrix_stdev = matrix_std,
                annot = 'all',
                title = 'Mean Rel. Actions Freq. - Last Window',
                ax = ax
            )
        row += 1

    # Row 5 and 4: mean relative joint actions frequency
    if (stats_joint_actions_freq is not None):
        ax_comb_1 = fig.add_subplot(gs[row, :])
        ax_comb_2 = fig.add_subplot(gs[row+1, :])

        if n_makers == 2:
            plot_makers_joint_actions_freq_hm(
                makers = makers,
                matrix = stats_joint_actions_freq.get_mean()[0],
                matrix_stdev = stats_joint_actions_freq.get_std(sample=False)[0],
                annot = annot,
                title = 'Mean Rel. Joint Actions Freq. - First Window',
                ax = ax_comb_1
            )

            plot_makers_joint_actions_freq_hm(
                makers = makers,
                matrix = stats_joint_actions_freq.get_mean()[1],
                matrix_stdev = stats_joint_actions_freq.get_std(sample=False)[1],
                annot = annot,
                title = 'Mean Rel. Joint Actions Freq. - Last Window',
                ax = ax_comb_2
            )
        else:
            plot_makers_joint_actions_freq_pc(
                makers = makers,
                matrix = stats_joint_actions_freq.get_mean()[0],
                threshold = 'above_mean_1_std',
                title = 'Mean Rel. Joint Actions Freq. - First Window',
                ax = ax_comb_1
            )

            plot_makers_joint_actions_freq_pc(
                makers = makers,
                matrix = stats_joint_actions_freq.get_mean()[1],
                threshold = 'above_mean_1_std',
                title = 'Mean Rel. Joint Actions Freq. - Last Window',
                ax = ax_comb_2
            )
        row += 2

    # Row 7: mean final belief
    if stats_belief is not None:
        for i, maker in enumerate(makers):
            matrix_mean = np.full(2 * (len(maker.prices),), np.nan)
            matrix_mean[*maker.price_to_index(maker.action_space).T] = stats_belief.get_mean()[i]
            matrix_std = np.full(2 * (len(maker.prices),), np.nan)
            matrix_std[*maker.price_to_index(maker.action_space).T] = stats_belief.get_std(sample=False)[i]

            subfig = fig.add_subfigure(gs[row, i])
            ax = subfig.add_subplot(111)
            plot_maker_belief(
                maker = maker,
                matrix = matrix_mean,
                matrix_stdev = matrix_std,
                annot = 'all',
                title = 'Mean Final Belief',
                ax = ax
            )
        row += 1

    plt.close()
    return fig


def plot_maker_actions(
    maker: Maker,
    round_range: slice = slice(None),
    ax: plt.Axes|None = None
) -> plt.Axes:
    """
    Plot the evolution of ask and bid prices for a single market maker over time.

    This function visualizes the history of actions (ask and bid prices) taken by a 
    specific maker. If no axis is provided, a new Matplotlib figure and axis are created.

    Parameters
    ----------
    maker : Maker
        The market maker to plot.
    round_range : slice or None, default=slice(None)
        A slice object to select a subset of rounds to plot.
        By default, all rounds are plotted.
    ax : matplotlib.axes.Axes or None, default=None
        The axis on which to plot. If None, a new figure and axis are created.

    Returns
    -------
    : matplotlib.axes.Axes
        The Matplotlib axis containing the plotted ask and bid prices.
    """
    if ax is None:
        _, ax = plt.subplots()

    actions = maker.history.get_actions(round_range)
    x = np.arange(*round_range.indices(len(maker.history)))

    ax.plot(x, actions[:, 0], label='Ask', color='red', alpha=0.5, marker='*')
    ax.plot(x, actions[:, 1], label='Bid', color='green', alpha=0.5, marker='*')

    ax.set_title(f'Ask/Bid Price Trend - {maker.name.capitalize()}')
    ax.set_xlabel('Round')
    ax.set_ylabel('Price')
    ax.grid(True)
    ax.legend()
    return ax


def plot_maker_actions_freq(
    maker: Maker,
    matrix: np.ndarray|None = None,
    matrix_stdev: np.ndarray|None = None,
    round_range: slice = slice(None),
    normalize: bool = True,
    annot: Literal['all', 'none', 'above_mean'] = 'all',
    title: str = 'Actions Frequency',
    ax: plt.Axes|None = None
) -> plt.Axes:
    """
    Plot a heatmap showing the frequency of a maker's ask/bid price actions.

    This function visualizes how often a trading agent (the maker) selected each
    ask/bid price combination over a given range of rounds. The resulting heatmap
    can display either raw frequencies or normalized probabilities (if `normalize=True`).

    Optionally, a standard deviation matrix (`matrix_stdev`) can be provided to
    display uncertainty information alongside the mean frequency values when
    `annot='above_mean'`.

    Parameters
    ----------
    maker : Maker
        The maker instance containing the trading history and price grid.
    matrix : np.ndarray or None, default=None
        A 2D array representing the ask/bid frequency matrix to plot.
        Each entry at position (i, j) corresponds to the frequency of choosing
        bid price `j` and ask price `i`. If None, the matrix is computed from
        `maker.history` over the specified `round_range`.
    matrix_stdev : np.ndarray or None, default=None
        A 2D array of the same shape as `matrix`, representing the standard
        deviation (or uncertainty) associated with each cell value. If provided,
        values are displayed as “value ± stdev” in the heatmap annotations when
        `annot='above_mean'`.
    round_range : slice, default=slice(None)
        The range of rounds to include when computing the action frequencies.
        Ignored if `matrix` is provided.
    normalize : bool, default=True
        If True, normalize frequencies so that the total sum equals 1, converting
        raw counts to probabilities.
    annot : {'all', 'none', 'above_mean'}, default='all'
        Controls whether and how to annotate heatmap cells:
        - `'all'` — annotate all cells with their values.
        - `'none'` — no annotations.
        - `'above_mean'` — annotate only cells with values above the matrix mean.
          If `matrix_stdev` is provided, these annotations include “± stdev”.
    title : str, default='Actions Frequency'
        Title of the heatmap. The maker's name is automatically appended.
    ax : matplotlib.axes.Axes or None, default=None
        The Matplotlib axis on which to draw the heatmap. 
        If None, a new figure and axis are created.

    Returns
    -------
    : matplotlib.axes.Axes
        The axis object containing the generated heatmap.

    Notes
    -----
    - The x-axis corresponds to ask prices.
    - The y-axis corresponds to bid prices.
    - If `normalize=True`, cell values represent relative frequencies (sum to 1).
    - If `matrix` is provided, it will be transposed to match the expected orientation.
    - If `matrix_stdev` is provided, it will also be transposed automatically.
    """
    if ax is None:
        _, ax = plt.subplots()
    
    if matrix is None:
        unique_actions, freqs = maker.history.compute_freqs(round_range, return_unique=True)
        unique_actions = maker.price_to_index(unique_actions)

        if normalize:
            freqs = freqs / sum(freqs)

        matrix = np.full(2*(len(maker.prices),), np.nan)
        matrix[unique_actions[:, 1], unique_actions[:, 0]] = freqs
    else:
        matrix = matrix.T
        matrix_stdev = matrix_stdev.T if matrix_stdev is not None else None

    fmt = ''
    base_line = 12 if matrix_stdev is None else 10
    if annot == 'all':
        if matrix_stdev is None:
            annotations = True
            fmt = f'.{DECIMAL_PLACES_VALUES}f' if normalize else '.0f'
        else:
            annotations = np.empty_like(matrix, dtype=object)
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    annotations[i, j] = f'{matrix[i, j]:.{DECIMAL_PLACES_VALUES}f}\n±{matrix_stdev[i, j]:.{DECIMAL_PLACES_VALUES}f}'
    elif annot == 'none':
        annotations = False
        fmt = f'.{DECIMAL_PLACES_VALUES}f' if normalize else '.0f'
    elif annot == 'above_mean':
        mean_val = np.nanmean(matrix)
        annotations = np.empty_like(matrix, dtype=object)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j] > mean_val:
                    if matrix_stdev is None:
                        annotations[i, j] = f'{matrix[i, j]:.{DECIMAL_PLACES_VALUES}f}'
                    else:
                        annotations[i, j] = f'{matrix[i, j]:.{DECIMAL_PLACES_VALUES}f}\n±{matrix_stdev[i, j]:.{DECIMAL_PLACES_VALUES}f}'
                else:
                    annotations[i, j] = ''

    sns.heatmap(
        matrix,
        annot = annotations,
        annot_kws = {'size': max(5, base_line - 0.5 * np.log10(matrix.size))},
        fmt = fmt,
        cmap = 'viridis',
        cbar = True,
        xticklabels = maker.prices,
        yticklabels = maker.prices,
        ax = ax
    )

    ax.set_title(f'{title} - {maker.name.capitalize()}')
    ax.set_xlabel('Ask Price')
    ax.set_ylabel('Bid Price')
    return ax


def plot_maker_belief(
    maker: Maker,
    belief_name: str|None = None,
    matrix: np.ndarray|None = None,
    matrix_stdev: np.ndarray|None = None,
    ronud_range: slice = slice(None),
    annot: Literal['all', 'none', 'above_mean'] = 'all',
    title: str = 'Belief',
    ax: plt.Axes|None = None
) -> plt.Axes:
    """
    Plot a heatmap of the maker's belief or a related variable across bid and ask prices.

    This function visualizes a 2D matrix representing the maker's internal belief 
    (or another related variable specified by `belief_name`) over combinations 
    of bid and ask prices. The data can be provided directly via `matrix` or 
    retrieved from the `Maker` instance. If `belief_name == 'extra'`, the data 
    is obtained from `maker.history.get_extras(ronud_range)`.

    Parameters
    ----------
    maker : Maker
        The maker instance containing relevant price data and belief-related variables.
    belief_name : str or None, default=None
        Name of the maker attribute representing the belief or related quantity 
        (e.g., `'belief'`, `'q_values'`, `'v_values'`, `'policy'`). 
        If `'extra'`, the data is retrieved using `maker.history.get_extras(ronud_range)`.
        Ignored if `matrix` is provided.
    matrix : np.ndarray or None, default=None
        Optional precomputed 2D array to plot. If provided, it is used directly 
        instead of retrieving belief data from `maker`.
    matrix_stdev : np.ndarray or None, default=None
        Optional 2D array of standard deviations corresponding to `matrix` values.
        When provided, each heatmap cell annotation includes the mean ± standard deviation.
    ronud_range : slice, default=slice(None)
        Range of rounds to consider when retrieving belief data from 
        `maker.history.get_extras`. Only used if `belief_name == 'extra'`.
    annot : {'all', 'none', 'above_mean'}, default='all'
        Controls how annotations are displayed:
        - `'all'` — annotate all cells.
        - `'none'` — no annotations.
        - `'above_mean'` — annotate only cells above the matrix mean.
    title : str, default='Belief'
        Title of the plot.
    ax : matplotlib.axes.Axes or None, default=None
        Axis on which to draw the heatmap. If None, a new figure and axis are created.

    Returns
    -------
    : matplotlib.axes.Axes
        The axis containing the generated heatmap.

    Notes
    -----
    - The x-axis represents ask prices, and the y-axis represents bid prices.
    - If `matrix_stdev` is provided, annotations include both mean and standard deviation.
    """
    if ax is None:
        _, ax = plt.subplots()
    
    if matrix is None:
        unique_actions = maker.price_to_index(maker.action_space)
        freqs = getattr(maker, belief_name) if belief_name != 'extra' else maker.history.get_extras(ronud_range)

        matrix = np.full(2*(len(maker.prices),), np.nan)
        matrix[unique_actions[:, 1], unique_actions[:, 0]] = freqs
    else:
        matrix = matrix.T
        matrix_stdev = matrix_stdev.T if matrix_stdev is not None else None

    fmt = ''
    base_line = 12 if matrix_stdev is None else 10
    if annot == 'all':
        if matrix_stdev is None:
            annotations = True
            fmt = f'.{DECIMAL_PLACES_VALUES}f'
        else:
            annotations = np.empty_like(matrix, dtype=object)
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    annotations[i, j] = f'{matrix[i, j]:.{DECIMAL_PLACES_VALUES}f}\n±{matrix_stdev[i, j]:.{DECIMAL_PLACES_VALUES}f}'
    elif annot == 'none':
        annotations = False
        fmt = f'.{DECIMAL_PLACES_VALUES}f'
    elif annot == 'above_mean':
        mean_val = np.nanmean(matrix)
        annotations = np.empty_like(matrix, dtype=object)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j] > mean_val:
                    if matrix_stdev is None:
                        annotations[i, j] = f'{matrix[i, j]:.{DECIMAL_PLACES_VALUES}f}'
                    else:
                        annotations[i, j] = f'{matrix[i, j]:.{DECIMAL_PLACES_VALUES}f}\n±{matrix_stdev[i, j]:.{DECIMAL_PLACES_VALUES}f}'
                else:
                    annotations[i, j] = ''

    sns.heatmap(
        matrix,
        annot = annotations,
        annot_kws = {'size': max(5, base_line - 0.5 * np.log10(matrix.size))},
        fmt = fmt,
        cmap = 'viridis',
        cbar = True,
        xticklabels = maker.prices,
        yticklabels = maker.prices,
        ax = ax
    )

    ax.set_title(f'{title} ({belief_name if belief_name is not None else 'custom'}) - {maker.name.capitalize()}')
    ax.set_xlabel('Ask Price')
    ax.set_ylabel('Bid Price')
    return ax


def plot_maker_belief_evolution_dc(
    maker: Maker,
    values: np.ndarray|None = None,
    adaptive_scale: bool = False,
    log_scale: bool = False,
    interval: int = 100,
) -> None:
    """
    Display the evolution of a market maker's belief over time as a heatmap.

    The function creates an interactive widget (slider and autoplay) to visualize
    the evolution of the market maker's belief for each frame. It is designed for 
    use in Jupyter notebooks and does not return any value.

    Parameters
    ----------
    maker : Maker
        The market maker instance whose belief evolution is to be visualized.
    values : np.ndarray or None, default=None
        Optional array of shape (n_frames, N) containing belief values for each frame.
        If None, the function uses `maker.history.get_extras()` to obtain the values.
    adaptive_scale : bool, default=False
        If True, the colorbar is updated for each frame based on the current data range.
    log_scale : bool, default=False
        If True, the values are transformed using `sign(value) * log10(abs(value))` for visualization.
    interval : int, default=100
        Time interval in milliseconds between frames during autoplay.
    """
    def make_dense(frame_idx):
        dense = np.full(2*(len(maker.prices),), np.nan)
        dense[unique_actions[:, 1], unique_actions[:, 0]] = values[frame_idx]
        return dense
    
    def update(change):
        frame = change['new']
        data = make_dense(frame)
        im.set_data(data)

        if adaptive_scale:
            vmin = np.nanmin(data)
            vmax = np.nanmax(data)
            im.set_clim(vmin, vmax)

        ax.set_title(f'Belief Evolution {'(Log10 Scale) ' if log_scale else ''}- {maker.name.capitalize()}\nFrame {frame}')
        fig.canvas.draw_idle()
        return
    
    unique_actions = maker.price_to_index(maker.action_space)
    values = maker.history.get_extras() if values is None else values
    n_frames = len(values)

    if log_scale:
        values = np.sign(values) * np.log10(np.where(values != 0, np.abs(values), 1))

    fig, ax = plt.subplots()
    initial_image = make_dense(0)
    if adaptive_scale:
        im = ax.imshow(initial_image, cmap='viridis', interpolation='nearest')
    else:
        im = ax.imshow(initial_image, cmap='viridis', interpolation='nearest', vmax=np.max(values), vmin=np.min(values))
    fig.colorbar(im)

    ax.set_title(f'Belief Evolution {'(Log10 Scale) ' if log_scale else ''}- {maker.name.capitalize()}\nFrame 0')
    ax.set_xlabel('Ask Price')
    ax.set_ylabel('Bid Price')
    ax.set_xticks(np.arange(len(maker.prices)))
    ax.set_yticks(np.arange(len(maker.prices)))
    ax.set_xticklabels(maker.prices)
    ax.set_yticklabels(maker.prices)

    play = widgets.Play(
        value = 0,
        min = 0,
        max = n_frames - 1,
        step = 1,
        interval = interval,
        description = 'Play',
        disabled = False
    )

    slider = widgets.IntSlider(min=0, max=n_frames - 1, step=1, description='Frame')
    widgets.jslink((play, 'value'), (slider, 'value'))

    slider.observe(update, names='value')

    controls = widgets.HBox([play, slider])
    display(widgets.VBox([controls]))
    return


def plot_maker_belief_evolution_sc(
    maker: Maker,
    matrix: np.ndarray|None = None,
    curr_idx: int|None = None,
    next_idx: int|None = None,
    title: str = 'Belief Evolution',
    ax: plt.Axes|None = None
) -> plt.Axes:
    """
    Plot the evolution of the maker's belief between two rounds.

    This function visualizes how a 2D belief-related matrix changes from one round
    (`curr_idx`) to the next (`next_idx`). The resulting heatmap displays the difference 
    `matrix[next] - matrix[curr]` for each bid-ask combination, while each cell is annotated 
    with the pair of values `(current, next)` for direct comparison. Data can be supplied 
    directly using `matrix`, or retrieved from the maker's history via 
    `maker.history.get_extras`.

    Parameters
    ----------
    maker : Maker
        The maker instance containing pricing information and stored historical data.
    matrix : np.ndarray or None, default=None
        Optional precomputed 3D array of shape `(2, n_prices, n_prices)` where  
        `matrix[0]` contains current-round values and `matrix[1]` contains next-round 
        values. If provided, it is used directly (after transposing). Otherwise, values 
        are obtained from `maker.history.get_extras(curr_idx)` and 
        `maker.history.get_extras(next_idx)`.
    curr_idx : int or None, default=None
        Round index for the “current” snapshot of the belief or tracked quantity. 
        Ignored if `matrix` is provided.
    next_idx : int or None, default=None
        Round index for the “next” snapshot of the belief or tracked quantity. 
        Ignored if `matrix` is provided.
    title : str, default='Belief Evolution'
        Title for the resulting heatmap.
    ax : matplotlib.axes.Axes or None, default=None
        Axis on which to draw the heatmap. If None, a new figure and axis are created.

    Returns
    -------
    : matplotlib.axes.Axes
        Axis containing the generated heatmap.

    Notes
    -----
    - The heatmap displays the difference between the next and current matrices.
    - Cell annotations contain two lines: the current-round value and the next-round value.
    - The x-axis represents ask prices, and the y-axis represents bid prices.
    - Only cells corresponding to valid actions (given by `maker.action_space`) are filled.
    """
    if ax is None:
        _, ax = plt.subplots()
        
    if matrix is None:
        unique_actions = maker.price_to_index(maker.action_space)

        matrix = np.full((2, len(maker.prices), len(maker.prices)), np.nan)
        matrix[0, unique_actions[:, 1], unique_actions[:, 0]] = maker.history.get_extras(curr_idx)
        matrix[1, unique_actions[:, 1], unique_actions[:, 0]] = maker.history.get_extras(next_idx)
    else:
        matrix = matrix.swapaxes(1, 2)

    annotations = np.empty(matrix.shape[1:], dtype=object)
    for i in range(matrix.shape[1]):
        for j in range(matrix.shape[2]):
            if not np.isnan(matrix[:, i, j]).all():
                annotations[i, j] = f'{matrix[0, i, j]:.{DECIMAL_PLACES_VALUES}f}\n{matrix[1, i, j]:.{DECIMAL_PLACES_VALUES}f}'

    sns.heatmap(
        matrix[1] - matrix[0],
        annot = annotations,
        annot_kws = {'size': 8},
        fmt = '',
        cmap = 'viridis',
        cbar = True,
        xticklabels = maker.prices,
        yticklabels = maker.prices,
        ax = ax
    )

    ax.set_title(f'{title} ({curr_idx} -> {next_idx}) - {maker.name.capitalize()}')
    ax.set_xlabel('Ask Price')
    ax.set_ylabel('Bid Price')
    return ax


def plot_maker_rewards(
    maker: Maker,
    round_range: slice = slice(None),
    nash_reward: float|None = None,
    coll_reward: float|None = None,
    ax: plt.Axes|None = None
) -> plt.Axes:
    """
    Plot the reward trend of a single maker across a range of rounds.

    Optionally includes horizontal reference lines for Nash equilibrium
    reward and collusion reward.

    Parameters
    ----------
    maker : Maker
        The maker agent whose rewards will be plotted.
    round_range : slice, default=slice(None)
        Slice defining which rounds to plot.
        By default, all rounds are plotted.
    nash_reward : float or None, default=None
        Reference value for the Nash equilibrium reward. If None, no line is plotted.
    coll_reward : float or None, default=None
        Reference value for the collusion reward. If None, no line is plotted.
    ax : matplotlib.axes.Axes or None, default=None
        Axis on which to plot. If None, a new axis is created.

    Returns
    -------
    : matplotlib.axes.Axes
        The axis object containing the plot.
    """
    if ax is None:
        _, ax = plt.subplots()

    cmap = plt.get_cmap("tab10")
    color = cmap(int(maker.name.split("_")[-1]) % 10)

    rewards = maker.history.get_rewards(round_range)
    x = np.arange(*round_range.indices(len(maker.history)))

    ax.plot(x, rewards, label=maker.name.capitalize(), color=color, marker='*')
    if nash_reward is not None:
        ax.axhline(nash_reward, label='Nash', color='blue', ls='--')
    if coll_reward is not None:
        ax.axhline(coll_reward, label='Collusion', color='red', ls='--')

    ax.set_title(f'Rewards Trend - {maker.name.capitalize()}')
    ax.set_xlabel('Round')
    ax.set_ylabel('Reward')
    ax.grid(True)
    ax.legend()
    return ax


def plot_makers_best_actions(
    makers: List[Maker],
    round_range: slice = slice(None),
    true_value: float = 0.5,
    ax: plt.Axes|None = None
) -> plt.Axes:
    """
    Plot the best prices (closest to the true value) for each maker across rounds.

    For each maker, bid and ask prices are displayed with different markers and colors.
    The best price per round (either the closest ask or bid to the true value) is highlighted.

    Parameters
    ----------
    makers : list of Maker
        List of `Maker` instances containing historical bid/ask prices.
    round_range : slice, default=slice(None)
        Range of rounds to include in the plot. By default, all rounds are plotted.
    true_value : float, default=0.5
        The true asset value used as reference to determine best prices.
    ax : matplotlib.axes.Axes or None, default=None
        Axis on which to draw the plot. If None, a new figure and axis are created.

    Returns
    -------
    : matplotlib.axes.Axes
        The axis containing the plot.
    """
    if ax is None:
        _, ax = plt.subplots()

    cmap = plt.get_cmap("tab10")
    markers = {'ask': 'o', 'bid': '*'}

    actions = np.array([maker.history.get_actions(round_range) for maker in makers])
    x = np.arange(*round_range.indices(actions.shape[1]))

    for i, maker in enumerate(makers):
        color = cmap(i % 10)

        ask_prices = actions[i, :, 0]
        bid_prices = actions[i, :, 1]

        ax.scatter(
            x,
            ask_prices,
            label = f'{maker.name.capitalize()} - Ask',
            color = color,
            marker = markers['ask'],
            s = 100 * (len(makers) - i)
        )
        ax.scatter(
            x,
            bid_prices,
            label = f'{maker.name.capitalize()} - Bid',
            color = color,
            marker = markers['bid'],
            s = 100 * (len(makers) - i)
        )

    min_ask_prices = np.min(actions[:, :, 0], axis=0)
    max_bid_prices = np.max(actions[:, :, 1], axis=0)

    best_is_ask = (true_value - min_ask_prices) >= (max_bid_prices - true_value)
    best_is_bid = (true_value - min_ask_prices) <= (max_bid_prices - true_value)

    ax.scatter(x[best_is_ask], min_ask_prices[best_is_ask], s=250, edgecolors='red', facecolors='none')
    ax.scatter(x[best_is_bid], max_bid_prices[best_is_bid], s=250, edgecolors='red', facecolors='none')

    ax.axhline(true_value,  label='True Value', color='black', ls=':',)

    ax.set_title('Best Price per Maker vs True Value')
    ax.set_xlabel('Round')
    ax.set_ylabel('Price')
    ax.grid(True)
    ax.legend(
        loc = 'upper center',
        bbox_to_anchor = (0.5, -0.15),
        ncol = len(makers) + 1,
    )
    return ax


def plot_makers_cci(
    xlabel: str,
    x: np.ndarray,
    cci: np.ndarray|None,
    std: np.ndarray|None = None,
    min: np.ndarray|None = None,
    max: np.ndarray|None = None,
    makers_name: List[str]|None = None,
    title: str = 'Calvano Collusion Index (CCI)',
    ax: plt.Axes|None = None,
) -> plt.Axes:
    """
    Plot the Calvano Collusion Index (CCI) for multiple makers.

    Each maker's CCI time series is plotted as a separate line. 
    The average CCI across makers is also included as a dashed red line.
    Optionally, standard deviation bands or min/max envelopes can be added.

    Parameters
    ----------
    xlabel : str
        Label of the horizontal axis.
    x : np.ndarray
        Horizontal axis values (e.g., rounds, time steps, or windows).
    cci : np.ndarray
        Array of shape (n_makers, n_windows) containing the CCI per maker.
    std : np.ndarray or None, default=None
        Array of shape (n_makers, n_windows) with the standard deviation of the CCI. 
        If provided, shaded regions representing ±1 standard deviation are plotted.
    min : np.ndarray or None, default=None
        Array of shape (n_makers, n_windows) with the minimum CCI values per maker.
        If provided, plotted as dashed boundary lines.
    max : np.ndarray or None, default=None
        Array of shape (n_makers, n_windows) with the maximum CCI values per maker.
        If provided, plotted as dashed boundary lines.
    makers_name : list of str or None, default=None
        Names of the makers, used as labels in the legend. If None, generic names are assigned.
    title : str, default='Calvano Collusion Index (CCI)'
        Title of the plot.
    ax : matplotlib.axes.Axes or None, default=None
        Axis on which to plot. If None, a new axis is created.

    Returns
    -------
    : matplotlib.axes.Axes
        The axis containing the plot.
    
    References
    ----------
    - Calvano, E., Calzolari, G., Denicolò, V., & Pastorello, S. (2020).  
    Artificial intelligence, algorithmic pricing, and collusion.  
    *American Economic Review, 110*(10), 3267-3297.  
    https://doi.org/10.1257/aer.20190623
    """
    if ax is None:
        _, ax = plt.subplots()

    cmap = plt.get_cmap("tab10")
 
    if std is None:
        std = [None] * len(cci)
    if min is None:
        min = [None] * len(cci)
    if max is None:
        max = [None] * len(cci)
    if makers_name is None:
        makers_name = [f'unknown_{i}' for i in range(len(cci))]

    for i, (maker, series_cci, series_std, series_min, series_max) in enumerate(zip(makers_name, cci, std, min, max)):
        ax.plot(x, series_cci, label=maker.capitalize(), color=cmap(i % 10), marker='*')
        if series_std is not None:
            ax.fill_between(
                x,
                series_cci - series_std,
                series_cci + series_std,
                color = cmap(i % 10),
                alpha = 0.3
            )
        if series_min is not None:
            ax.plot(x, series_min, color=cmap(i % 10), alpha=0.5, ls='--')
        if series_max is not None:
            ax.plot(x, series_max, color=cmap(i % 10), alpha=0.5, ls='--')
    
    ax.plot(x, cci.mean(axis=0), label='Average', color='red', ls='--')

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Collusion Index')
    ax.grid(True)
    ax.legend()
    return ax


def plot_makers_joint_actions_freq_hm(
    makers: Tuple[Maker, Maker],
    matrix: np.ndarray|None = None,
    matrix_stdev: np.ndarray|None = None,
    round_range: slice = slice(None),
    normalize: bool = True,
    annot: Literal['all', 'none', 'above_mean'] = 'all',
    title: str = 'Joint Actions Frequency',
    ax: plt.Axes|None = None
) -> plt.Axes:
    """
    Plot a heatmap showing the joint action frequencies of two makers.

    This function visualizes how often pairs of actions — one from each maker —
    occur together over a given range of rounds. The resulting heatmap provides
    insight into correlations or interactions between the two makers' behaviors.
    Frequencies can be normalized to sum to 1. Optionally, a standard deviation
    matrix can be provided to display uncertainty alongside each cell.

    If a precomputed joint frequency matrix is not provided via `matrix`, it is
    automatically constructed from the makers' recorded action histories.

    Parameters
    ----------
    makers : tuple of Maker
        A tuple containing exactly two `Maker` instances whose joint action
        frequencies will be computed and visualized.
    matrix : np.ndarray or None, default=None
        A 2D array representing the joint frequency of actions. The entry at
        position (i, j) corresponds to how often maker 1 took action `j` while
        maker 2 took action `i`. If None, the matrix is computed from the makers'
        recorded histories over the given `round_range`.
    matrix_stdev : np.ndarray or None, default=None
        Optional 2D array of standard deviations corresponding to `matrix`. Used
        to annotate each cell with ± value if provided.
    round_range : slice, default=slice(None)
        The range of rounds to include when computing joint action frequencies.
        Ignored if `matrix` is provided.
    normalize : bool, default=True
        If True, normalize frequencies so that the total sum equals 1.
        Ignored if `matrix` is provided.
    annot : {'all', 'none', 'above_mean'}, default='all'
        Controls cell annotations:
        - `'all'` — annotate every cell with its value (and ± if `matrix_stdev` is given).
        - `'none'` — no annotations.
        - `'above_mean'` — annotate only cells with values above the matrix mean.
    title : str, default='Joint Actions Frequency'
        The title of the heatmap.
    ax : matplotlib.axes.Axes or None, default=None
        Axis on which to draw the heatmap. If None, a new figure and axis are created.

    Returns
    -------
    : matplotlib.axes.Axes
        The axis object containing the rendered heatmap.

    Raises
    ------
    ValueError
        If `makers` does not contain exactly two `Maker` instances.

    Notes
    -----
    - The x-axis corresponds to actions taken by the first maker (`makers[0]`).
    - The y-axis corresponds to actions taken by the second maker (`makers[1]`).
    - If `normalize=True`, cell values represent relative frequencies (sum to 1).
    - If a precomputed matrix is provided, it is transposed to match the expected
      orientation (x-axis = maker 1, y-axis = maker 2).
    - Annotations display values in decimal format controlled by `DECIMAL_PLACES_VALUES`.
    """
    if ax is None:
        _, ax = plt.subplots()
    
    if len(makers) != 2:
        ValueError(f'`makers` must be a tuple containing exactly two Maker instances, but got {len(makers)}')

    if matrix is None:
        joint_actions = np.concat([
            makers[0].history.get_actions(round_range, return_index=True)[:, None],
            makers[1].history.get_actions(round_range, return_index=True)[:, None]
        ], axis=1)
        unique_joint_actions, freqs = np.unique(joint_actions, return_counts=True, axis=0)

        if normalize:
            freqs = freqs / sum(freqs)

        matrix = np.full((makers[0].n_arms, makers[1].n_arms), .0 if normalize else 0)
        matrix[unique_joint_actions[:, 1], unique_joint_actions[:, 0]] = freqs
    else:
        matrix = matrix.T
        matrix_stdev = matrix_stdev.T if matrix_stdev is not None else None

    fmt = ''
    base_line = 12 if matrix_stdev is None else 10
    if annot == 'all':
        if matrix_stdev is None:
            annotations = True
            fmt = f'.{DECIMAL_PLACES_VALUES}f' if normalize else '.0f'
        else:
            annotations = np.empty_like(matrix, dtype=object)
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    annotations[i, j] = f'{matrix[i, j]:.{DECIMAL_PLACES_VALUES}f}\n±{matrix_stdev[i, j]:.{DECIMAL_PLACES_VALUES}f}'
    elif annot == 'none':
        annotations = False
        fmt = f'.{DECIMAL_PLACES_VALUES}f' if normalize else '.0f'
    elif annot == 'above_mean':
        mean_val = np.nanmean(matrix)
        annotations = np.empty_like(matrix, dtype=object)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j] > mean_val:
                    if matrix_stdev is None:
                        annotations[i, j] = f'{matrix[i, j]:.{DECIMAL_PLACES_VALUES}f}'
                    else:
                        annotations[i, j] = f'{matrix[i, j]:.{DECIMAL_PLACES_VALUES}f}\n±{matrix_stdev[i, j]:.{DECIMAL_PLACES_VALUES}f}'
                else:
                    annotations[i, j] = ''

    sns.heatmap(
        matrix,
        annot = annotations,
        annot_kws = {'size': max(5, base_line - 0.5 * np.log10(matrix.size))},
        fmt = fmt,
        cmap = 'viridis',
        cbar = True,
        xticklabels = makers[0].action_space,
        yticklabels = makers[1].action_space,
        ax = ax
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
    ax.set_xlabel(makers[0].name.capitalize())
    ax.set_ylabel(makers[1].name.capitalize())
    ax.set_title(title)
    return ax


def plot_makers_joint_actions_freq_pc(
    makers: List[Maker],
    matrix: np.ndarray|None = None,
    round_range: slice = slice(None),
    threshold: float|Literal['all', 'above_mean', 'above_mean_X_std'] = 'above_mean',
    title: str = 'Relative Joint Actions Frequency',
    location: Literal['left', 'right', 'top', 'bottom'] = 'right',
    pad: float = 0.01,
    ax: plt.Axes|None = None
)-> plt.Axes:
    """
    Plot a parallel coordinates-style visualization of joint action frequencies for multiple makers.

    This function visualizes how often combinations of actions across multiple makers occur
    together. Each maker is represented along the x-axis, and action indices along the y-axis.
    Lines connecting actions illustrate the frequency of each joint action, with line width
    and color intensity representing normalized frequency values. Optionally, a threshold
    can filter which joint actions are displayed.

    If a precomputed joint frequency matrix is not provided via `matrix`, it is automatically
    constructed from the makers' recorded action histories over the specified `round_range`.

    Parameters
    ----------
    makers : list of Maker
        A list of `Maker` instances whose joint action frequencies will be visualized.
    matrix : np.ndarray or None, default=None
        Precomputed joint frequency matrix. If None, the matrix is computed from the
        makers' action histories over the given `round_range`.
    round_range : slice, default=slice(None)
        Range of rounds to include when computing joint action frequencies.
        Ignored if `matrix` is provided.
    threshold : float or {'all', 'above_mean', 'above_mean_X_std'}, default='above_mean'
        Minimum frequency value for joint actions to be displayed:
        - `'all'` — display all joint actions.
        - `'above_mean'` — display only joint actions with frequency above the mean.
        - `'above_mean_X_std'` — display joint actions above mean + X standard deviations
          (replace X with a numeric value in the string, e.g., `'above_mean_2_std'`).
        - float — display joint actions with frequency ≥ threshold.
    title : str, default='Relative Joint Actions Frequency'
        Title of the plot.
    location : {'left', 'right', 'top', 'bottom'}, default='right'
        The location, relative to the parent Axes, where the colorbar Axes is created.
    pad : float, default=0.0
        Padding used to centre the color bar.
    ax : matplotlib.axes.Axes or None, default=None
        Axis on which to draw the plot. If None, a new figure and axis are created.

    Returns
    -------
    : matplotlib.axes.Axes
        The axis object containing the rendered parallel coordinates plot.

    Raises
    ------
    ValueError
        If `threshold` is not a recognized string or a float.

    Notes
    -----
    - The x-axis corresponds to different makers.
    - The y-axis corresponds to action indices.
    - Line width and color intensity represent the normalized frequency of each joint action.
    - When makers have different action spaces, action labels are annotated at each point.
    - If a precomputed matrix is provided, it is used directly for plotting; otherwise,
      it is computed from the makers' action histories.
    - Random jitter is applied to the vertical position of points to improve visibility
      when multiple joint actions share the same value.
    - Ensure that `np.concatenate` is used instead of `np.concat` when combining action arrays.
    """
    if ax is None:
        _, ax = plt.subplots(constrained_layout=True)
    
    if matrix is None:
        joint_actions = np.concat([
            maker.history.get_actions(round_range, return_index=True)[:, None] for maker in makers
        ], axis=1)
        unique_joint_actions, freqs = np.unique(joint_actions, return_counts=True, axis=0)

        matrix = np.full(tuple(maker.n_arms for maker in makers), .0)

        matrix[tuple(unique_joint_actions.T)] = freqs / sum(freqs)
    norm_matrix = matrix / matrix.max()

    action_spaces = [maker.action_space for maker in makers]
    shapes = [action_space.shape for action_space in action_spaces]

    action_space = action_spaces[0]
    all_equal = True

    if len(set(shapes)) > 1:
        all_equal = False
    else:
        all_equal = all(np.array_equal(action_space, arr) for arr in action_spaces[1:])

    if threshold == 'all':
        threshold = float('-inf')
    elif threshold == 'above_mean':
        threshold = matrix.mean()
    elif (match := re.match(r'^above_mean_(-?\d+(?:\.\d+)?)_std$', threshold)):
        threshold = matrix.mean() + float(match.group(1)) * matrix.std()
    elif not isinstance(threshold, float):
        ValueError(f'Unrecognized threshold value: {threshold}')

    x = np.arange(len(makers))
    sorted_indices = np.array(np.unravel_index(np.argsort(matrix, axis=None)[::-1], matrix.shape)).T

    for idx in sorted_indices:
        if matrix[*idx] >= threshold:
            linewidth = 1 + 5 * norm_matrix[*idx]**2
            y_jittered = idx + np.random.uniform(-JITTER_STRENGTH, JITTER_STRENGTH, size=idx.shape)
            ax.plot(x, y_jittered, color=plt.cm.viridis(norm_matrix[*idx]), linewidth=linewidth)

    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=matrix.min(), vmax=matrix.max()))
    cbar = plt.colorbar(sm, ax=ax, pad=pad, location=location)

    if not all_equal:
        index = [slice(None)] * len(makers)
        for i, maker in enumerate(makers):
            for j, action in enumerate(maker.action_space):
                index[i] = j
                if (matrix[tuple(index)] >= threshold).any():
                    ax.text(
                        i,
                        j,
                        action,
                        ha = 'center',
                        va = 'center',
                        fontsize = 8,
                        color = 'black',
                        alpha = 0.8,
                        bbox = dict(facecolor='white', edgecolor='none', alpha=0.8, boxstyle='round,pad=0.2')
                    )

    ax.set_title(title)
    ax.set_xlabel('Maker')
    ax.set_ylabel('Action')
    ax.set_xticks(x)
    ax.set_yticks(np.arange(len(action_space)) if all_equal else [])
    ax.set_xticklabels([maker.name.capitalize() for maker in makers])
    ax.set_yticklabels(action_space if all_equal else [])
    ax.xaxis.grid(True, color='gray')
    ax.yaxis.grid(True, color='lightgray', linestyle='--')

    ax2 = ax.twinx()
    ax2.set_ylabel(ax.get_ylabel())
    ax2.set_yticks(ax.get_yticks())
    ax2.set_yticklabels(ax.get_yticklabels())
    return ax
