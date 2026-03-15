""" Common utilities.
"""
import numpy as np



def scale_rewards_array(
    rewards: np.ndarray|float,
    n_rounds: int,
    min_reward: float,
    max_reward: float,
    target_min: float = -1.0,
    target_max: float = 1.0
) -> np.ndarray|float:
    """
    Scale rewards from a given range to a target range.

    Parameters
    ----------
    rewards : np.ndarray or float
        The reward value(s) to be scaled.
        Can be a single float or a NumPy array of floats representing multiple rewards.
    n_rounds : int
        Number of rounds used to compute the rewards.
    min_reward : float
        Minimum possible reward per round.
    max_reward : float
        Maximum possible reward per round.
    target_min : float, default=-1.0
        Minimum value of the target range.
    target_max : float, default=1.0
        Maximum value of the target range.

    Returns
    -------
    : np.ndarray or float
        Scaled rewards in the target range.
        If `rewards` is an array, its shape is preserved.

    Raises
    ------
    ValueError
        If min_reward and max_reward are equal (to avoid division by zero).
    """
    if min_reward == max_reward:
        raise ValueError('`min_reward` and `max_reward` must be different to avoid division by zero')

    original_min = n_rounds * min_reward
    original_max = n_rounds * max_reward

    scaled = target_min + (rewards - original_min) * (target_max - target_min) / (original_max - original_min)
    return scaled


def split_array(arr: np.ndarray, window_size: int) -> np.ndarray:
    """
    Split an array into sub-arrays of fixed window size along the last axis.

    If `window_size` is non-positive, the array is reshaped so that the last
    axis becomes a single window of length equal to its size.
    This is useful, for example, to ensure a consistent shape
    when no actual splitting is performed.

    Parameters
    ----------
    arr : np.ndarray
        Input array to be split.
    window_size : int
        Size of each window. Must be a positive integer.
        If <= 0, the array is reshaped to (..., 1, N), where N is the
        original length of the last axis.

    Returns
    -------
    : np.ndarray
        Reshaped array with shape (..., n_windows, window_size).

    Raises
    ------
    ValueError
        If `window_size` is not a divisor of the length of the last axis.
    """
    if window_size <= 0:
        return arr.reshape(arr.shape[:-1] + (1, -1))
    return arr.reshape(arr.shape[:-1] + (-1, window_size))


def get_calvano_collusion_index(
    rewards: np.ndarray,
    nash_reward: np.ndarray|float,
    coll_reward: np.ndarray|float,
    window_size: int = 0,
    decimal_places: int = 3
) -> np.ndarray:
    """
    Compute the Calvano Collusion Index (CCI) from agent rewards.

    The CCI measures the degree of collusion relative to Nash equilibrium
    and perfect collusion benchmarks. Rewards are optionally aggregated
    over fixed-size windows before computing the index.

    Parameters
    ----------
    rewards : np.ndarray
        Array of shape (n_agents, n_rounds) containing per-agent rewards.
    nash_reward : float or np.ndarray
        Total reward across all agents under Nash equilibrium. If an array is
        provided, it is interpreted as a time series and may be windowed.
    coll_reward : float or np.ndarray
        Total reward across all agents under perfect collusion. If an array is
        provided, it is interpreted as a time series and may be windowed.
    window_size : int, default=0
        Size of non-overlapping windows (in rounds) used to aggregate rewards.
        If 0, rewards are not windowed and are averaged over all rounds.
    decimal_places : int, default=3
        Number of decimal places used when rounding the benchmark rewards.

    Returns
    -------
    : np.ndarray
        Array of CCI values with shape (n_agents, n_windows), where `n_windows`
        depends on the selected `window_size`.

    References
    ----------
    - Calvano, E., Calzolari, G., Denicolò, V., & Pastorello, S. (2020).
    Artificial intelligence, algorithmic pricing, and collusion.
    *American Economic Review, 110*(10), 3267-3297.
    https://doi.org/10.1257/aer.20190623
    """
    if isinstance(nash_reward, float):
        nash_reward = np.array([nash_reward])
    if isinstance(coll_reward, float):
        coll_reward = np.array([coll_reward])

    nash_reward = np.round(nash_reward / len(rewards), decimal_places)
    coll_reward = np.round(coll_reward / len(rewards), decimal_places)

    diff = coll_reward - nash_reward
    diff[np.isclose(coll_reward, nash_reward)] = 1

    cci = (rewards - nash_reward) / diff
    return split_array(cci, window_size).mean(axis=-1)


def get_relative_deviation_competition(
    rewards: np.ndarray,
    nash_reward: np.ndarray|float,
    window_size: int = 0,
    decimal_places: int = 3
) -> np.ndarray:
    """
    Compute the Relative Deviation from Competition (RDC) index from agent rewards.

    The RDC measures how much agents' realized rewards exceed (or fall below)
    the competitive benchmark, defined by the Nash equilibrium reward.
    Optionally, rewards can be aggregated over fixed-size windows before
    computing the index.

    The RDC for an agent or window is defined as:

        RDC = (r - r_comp) / r_comp
    
    where r is the observed average reward and r_comp is the per-agent
    Nash equilibrium reward.

    Parameters
    ----------
    rewards : np.ndarray
        Array of shape (n_agents, n_rounds) containing per-agent rewards.
    nash_reward : float or np.ndarray
        Total reward across all agents under Nash equilibrium. If an array is
        provided, it is interpreted as a time series and may be aggregated
        using the specified `window_size`. The reward is internally converted
        to a per-agent benchmark.
    window_size : int, default=0
        Size of non-overlapping windows (in rounds) used to aggregate rewards.
        If 0, rewards are not windowed and are averaged over all rounds.
    decimal_places : int, default=3
        Number of decimal places used when rounding the per-agent Nash
        equilibrium reward.

    Returns
    -------
    : np.ndarray
        Array of RDC values with shape (n_agents, n_windows), where
        `n_windows` depends on the `window_size`.
    """
    if isinstance(nash_reward, float):
        nash_reward = np.array([nash_reward])

    nash_reward = np.round(nash_reward / len(rewards), decimal_places)
    
    rdc = (rewards - nash_reward) / (nash_reward if nash_reward != 0 else 1)
    return split_array(rdc, window_size).mean(axis=-1)
