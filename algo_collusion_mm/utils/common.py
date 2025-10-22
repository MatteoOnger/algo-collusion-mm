import numpy as np



def scale_rewards_array(
    rewards: np.ndarray|float,
    n_episodes: int,
    min_reward: float,
    max_reward: float,
    target_min: float = -1.0,
    target_max: float = 1.0
) -> np.ndarray | float:
    """
    Scales (cumulative) rewards from a given range to a target range (default [-1, 1]).

    Parameters
    ----------
    rewards : np.ndarray or float
        Array or float of rewards to be scaled.
    n_episodes : int
        Number of episodes over which the cumulative rewards were computed.
    min_reward : float
        Minimum possible reward per episode.
    max_reward : float
        Maximum possible reward per episode.
    target_min : float, default=-1
        Minimum value of the target range.
    target_max : float, default=1
        Maximum value of the target range.

    Returns
    -------
    : np.ndarray or float
        Scaled cumulative rewards in the target range.

    Raises
    ------
    ValueError
        If min_reward and max_reward are equal (to avoid division by zero).
    """
    if min_reward == max_reward:
        raise ValueError('`min_reward` and `max_reward` must be different to avoid division by zero')

    original_min = n_episodes * min_reward
    original_max = n_episodes * max_reward

    scaled = target_min + (rewards - original_min) * (target_max - target_min) / (original_max - original_min)
    return scaled


def split_array(arr: np.ndarray, window_size: int) -> np.ndarray:
    """
    Split an array into sub-arrays of fixed window size along the last axis.

    If `window_size` is non-positive, the array is reshaped so that the last
    axis becomes a single window of length equal to its size.
    This is useful, for example, to ensure a consistent 3D shape
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
        If `window_size <= 0`, returns the original array.

    Raises
    ------
    ValueError
        If `window_size` is not a divisor of the length of the last axis.
    """
    if window_size <= 0:
        return arr.reshape(arr.shape[:-1] + (1, -1))
    return arr.reshape(arr.shape[:-1] + (-1, window_size))


def get_calvano_collusion_index(rewards: np.ndarray, nash_reward: float, coll_reward: float, window_size: int = 0) -> np.ndarray:
    """
    Compute the Calvano Collusion Index (CCI) from agent rewards.

    The CCI measures the degree of collusion relative to Nash equilibrium
    and perfect collusion benchmarks. Rewards are optionally aggregated
    over fixed-size windows before computing the index.

    Parameters
    ----------
    rewards : np.ndarray
        Array of shape (n_agents, n_episodes) containing per-agent rewards.
    nash_reward : float
        Benchmark reward under Nash equilibrium (total across all agents).
    coll_reward : float
        Benchmark reward under perfect collusion (total across all agents).
    window_size : int, default=0
        Size of the episode window for reward aggregation.
        If 0, no windowing is applied.

    Returns
    -------
    : np.ndarray
        Array of CCI values per agent and per window.

    References
    --------
    - Calvano, E., Calzolari, G., Denicolò, V., & Pastorello, S. (2020).
    Artificial intelligence, algorithmic pricing, and collusion.
    *American Economic Review, 110*(10), 3267–3297.
    https://doi.org/10.1257/aer.20190623
    """
    nash_reward /= len(rewards)
    coll_reward /= len(rewards)

    rewards = split_array(rewards, window_size)
    avg_rewards = rewards.mean(axis=-1)

    cci = (avg_rewards - nash_reward) / (coll_reward - nash_reward)
    return cci