""" Game-theory utilities.
"""
import numpy as np

from itertools import product
from typing import Dict, List, Tuple



DECIMAL_PLACES = 3
"""
Number of decimal places to use when rounding numerical outputs, such as rewards.
"""
MAX_JOINT_ACTION_CACHE = 5
"""
Maximum number of cached joint action shapes to store in the internal `_joint_action_cache`.

If the cache exceeds this limit, it will be cleared to manage memory and maintain performance.
"""
TOL = 1e-08
"""
Numerical tolerance used for floating-point comparisons.
"""



_joint_action_cache: Dict[Tuple[int, ...], np.ndarray] = {}
"""
**INTERNAL USE ONLY — DO NOT TOUCH**

Internal cache mapping action space shapes to their corresponding precomputed joint action arrays.

Used by `_get_joint_actions` to avoid recomputation when checking equilibrium conditions across
strategy profiles with the same action space. Do not modify this object directly.
Doing so may result in invalid equilibrium checks or silent computation errors.
"""



def _get_joint_actions(shape: Tuple[int, ...]) -> np.ndarray:
    """
    Get or compute the full joint action space for a game with the given action space shape.

    If the shape has been queried before and the result is cached, the cached result is returned.
    Otherwise, the full Cartesian product is computed and stored in the internal cache.
    If the cache exceeds MAX_JOINT_ACTION_CACHE entries, it is cleared.

    Parameters
    ----------
    shape : tuple of int
        A tuple specifying the number of actions for each player.

    Returns
    -------
    : np.ndarray
        An array of shape (num_joint_actions, n_players), where each row is a joint action tuple.
    """
    global _joint_action_cache

    if shape not in _joint_action_cache:
        if len(_joint_action_cache) >= MAX_JOINT_ACTION_CACHE:
            _joint_action_cache.clear()
        _joint_action_cache[shape] = np.array(list(product(*[range(s) for s in shape])))
    
    return _joint_action_cache[shape]



def compute_joint_actions_and_rewards(action_spaces: List[np.ndarray], true_value: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the joint action space and resulting rewards for each maker in a trading scenario.

    For each joint action, the function:
    - Determines whether the trader prefers to buy, sell, or is indifferent (random choice).
    - Calculates which makers are selected under each operation type.
    - Computes the average number of times each maker is selected.
    - Computes the reward for each maker based on the trade and the average selection count.

    Parameters
    ----------
    action_spaces : list of np.ndarray
        List of length `n_agents`, where each array is of shape (n_actions, 2), representing
        the ask and bid prices set by each maker for every action.
    true_value : float
        The true value of the traded asset, used by the trader to decide whether to buy or sell.

    Returns
    -------
    all_combinations : np.ndarray
        The full joint action space, reshaped to match the input action_spaces plus maker dimensions.
        Shape: (*[len(e) for e in action_spaces], 2, n_agents)
    rewards : np.ndarray
        The reward assigned to each maker under every joint action, shaped similarly to `all_combinations`
        with the final dimension indicating the player.
        Shape: (*[len(e) for e in action_spaces], n_agents)
    """
    n_agents = len(action_spaces)

    all_combinations = np.array(list(product(*action_spaces)))
    all_combinations = np.transpose(all_combinations, (0, 2, 1))

    min_ask_prices = np.min(all_combinations[:, 0], axis=-1)
    max_bid_prices = np.max(all_combinations[:, 1], axis=-1)

    # What the trader will do: buy (0), sell (1) or choose random if equally convenient (2)
    selected_operation = np.zeros(len(all_combinations), dtype=int)
    selected_operation[
        ~np.isclose(
            (true_value - min_ask_prices),
            (max_bid_prices - true_value),
            atol = TOL
        ) & (
            (true_value - min_ask_prices) < (max_bid_prices - true_value)
    )] = 1
    selected_operation[np.isclose((true_value - min_ask_prices), (max_bid_prices - true_value), atol=TOL)] = 2

    # Op. 0 and 1: makers selected when trader has a clear buy or sell preference
    selected_makers = np.where(
        (
            (selected_operation == 0)[:, None] & (all_combinations[:, 0, :] == min_ask_prices[:, None])
        ) | (
            (selected_operation == 1)[:, None] & (all_combinations[:, 1, :] == max_bid_prices[:, None])
        )
    )
    # Op. 2: makers that only offered the best ask (no matching best bid)
    selected_makers_ask_only = np.where(
        (selected_operation == 2)[:, None] & 
        (all_combinations[:, 0, :] == min_ask_prices[:, None]) &
        ~(all_combinations[:, 1, :] == max_bid_prices[:, None])
    )
    # Op. 2: Makers that only offered the best bid (no matching best ask)
    selected_makers_bid_only = np.where(
        (selected_operation == 2)[:, None] &
        ~((all_combinations[:, 0, :] == min_ask_prices[:, None])) &
        (all_combinations[:, 1, :] == max_bid_prices[:, None])
    )
    # Op. 2: makers that offered both the best ask and the best bid
    selected_makers_both = np.where(
        (selected_operation == 2)[:, None] & 
        ((all_combinations[:, 0, :] == min_ask_prices[:, None]) &
         (all_combinations[:, 1, :] == max_bid_prices[:, None]))
    )

    # Count selected makers
    count_selected_ask_makers = np.sum(all_combinations[:, 0, :] == min_ask_prices[:, None], axis=1)
    count_selected_bid_makers = np.sum(all_combinations[:, 1, :] == max_bid_prices[:, None], axis=1)

    # Initialize array to store the average number of selected makers
    avg_selected_makers = np.zeros((len(all_combinations), n_agents))

    # Case 1: makers selected when trader had a clear preference (buy or sell)
    avg_selected_makers[*selected_makers] = np.where(
        selected_operation == 0,
        count_selected_ask_makers,
        count_selected_bid_makers
    )[selected_makers[0]]
    # Case 2: makers that had only a convenient ask price (no matching bid)
    # these are selected only when the trader chooses to buy (50% chance), so scale their count by 0.5
    avg_selected_makers[*selected_makers_ask_only] = count_selected_ask_makers[selected_makers_ask_only[0]] * 2
    # Case 3: makers that had only a convenient bid price
    # selected only when the trader chooses to sell (50% chance)
    avg_selected_makers[*selected_makers_bid_only] = count_selected_bid_makers[selected_makers_bid_only[0]] * 2
    # Case 4: Makers that had both a convenient ask and bid
    # Regardless of the trader's choice, they will always be selected, so take the avg of ask and bid selection counts
    avg_selected_makers[*selected_makers_both] = (
        count_selected_ask_makers[selected_makers_both[0]] / 2 + 
        count_selected_bid_makers[selected_makers_both[0]] / 2
    )

    # Compute trader reward
    reward = np.where(
        selected_operation == 0,
        true_value - min_ask_prices,
        max_bid_prices - true_value
    )[:, None]

    # Compute makers rewards
    rewards = np.round(
        np.divide(
            -reward,
            avg_selected_makers,
            out = np.zeros_like(avg_selected_makers),
            where = ~np.isclose(avg_selected_makers, 0, atol=TOL)
        ), 
        decimals = DECIMAL_PLACES
    )

    # Reshape
    newshape = [len(e) for e in action_spaces] 
    all_combinations = np.reshape(all_combinations, newshape + [2, n_agents])
    rewards = np.reshape(rewards, newshape + [n_agents,])
    return all_combinations, rewards


def is_cce(payoffs: np.ndarray, strategy_profile: np.ndarray, verbose: bool = False, fast: bool = True) -> bool:
    """
    Check whether a given strategy profile is a Coarse Correlated Equilibrium (CCE).

    A profile is a CCE if no player can improve their expected payoff by unilaterally deviating
    to a fixed pure strategy, assuming they only observe the distribution over joint actions.

    Parameters
    ----------
    payoffs : np.ndarray
        Array of shape (*action_spaces, n_players), giving the payoff to each player for every joint action.
    strategy_profile : np.ndarray
        A joint distribution over all joint actions. Must sum to 1. Shape should match `payoffs.shape[:-1]`.
    verbose : bool, default=False
        If True, prints a message for each player that has an incentive to deviate.
    fast : bool, default=True
        If True, returns immediately when a deviation is found. Otherwise, checks all players before returning.

    Returns
    -------
    : bool
        True if the given strategy profile is a coarse correlated equilibrium, False otherwise.
    
    See Also
    --------
    - Barman, S., & Ligett, K. (2015). Finding any nontrivial coarse correlated equilibrium is hard.
    ACM SIGecom Exchanges, 14(1), 76-79.
    """
    shape = payoffs.shape[:-1]
    n_players = payoffs.shape[-1]

    # Flatten for vectorized computation
    payoffs_flat = payoffs.reshape(-1, n_players)
    probs_flat = strategy_profile.flatten()
    probs_flat /= probs_flat.sum()  # Normalize

    # Fetch or compute joint actions
    joint_actions = _get_joint_actions(shape)

    flag = True
    for i in range(n_players):
        n_actions_i = shape[i]
        for a_i_prime in range(n_actions_i):
            # Deviation: force player i to use a_i_prime
            deviated_actions = joint_actions.copy()
            deviated_actions[:, i] = a_i_prime

            # Convert multi-index to flat index
            indices_original = np.ravel_multi_index(joint_actions.T, dims=shape)
            indices_deviated = np.ravel_multi_index(deviated_actions.T, dims=shape)

            # Expected payoff when following recommendation and when deviating to a_i_prime
            lhs = np.sum(probs_flat * payoffs_flat[indices_original, i])
            rhs = np.sum(probs_flat * payoffs_flat[indices_deviated, i])

            if lhs + TOL < rhs:
                if verbose:
                    print(f'Player {i} has incentive to deviate to {a_i_prime}: {lhs:.4f} < {rhs:.4f}')
                if fast:
                    return False
                else:
                    flag = False
    return flag


def is_ce(payoffs: np.ndarray, strategy_profile: np.ndarray, verbose: bool = False, fast: bool = True) -> bool:
    """
    Check whether a given joint strategy profile is a Correlated Equilibrium (CE).

    A profile is a CE if, for each player and each pair of actions (a_i, a_i′),
    the player has no incentive to deviate from the recommended action a_i to a_i′,
    assuming all other players follow their recommendations.

    Parameters
    ----------
    payoffs : np.ndarray
        Array of shape (*action_spaces, n_players), giving the payoff to each player for each joint action.
    strategy_profile : np.ndarray
        Joint distribution over all joint actions. Must sum to 1. Shape should match payoffs.shape[:-1].
    verbose : bool, default=False
        If True, prints information when a profitable deviation is detected.
    fast : bool, default=True
        If True, returns immediately after the first violation. If False, checks all pairs.

    Returns
    -------
    : bool
        True if the given strategy profile satisfies the conditions of a correlated equilibrium.
    
    See Also
    --------
    - Aumann, R. J. (1974). Subjectivity and correlation in randomized strategies.
    Journal of mathematical Economics, 1(1), 67-96.
    """
    shape = payoffs.shape[:-1]
    n_players = payoffs.shape[-1]

    # Normalize just in case
    probs = strategy_profile.reshape(-1)
    probs /= probs.sum()
    payoffs_flat = payoffs.reshape(-1, n_players)
    joint_actions = _get_joint_actions(shape)

    flag = True
    for i in range(n_players):
        n_actions = shape[i]

        for a_i in range(n_actions):
            # Identify indices where player i is recommended action a_i
            mask = joint_actions[:, i] == a_i
            if not np.any(mask):
                continue

            for a_i_prime in range(n_actions):
                if a_i == a_i_prime:
                    continue

                # Copy joint actions, change player i's action to a_i_prime
                deviated_actions = joint_actions[mask].copy()
                deviated_actions[:, i] = a_i_prime

                idx_original = np.ravel_multi_index(joint_actions[mask].T, shape)
                idx_deviated = np.ravel_multi_index(deviated_actions.T, shape)

                lhs = np.sum(probs[idx_original] * payoffs_flat[idx_original, i])
                rhs = np.sum(probs[idx_original] * payoffs_flat[idx_deviated, i])

                if lhs + TOL < rhs:
                    if verbose:
                        print(f"Player {i} prefers to deviate from {a_i} → {a_i_prime}: {lhs:.4f} < {rhs:.4f}")
                    if fast:
                        return False
                    flag = False

    return flag


def is_ne(payoffs: np.ndarray, strategies: List[np.ndarray], verbose: bool = False, fast: bool = True) -> bool:
    """
    Check whether a given mixed strategy profile is a Nash Equilibrium.

    A profile is a Nash Equilibrium if no player can improve their expected payoff by deviating
    to any pure strategy, assuming all players act independently according to their mixed strategies.

    Parameters
    ----------
    payoffs : np.ndarray
        Array of shape (*action_spaces, n_players), giving the payoff to each player for every joint action.
    strategies : list of np.ndarray
        List of length `n_players`, where each array contains the mixed strategy of a player and sums to 1.
    verbose : bool, default=False
        If True, prints a message for each player that has an incentive to deviate.
    fast : bool, default=True
        If True, returns immediately when a deviation is found. Otherwise, checks all players before returning.

    Returns
    -------
    : bool
        True if the given strategy profile is a Nash Equilibrium, False otherwise.
    
    See Also
    --------
    - Nash, J. F. (2024). Non-cooperative games.
    In The Foundations of Price Theory Vol 4 (pp. 329-340). Routledge.
    """
    n_players = len(strategies)
    shape = payoffs.shape[:-1]

    # Build full joint probability distribution from independent strategies
    grid = np.meshgrid(*[np.arange(s) for s in shape], indexing='ij')
    joint_prob = np.ones(shape)
    for i, strat in enumerate(strategies):
        joint_prob *= strat[grid[i]]

    flag = True
    for i in range(n_players):
        # Compute expected payoff for current strategy of player i
        expected_payoff = np.sum(joint_prob * payoffs[..., i])

        for a_i_prime in range(shape[i]):
            # Construct strategy where player i always plays a_i_prime
            grid_prime = list(grid)
            grid_prime[i] = np.full_like(grid[i], a_i_prime)  # fix to a_i_prime

            joint_prob_prime = np.ones_like(joint_prob)
            for j, strat in enumerate(strategies):
                if j == i:
                    joint_prob_prime *= 1  # deterministic
                else:
                    joint_prob_prime *= strat[grid_prime[j]]

            expected_deviation_payoff = np.sum(joint_prob_prime * payoffs[..., i])

            if expected_payoff + TOL < expected_deviation_payoff:
                if verbose:
                    print(f'Player {i} can profitably deviate to pure action {a_i_prime}: '
                          f'{expected_payoff:.4f} < {expected_deviation_payoff:.4f}')
                if fast:
                    return False
                else:
                    flag = False
    return flag
