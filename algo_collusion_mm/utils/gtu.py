""" Game-theory utilities.
"""
import cvxpy as cp
import numpy as np

from itertools import product
from typing import Dict, List, Literal, Tuple



DECIMAL_PLACES = 3
"""
Number of decimal places to use when rounding numerical outputs, such as rewards.
"""
MAX_JOINT_ACTIONS_CACHE_SIZE = 5
"""
Maximum number of cached joint action shapes to store in the internal `_joint_actions_cache`.

If the cache exceeds this limit, it will be cleared to manage memory and maintain performance.
"""
TOL = 1e-08
"""
Numerical tolerance used for floating-point comparisons.
"""



_joint_actions_cache: Dict[Tuple[int, ...], np.ndarray] = {}
"""
**INTERNAL USE ONLY — DO NOT TOUCH**

Internal cache mapping action space shapes to their corresponding precomputed joint actions arrays.

Used by `_get_joint_actions` to avoid recomputation when checking equilibrium conditions across
strategy profiles with the same action space. Do not modify this object directly.
Doing so may result in invalid equilibrium checks or silent computation errors.
"""



def _get_joint_actions(shape: Tuple[int, ...]) -> np.ndarray:
    """
    Get or compute the full joint action space for a game with the given action space shapes.

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
    global _joint_actions_cache

    if shape not in _joint_actions_cache:
        if len(_joint_actions_cache) >= MAX_JOINT_ACTIONS_CACHE_SIZE:
            _joint_actions_cache.clear()
        _joint_actions_cache[shape] = np.array(list(product(*[range(s) for s in shape])))
    
    return _joint_actions_cache[shape]



def compute_joint_actions_and_rewards(
    action_spaces: List[np.ndarray],
    true_value: float,
    tie_breaker: Literal['buy', 'sell', 'rand'] = 'rand'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the joint action space and resulting rewards for each maker in a trading scenario.

    Parameters
    ----------
    action_spaces : list of np.ndarray
        List of length `n_agents`, where each array is of shape (n_actions, 2), representing
        the ask and bid prices set by each maker for every action.
    true_value : float
        The true value of the traded asset, used by the trader to decide whether to buy or sell.
    tie_breaker : {'buy', 'sell', 'rand', 'alt'}
        Rule used to resolve ties between equally favorable prices:
        - 'buy': always prefer buying in case of tie.
        - 'sell': always prefer selling in case of tie.
        - 'rand': break ties randomly (50/50).

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
    all_combinations = np.transpose(all_combinations, (0, 2, 1))  # -> (_, 2, n_agents)

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

    if tie_breaker == 'rand':
        selected_operation[np.isclose((true_value - min_ask_prices), (max_bid_prices - true_value), atol=TOL)] = 2
    elif tie_breaker == 'sell':
        selected_operation[np.isclose((true_value - min_ask_prices), (max_bid_prices - true_value), atol=TOL)] = 1

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


def find_best_cce(
    payoffs: np.ndarray,
    objective: Literal['social_welfare', 'single_agent'],
    target_player: int | None = None,
    solver: str = cp.CLARABEL
) -> Tuple[np.ndarray, float]:
    """
    Compute a coarse correlated equilibrium (CCE) that maximizes a chosen objective.

    A CCE is a joint probability distribution over all players' actions such that 
    no player can increase their expected payoff by unilaterally deviating to any 
    fixed pure strategy. This function formulates and solves the corresponding 
    optimization problem as a linear program using CVXPY.

    The objective can be set to either:
    - social welfare — maximize the sum of expected payoffs for all players.
    - single agent — maximize the expected payoff of one specified player.

    Parameters
    ----------
    payoffs : np.ndarray
        An array of shape `(*action_spaces, n_players)` specifying the payoff for 
        each player at every possible joint action. The last axis corresponds to players, 
        while each preceding axis corresponds to a player's discrete action space.
        Example: for a 2-player game where player 1 has 3 actions and player 2 has 4 actions,
        the shape would be (3, 4, 2).
    objective : {'social_welfare', 'single_agent'}
        The optimization objective to maximize:
        - `'social_welfare'`: sum of expected payoffs of all players.
        - `'single_agent'`: expected payoff of a single player (requires `target_player`).
    target_player : int or None, default=None
        Index of the player whose payoff should be maximized when 
        `objective='single_agent'`. Must be provided in that case.
    solver : str, default=cp.CLARABEL
        The solver to use for the underlying CVXPY optimization problem.  
        You can view the list of solvers available in your environment by running:
        `cvxpy.installed_solvers()`.

    Returns
    -------
    p_opt : np.ndarray
        The optimal joint probability distribution over all action profiles, 
        with shape `(*action_spaces,)`. The probabilities sum to 1.
    value_opt : float
        The optimal value of the chosen objective under the computed CCE.

    Raises
    ------
    ValueError
        - If `objective='single_agent'` but `target_player` is not provided.
        - If `objective` is not one of `'social_welfare'` or `'single_agent'`.
        - If no optimal solution (CCE) can be found by the solver.

    Notes
    -----
    - This function uses linear programming via CVXPY.
    - The chosen solver can influence performance and numerical stability.
      If you encounter solver issues, try switching to another available solver.
    """
    if objective == 'single_agent' and target_player is None:
        raise ValueError('target_player must be provided when objective is "single_agent"')
    if objective not in ['social_welfare', 'single_agent']:
        raise ValueError(f'Invalid objective: {objective}. Must be "social_welfare" or "single_agent".')

    payoffs = np.moveaxis(payoffs, -1, 0)  # -> (n_players, *action_spaces)
    n_players = payoffs.shape[0]
    shape = payoffs.shape[1:]
    n_joint = np.prod(shape)

    # Flatten payoffs for vectorized computation
    payoffs_flat = payoffs.reshape(n_players, n_joint)  # -> (n_players, n_joint)

    # All joint actions as flat indices
    joint_actions = np.array(list(np.ndindex(shape)))  # -> (n_joint, n_players)
    
    # Decision variable: probability distribution over joint actions
    p = cp.Variable(n_joint, nonneg=True)

    # Constraint: probabilities sum to 1
    constraints = [cp.sum(p) == 1]

    # Vectorized CCE constraints
    for i in range(n_players):
        n_actions_i = shape[i]
        for a_i_prime in range(n_actions_i):
            deviated_actions = joint_actions.copy()
            deviated_actions[:, i] = a_i_prime

            # Convert multi-indices to flat indices
            idx_original = np.ravel_multi_index(joint_actions.T, dims=shape)
            idx_deviated = np.ravel_multi_index(deviated_actions.T, dims=shape)

            lhs = payoffs_flat[i, idx_original] @ p
            rhs = payoffs_flat[i, idx_deviated] @ p
            constraints.append(lhs >= rhs)

    # Objective function
    if objective == 'social_welfare':
        total_payoffs = payoffs_flat.sum(axis=0)
        objective_expr = cp.Maximize(total_payoffs @ p)
    else:
        objective_expr = cp.Maximize(payoffs_flat[target_player] @ p)

    # Solve the linear program
    problem = cp.Problem(objective_expr, constraints)
    problem.solve(solver=solver)

    if problem.status not in ['optimal', 'optimal_inaccurate']:
        raise ValueError('No optimal CCE could be found by the solver.')

    # Reshape probabilities to match action dimensions
    p_opt = np.array(p.value).reshape(shape)
    return p_opt, problem.value


def is_cce(payoffs: np.ndarray, strategy_profile: np.ndarray, verbose: bool = False, fast: bool = True) -> bool:
    """
    Check whether a given strategy profile is a Coarse Correlated Equilibrium (CCE).

    A profile is a CCE if no player can improve their expected payoff by unilaterally deviating
    to a fixed pure strategy, assuming they only observe the distribution over joint actions.

    Parameters
    ----------
    payoffs : np.ndarray
        An array of shape (*action_spaces, n_players) specifying the payoff for each player at every
        possible joint action. Here, `*action_spaces` means one axis per player's action set
        (e.g., if there are 3 players with 2, 3, and 2 actions respectively, the shape would be (2, 3, 2, n_players)).
    strategy_profile : np.ndarray
        A probability distribution over joint actions, of shape `(*action_spaces,)`.
        Each entry gives the probability of that joint action being played. The distribution must sum to 1.
    verbose : bool, default=False
        If True, prints a message for each player that has an incentive to deviate.
    fast : bool, default=True
        If True, returns immediately when a deviation is found. Otherwise, checks all players before returning.

    Returns
    -------
    : bool
        True if the given strategy profile is a coarse correlated equilibrium, False otherwise.
    
    References
    ----------
    - Barman, S., & Ligett, K. (2015). Finding any nontrivial coarse correlated equilibrium is hard.
    ACM SIGecom Exchanges, 14(1), 76-79.
    """
    shape = payoffs.shape[:-1]
    n_players = payoffs.shape[-1]

    # Flatten for vectorized computation
    payoffs_flat = payoffs.reshape(-1, n_players)
    probs_flat = strategy_profile.flatten()
    # Normalize (just in case)
    probs_flat /= probs_flat.sum()

    # Fetch or compute joint actions
    joint_actions = _get_joint_actions(shape)  # -> (n_joint_actions, n_players)

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

    A profile is a CE if, for each player and each pair of actions (a_i, a_i'),
    the player has no incentive to deviate from the recommended action a_i to a_i',
    assuming all other players follow their recommendations.

    Parameters
    ----------
    payoffs : np.ndarray
        An array of shape (*action_spaces, n_players) specifying the payoff for each player at every
        possible joint action. Here, `*action_spaces` means one axis per player's action set
        (e.g., if there are 3 players with 2, 3, and 2 actions respectively, the shape would be (2, 3, 2, n_players)).
    strategy_profile : np.ndarray
        A probability distribution over joint actions, of shape `(*action_spaces,)`.
        Each entry gives the probability of that joint action being played. The distribution must sum to 1.
    verbose : bool, default=False
        If True, prints information when a profitable deviation is detected.
    fast : bool, default=True
        If True, returns immediately after the first violation. If False, checks all pairs.

    Returns
    -------
    : bool
        True if the given strategy profile satisfies the conditions of a correlated equilibrium.
    
    References
    ----------
    - Aumann, R. J. (1974). Subjectivity and correlation in randomized strategies.
    Journal of mathematical Economics, 1(1), 67-96.
    """
    shape = payoffs.shape[:-1]
    n_players = payoffs.shape[-1]

    # Flatten for vectorized computation
    payoffs_flat = payoffs.reshape(-1, n_players)
    probs_flat = strategy_profile.reshape(-1)
    # Normalize (just in case)
    probs_flat /= probs_flat.sum()
    
    # Fetch or compute joint actions
    joint_actions = _get_joint_actions(shape)  # -> (n_joint_actions, n_players)

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

                # Convert multi-index to flat index
                idx_original = np.ravel_multi_index(joint_actions[mask].T, shape)
                idx_deviated = np.ravel_multi_index(deviated_actions.T, shape)

                # Expected payoff when following recommendation and when deviating to a_i_prime
                lhs = np.sum(probs_flat[idx_original] * payoffs_flat[idx_original, i])
                rhs = np.sum(probs_flat[idx_original] * payoffs_flat[idx_deviated, i])

                if lhs + TOL < rhs:
                    if verbose:
                        print(f"Player {i} prefers to deviate from {a_i} → {a_i_prime}: {lhs:.4f} < {rhs:.4f}")
                    if fast:
                        return False
                    flag = False
    return flag


def is_ne(payoffs: np.ndarray, strategies: np.ndarray, verbose: bool = False, fast: bool = True) -> bool:
    """
    Check whether a given strategy profile (product of mixed strategies) is a Nash Equilibrium.

    Parameters
    ----------
    payoffs : np.ndarray
        An array of shape (*action_spaces, n_players) specifying the payoff for each player at every
        possible joint action. Here, `*action_spaces` means one axis per player's action set
        (e.g., if there are 3 players with 2, 3, and 2 actions respectively, the shape would be (2, 3, 2, n_players)).
    strategies : np.ndarray
        Array of shape `(n_players, max_actions_i)`, where each row gives a player's mixed strategy
        over their available actions. If players have different numbers of actions, unused entries
        (beyond each player's action count) should be zero. Each player's strategy must sum to 1.
    verbose : bool, default=False
        If True, prints information when a profitable deviation is detected.
    fast : bool, default=True
        If True, returns immediately after the first violation. If False, checks all players and actions.

    Returns
    -------
    : bool
        True if the given strategy profile is a Nash Equilibrium.
    
    References
    ----------
    - Nash, J. F. (2024). Non-cooperative games.
    In The Foundations of Price Theory Vol 4 (pp. 329-340). Routledge.
    """
    shape = payoffs.shape[:-1]
    n_players = payoffs.shape[-1]

    # Flatten for vectorized computation
    payoffs_flat = payoffs.reshape(-1, n_players)
    # Normalize each player's strategy (just in case)
    strategies /= strategies.sum(axis=1, keepdims=True)

    # Fetch or compute joint actions
    joint_actions = _get_joint_actions(shape)  # -> (n_joint_actions, n_players)
    n_joint_actions = joint_actions.shape[0]

    # Compute the probability of each joint action as product of marginal probs
    joint_probs = np.ones(n_joint_actions)
    for i in range(n_players):
        joint_probs *= strategies[i, joint_actions[:, i]]

    flag = True
    for i in range(n_players):
        n_actions = shape[i]
        for a_i_prime in range(n_actions):
            # Compute the deviated joint distribution where player i unilaterally plays a_i_prime
            deviated_joint_probs = np.ones(n_joint_actions)
            for j in range(n_players):
                if j == i:
                    deviated_joint_probs *= (joint_actions[:, j] == a_i_prime).astype(float)
                else:
                    deviated_joint_probs *= strategies[j, joint_actions[:, j]]

            # Expected payoff when following recommendation and when deviating to a_i_prime
            expected_payoff = np.sum(joint_probs * payoffs_flat[:, i])
            deviated_expected_payoff = np.sum(deviated_joint_probs * payoffs_flat[:, i])

            if expected_payoff + TOL < deviated_expected_payoff:
                if verbose:
                    print(f'Player {i} has incentive to deviate to {a_i_prime}: {expected_payoff:.4f} < {deviated_expected_payoff:.4f}')
                if fast:
                    return False
                flag = False
    return flag
