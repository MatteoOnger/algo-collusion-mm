import matplotlib.pyplot as plt
import numpy as np
import os
import time

from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from datetime import datetime
from functools import reduce
from typing import Any, Callable, Dict, List, Tuple, Union

import algo_collusion_mm.utils.gtu as gtu
import algo_collusion_mm.utils.plots as plots

from algo_collusion_mm.agents import Agent, Maker, Trader
from algo_collusion_mm.envs import CGMEnv
from algo_collusion_mm.utils.common import get_calvano_collusion_index, get_relative_deviation_competition
from algo_collusion_mm.utils.stats import OnlineVectorStats
from algo_collusion_mm.utils.storage import ExperimentStorage



def _merge_dicts(d1: Dict[str, Any], d2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries.

    For each key:
    - If both values are dicts, merge them recursively.
    - If both values exist but are not dicts, wrap the pair into a list `[v1, v2]`.
    - If the key exists only in `d2`, deep-copy its value.

    Parameters
    ----------
    d1 : dict of str to any
        Base dictionary.
    d2 : dict of str to any
        Dictionary whose values override or merge into `d1`.

    Returns
    -------
    : dict of str to any
        A new dictionary representing the merged result.
    """
    result = deepcopy(d1)
    for k, v2 in d2.items():
        if k in result:
            v1 = result[k]
            if isinstance(v1, dict) and isinstance(v2, dict):
                result[k] = _merge_dicts(v1, v2)
            else:
                result[k] = [v1, v2]
        else:
            result[k] = deepcopy(v2)
    return result


def _merge_dict_lists(dlists: List[Union[List[Dict[str, Any]], Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Merge two items that may each be either a list of dicts or a single dict.

    Rules:
    - If both items are lists, merge dict pairs elementwise.
    - If one is a list and the other a dict, merge the dict into each element of the list.
    - If both are dicts, return a 2-element list containing them.

    Parameters
    ----------
    dlists : list
        A length-2 list containing two elements, each either a dict or a list of dicts.

    Returns
    -------
    : list of dict
        The merged list of dictionaries.
    """
    assert len(dlists) == 2

    l1, l2 = dlists
    
    if isinstance(l1, list) and isinstance(l2, list):
        merged = []
        for d1, d2 in zip(l1, l2):
            merged.append(_merge_dicts(d1, d2))
        return merged

    if isinstance(l1, list) and isinstance(l2, dict):
        return [_merge_dicts(d, deepcopy(l2)) for d in l1]

    if isinstance(l1, dict) and isinstance(l2, list):
        return [_merge_dicts(deepcopy(l1), d) for d in l2]

    return [l1, l2]


def _multiple_runs(
    saver_base_path: str,
    generate_vt: Callable[[], float],
    info_level: str,
    n_arms: int,
    n_makers: int,
    n_traders: int,
    agents_params: Dict[str, Any],
    nash_reward: float,
    coll_reward: float,
    decimal_places: int = 3,
    n_episodes: int = 100,
    n_rounds: int = 10_000,
    n_windows: int = 100,
    run_id: int = -1
) -> Tuple[OnlineVectorStats, OnlineVectorStats, OnlineVectorStats, OnlineVectorStats, OnlineVectorStats, OnlineVectorStats, bool, bool]:
    """
    Execute multiple training episodes for a single experiment configuration
    and collect statistics over all episodes.

    Parameters
    ----------
    saver_base_path : str
        Path where experiment logs, JSON files, and generated plots are saved.
    generate_vt : callable
        Function returning the value of the VT signal used by the environment.
    info_level : str
        Information level passed to the environment (e.g. 'full', 'partial').
    n_arms : int
        Number of available actions for each maker.
    n_makers : int
        Number of maker agents.
    n_traders : int
        Number of trader agents.
    agents_params : dict of str to any
        Parameter dictionary specifying agent classes and their configuration.
    nash_reward : float
        Reward corresponding to the competitive (Nash) outcome.
    coll_reward : float
        Reward corresponding to perfect collusion.
    decimal_places : int, default=3
        Number of decimals to use when rounding logged metrics.
    n_episodes : int, default=100
        Number of episodes to simulate.
    n_rounds : int, default=10000
        Number of rounds per episode.
    n_windows : int, default=100
        Number of windows used to compute windowed statistics.
    run_id : int, default=-1
        Experiment identifier used for logging and plot titles.

    Returns
    -------
    : tuple of OnlineVectorStats and bool
        A tuple of six OnlineVectorStats objects containing summary statistics for:
        - (OnlineVectorStats) CCI values;
        - (OnlineVectorStats) Sorted CCI values;
        - (OnlineVectorStats) RDC values;
        - (OnlineVectorStats) Action frequencies;
        - (OnlineVectorStats) Joint action frequencies;
        - (OnlineVectorStats) Action values.
        - (bool) True if the average strategy profile is a CCE.
        - (bool) True if the average strategy is a NE.

    Notes
    -----
    All agents *within the same episode* must share the **same `action_space`**.
    """
    window_size = n_rounds // n_windows

    # Extract meta parameters
    maker_classes = [maker_param['meta']['class'] for maker_param in agents_params['maker']]
    maker_prefixes= [maker_param['meta']['prefix'] for maker_param in agents_params['maker']]

    trader_classes = [trader_param['meta']['class'] for trader_param in agents_params['trader']]
    trader_prefixes = [maker_param['meta']['prefix'] for maker_param in agents_params['trader']]

    # Replace stings with lambdas
    for i in range(n_makers):
        if 'scale_rewards' in  agents_params['maker'][i]['params']:
            agents_params['maker'][i]['params']['scale_rewards'] = eval(agents_params['maker'][i]['params']['scale_rewards'])

    # To compute online statistics
    stats_cci = OnlineVectorStats((n_makers, n_windows))
    stats_sorted_cci = OnlineVectorStats((n_makers, n_windows))
    stats_rdc = OnlineVectorStats((n_makers, n_windows))
    stats_action_freq = OnlineVectorStats((n_makers, 2, n_arms))
    stats_joint_action_freq = OnlineVectorStats((2,) + n_makers * (n_arms,))
    stats_action_values = OnlineVectorStats((n_makers, n_arms))
    stats_rwd = OnlineVectorStats(n_makers)

    # To save experimental results
    saver = ExperimentStorage(saver_base_path)
    saver.print_and_save(f'Started at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', silent=True)

    start_time = time.time()
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    saver.print_and_save(f'Started at {current_time}', silent=True)

    # Create agents
    makers: List[Maker] = [
        maker_classes[i](
            name = f'{maker_prefixes[i]}_{i}',
            **agents_params['maker'][i]['params'],
        ) for i in range(n_makers)
    ] 
    traders: List[Trader] = [
        trader_classes[i](
            name = f'{trader_prefixes[i]}_{i}',
            **agents_params['trader'][i]['params'],
        ) for i in range(n_traders)
    ]
    agents: Dict[str, Agent] = {agent.name: agent for agent in makers + traders}

    # Create env
    env = CGMEnv(
        generate_vt = eval(generate_vt),
        n_rounds = n_rounds,
        makers = makers,
        traders = traders,
        decimal_places = decimal_places,
        info_level = info_level
    )

    for i in range(n_episodes):
        if i % 10 == 0:
            saver.print_and_save(f'Running episode {i} ...', silent=True)

        # Reset env
        _, info = env.reset()

        # Reset the agent and change seed
        for agent in agents.values():
            agent.reset()
            agent.update_seed()

        # Play one episode
        for next_player in env.agent_iter():
            action = next_player.act(env.observe(next_player))
            _, rewards, _, _, infos = env.step(action)
            if infos['round_finished']:
                for agent in env.possible_agents:
                    agent.update(rewards[agent.name], infos[agent.name])

        # Compute calvano collusion idex per window and agent
        cci = get_calvano_collusion_index(
            np.array([maker.history.get_rewards() for maker in makers]),
            nash_reward = nash_reward,
            coll_reward = coll_reward,
            window_size = window_size,
            decimal_places = decimal_places
        )

        # Compute relative deviation from competition index per window and agent
        rdc = get_relative_deviation_competition(
            np.array([maker.history.get_rewards() for maker in makers]),
            nash_reward = nash_reward,
            window_size = window_size,
            decimal_places = decimal_places
        )
        
        # Collect info
        info = {
            'parmas' : {
                'seed' : {name : agent._seed for name, agent in agents.items()},
            },
            'action_values':{
                n_windows : {maker.name : str(maker.action_values).replace('\n', '') for maker in makers}
            },
            'freq_actions' : {
                0 : {maker.name : str(maker.history.compute_freqs(slice(0, window_size))).replace('\n', '') for maker in makers},
                n_windows : {maker.name : str(maker.history.compute_freqs(slice(-window_size, None))).replace('\n', '') for maker in makers},
                'global' : {maker.name : str(maker.history.compute_freqs()).replace('\n', '') for maker in makers}
            },
            'most_common_action' : {
                0 : {maker.name : str(maker.history.compute_most_common(slice(0, window_size))) for maker in makers},
                n_windows : {maker.name : str(maker.history.compute_most_common(slice(-window_size, None))) for maker in makers},
                'global' : {maker.name : str(maker.history.compute_most_common()) for maker in makers}
            },
            'cumulative_rewards' : {
                0 : {name : round(float(agent.history.get_rewards(slice(0, window_size)).sum()), decimal_places) for name, agent in agents.items()},
                n_windows : {name : round(float(agent.history.get_rewards(slice(-window_size, None)).sum()), decimal_places) for name, agent in agents.items()},
                'global' : env.cumulative_rewards
            },
            'cci' : {
                0  : {maker.name : round(float(cci[idx, 0]), 3) for idx, maker in enumerate(makers)},
                n_windows  : {maker.name : round(float(cci[idx, -1]), 3) for idx, maker in enumerate(makers)},
                'global' : {maker.name : round(float(cci[idx, :].mean()), 3) for idx, maker in enumerate(makers)},
            },
            'rdc' : {
                0  : {maker.name : round(float(rdc[idx, 0]), 3) for idx, maker in enumerate(makers)},
                n_windows  : {maker.name : round(float(rdc[idx, -1]), 3) for idx, maker in enumerate(makers)},
                'global' : {maker.name : round(float(rdc[idx, :].mean()), 3) for idx, maker in enumerate(makers)},
            }
        }
        
        # Sort agents according to the CCI of the last window
        sorted_cci = cci[np.argsort(cci[:, -1])[::-1]]

        # Joint actions frequency first and last window
        matrix = np.zeros((2,) + n_makers * (n_arms,))

        joint_actions = np.array([
            maker.history.get_actions(slice(window_size), return_index=True) for maker in makers
        ]).T
        unique_joint_actions, freqs = np.unique(joint_actions, return_counts=True, axis=0)
        matrix[0][tuple(unique_joint_actions.T)] = freqs / window_size
        
        joint_actions = np.array([
            maker.history.get_actions(slice(-window_size, None), return_index=True) for maker in makers
        ]).T
        unique_joint_actions, freqs = np.unique(joint_actions, return_counts=True, axis=0)
        matrix[1][tuple(unique_joint_actions.T)] = freqs / window_size
        
        # Update statistics
        stats_cci.update(cci)
        stats_sorted_cci.update(sorted_cci)
        stats_rdc.update(rdc)
        stats_action_freq.update(np.array(
            [[
                maker.history.compute_freqs(slice(window_size)),
                maker.history.compute_freqs(slice(-window_size, None))
            ] for maker in makers]
        ) / window_size)
        stats_joint_action_freq.update(matrix)
        stats_action_values.update(np.array([maker.action_values for maker in makers]))
        stats_rwd.update(np.array([env.cumulative_rewards[maker.name] for maker in makers]))

        # Save and print results
        dir = saver.save_episode(info=info)
        saver.print_and_save(
            f'{(i+1):03} {'*' if (cci[:, -1] >= 0.45).any() else ' '} -> CCI:{info['cci'][n_windows]}'.ljust(60) + f' ({dir})',
            silent = True
        )
    
    end_time = time.time()
    execution_time = end_time - start_time
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    saver.print_and_save(f'Done at {current_time} | Execution time: {execution_time:.2f} seconds', silent=True)

    # Save plot
    title = maker_classes[0].__name__ if len(set(maker_classes)) == 1 else "Heterogeneous MMs"

    fig = plots.plot_all_stats(
        window_size = window_size,
        makers = makers,
        stats_cci = stats_cci,
        stats_sorted_cci = stats_sorted_cci,
        stats_rdc = stats_rdc,
        stats_actions_freq = stats_action_freq,
        stats_joint_actions_freq = stats_joint_action_freq,
        stats_action_values = stats_action_values,
        title = f'{title} - Makers Statistics Summary Plot - Experiment:{run_id + 1}'
    )
    saver.save_figures({f'PLOT': fig})

    # Save and print results
    exp_rwd = stats_cci.get_mean()[:, -1] * (coll_reward/n_makers - nash_reward/n_makers) + nash_reward/n_makers
    action_spaces = np.repeat(makers[0].action_space[None, :], repeats=n_makers, axis=0)

    _, rewards = gtu.compute_joint_actions_and_rewards(action_spaces, true_value=0.5, tie_breaker=traders[0].tie_breaker)
    is_cce = gtu.is_cce(rewards, stats_joint_action_freq.get_mean()[1], strict=False, fast=True, verbose=False)
    is_ne = gtu.is_ne(rewards,  stats_action_freq.get_mean()[:, 1, :], strict=False, fast=True, verbose=True)

    is_indip = np.isclose(
        stats_joint_action_freq.get_mean()[1],
        reduce(np.multiply.outer, stats_action_freq.get_mean()[:, 1, :]),
        atol = 1e-05
    ).all()

    saver.print_and_save(
        f'Results:\n'
        f'- Last window:\n'
        f' - [CCI] Average: {np.round(stats_cci.get_mean()[:, -1], 4)}\n'
        f' - [CCI] Minimum: {np.round(stats_cci.get_min()[:, -1], 4)}\n'
        f' - [CCI] Maximum: {np.round(stats_cci.get_max()[:, -1], 4)}\n'
        f' - [CCI] Standard deviation: {np.round(stats_cci.get_std(sample=False)[:, -1], 4)}\n'
        f' - [SORTED CCI] Average: {np.round(stats_sorted_cci.get_mean()[:, -1], 4)}\n'
        f' - [SORTED CCI] Minimum: {np.round(stats_sorted_cci.get_min()[:, -1], 4)}\n'
        f' - [SORTED CCI] Maximum: {np.round(stats_sorted_cci.get_max()[:, -1], 4)}\n'
        f' - [SORTED CCI] Standard deviation: {np.round(stats_sorted_cci.get_std(sample=False)[:, -1], 4)}\n'
        f' - [RDC] Average: {np.round(stats_rdc.get_mean()[:, -1], 4)}\n'
        f' - [RWD] Expected: {np.round(exp_rwd, 4)}\n'
        f' - [CCE] Is an equilibrium: {is_cce}\n'
        f' - [NE] Is an equilibrium: {is_ne}\n'
        f' - [NE] Is independent: {is_indip}\n' 
        f'- Global:\n'
        f' - [RWD] Average: {np.round(stats_rwd.get_mean(), 4)}\n'
        f' - [RWD] Standard deviation: {np.round(stats_rwd.get_std(sample=False), 4)}',
        silent = True
    )
    return stats_cci, stats_sorted_cci, stats_rdc, stats_action_freq, stats_joint_action_freq, stats_action_values, is_cce, is_ne


def _worker(
    run_id: int,
    params: List[Dict[str, Any]],
) -> Tuple[OnlineVectorStats, OnlineVectorStats, OnlineVectorStats, OnlineVectorStats, OnlineVectorStats, OnlineVectorStats, bool, bool]:
    """
    Worker function executed in a separate process for one experiment.

    - Logs start and end of the run.
    - Saves the parameter configuration for the run.
    - Executes `_multiple_runs` with the specified run parameters.

    Parameters
    ----------
    run_id : int
        Index of the experiment in the parameter list.
    params : list of dict of str to any
        Merged fixed and variable parameter configurations for all runs.

    Returns
    -------
    : tuple of OnlineVectorStats and bool
        The statistics tuple returned by `_multiple_runs`.
    """
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{now}] [Run {run_id}] Starting...', flush=True)

    saver = ExperimentStorage(params[run_id]['env']['saver_base_path'])
    saver.save_jsons({f'PARAMS': params[run_id]})
    del saver

    results = _multiple_runs(
        run_id = run_id,
        agents_params = params[run_id]['agent'],
        **params[run_id]['env']
    )

    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{now}] [Run {run_id}] Finished', flush=True)
    return results


def run_experiment_suite(
    fixed_params: Dict[str, Any],
    variable_params: List[Dict[str, Any]],
    base_path: str = os.path.join('.', 'experiments', 'default'),
    max_workers: int = 4,
) -> None:
    """
    Run a suite of experiments in parallel using fixed parameters and
    per-experiment variable parameters.

    Parameters
    ----------
    fixed_params : dict of str to any
        Parameter dictionary containing the settings that remain the same for
        every experiment in the suite.
    variable_params : list of dict of str to any
        A list of dictionaries, each representing the parameters that vary
        for a single experiment.
    base_path : str, default='./experiments/default'
        Directory where experiment results, logs, and plots are stored.
    max_workers : int, default=4
        Number of worker processes to use for parallel execution.

    Examples
    --------
    Running two experiments that differ only by the `epsilon` parameter of the
    maker agent:

    >>> action_space = np.array([[.6, .4], [1., 0.]])
    >>> epsilons = [0.001, 0.01]

    >>> fixed_params = {
    ...     'env': {
    ...         'generate_vt': 'lambda: 0.5',
    ...         'info_level': 'partial',
    ...         'n_arms': len(action_space),
    ...         'n_makers': 2,
    ...         'n_traders': 1,
    ...         'nash_reward': 0.1,
    ...         'coll_reward': 0.5,
    ...         'decimal_places': 3,
    ...         'n_episodes': 10,
    ...         'n_rounds': 1000,
    ...         'n_windows': 100,
    ...     },
    ...     'agent': {
    ...         'maker': {
    ...             'meta': {'prefix': 'maker_u', 'class': MakerEXP3},
    ...             'params': {
    ...                 'action_space': action_space,
    ...                 'scale_rewards': 'lambda x: x/0.5',
    ...             }
    ...         },
    ...         'trader': {
    ...             'meta': {'prefix': 'trader', 'class': NoPassTrader},
    ...             'params': {'tie_breaker': 'rand'}
    ...         }
    ...     }
    ... }

    >>> variable_params = [
    ...     {
    ...         'env': {},
    ...         'agent': {
    ...             'maker': {'meta': {}, 'params': {'epsilon': eps}},
    ...             'trader': {'meta': {}, 'params': {}},
    ...         }
    ...     }
    ...     for eps in epsilons
    ... ]

    >>> run_experiment_suite(fixed_params, variable_params)

    Notes
    -----
    The only requirement for agents *within the same episode* is that they
    **must all share the exact same `action_space`**.
    """
    n_parallel_runs = len(variable_params)

    # Merge fixed and variable parameters
    params = list()
    for i in range(n_parallel_runs):
        params.append(
            _merge_dicts(variable_params[i], fixed_params)
        )

        params[i]['env']['saver_base_path'] = os.path.join(base_path, f'experiment_{(i+1):03}')

        n_makers = params[i]['env']['n_makers']
        n_traders = params[i]['env']['n_traders']

        if isinstance(params[i]['agent']['maker'], list):
            params[i]['agent']['maker'] = _merge_dict_lists(params[i]['agent']['maker'])
        else:
            params[i]['agent']['maker'] = [deepcopy(params[i]['agent']['maker']) for _ in range(n_makers)]
            
        if isinstance(params[i]['agent']['trader'], list):
            params[i]['agent']['trader'] = _merge_dict_lists(params[i]['agent']['trader'])
        else:
            params[i]['agent']['trader'] = [deepcopy(params[i]['agent']['trader']) for _ in range(n_traders)]

        assert len(params[i]['agent']['maker']) == n_makers
        assert len(params[i]['agent']['trader']) == n_traders

    saver = ExperimentStorage(base_path)

    start_time = time.time()
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    saver.print_and_save(f'Started at {current_time}')

    # Run Parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_worker, run_id, params)
            for run_id in range(n_parallel_runs)
        ]
    results_list = [f.result() for f in futures]

    # Save results
    saver.save_objects({'results_list': results_list})

    # Print results
    for i, results in enumerate(results_list):
        n_makers = params[i]['env']['n_makers']
        action_space = params[i]['agent']['maker'][0]['params']['action_space']
        action_values_attrs = [params[i]['agent']['maker'][j]['params'].get('action_values_attr', 'default') for j in range(n_makers)]

        stats_cci, _, stats_rdc, stats_action_freq, stats_joint_action_freq, stats_action_values, is_cce, is_ne = results

        min_cci, mean_cci, max_cci = stats_cci.get_min(), stats_cci.get_mean(), stats_cci.get_max()
        std_cci = stats_cci.get_std(sample=False)

        mean_rdc = stats_rdc.get_mean()

        mean_action_freq = stats_action_freq.get_mean()[:, 1, :]
        most_common_action_idx = np.argmax(mean_action_freq, axis=1)

        mean_joint_action_freq = stats_joint_action_freq.get_mean()[1]
        most_common_joint_action_idx = np.unravel_index(np.argmax(mean_joint_action_freq), mean_joint_action_freq.shape)

        mean_action_values = stats_action_values.get_mean()
        best_action_idx = np.argmax(mean_action_values, axis=1)

        action_values_attr = action_values_attrs[0] if len(set(action_values_attrs)) == 1 else "Heterogeneous values"

        saver.print_and_save(
            f'{(i+1):03} {'*' if (mean_cci[:, -1] >= 0.45).any() else ' '} - Last windows ->\n'
            f' - [CCI] Average: {np.round(mean_cci[:, -1], 4)}\n'
            f' - [CCI] Standard deviation: {np.round(std_cci[:, -1], 4)}\n'
            f' - [CCI] Minimum and Maximum:{np.round(min_cci[:, -1], 4)}, {np.round(max_cci[:, -1], 4)}\n'
            f' - [RDC] Average: {np.round(mean_rdc[:, -1], 4)}\n'
            f' - [ACTION] Most common: {str(action_space[most_common_action_idx]).replace('\n', '')}\n'
            f' - [ACTION] Relative frequency: {np.round(mean_action_freq[np.arange(n_makers), most_common_action_idx], 4)}\n'
            f' - [JOINT ACTION] Most common: {str(action_space[most_common_joint_action_idx, :]).replace('\n', '')}\n'
            f' - [JOINT ACTION] Relative frequency: {np.round(mean_joint_action_freq[most_common_joint_action_idx], 4)}\n'
            f' - [ACTION VALUES] Best action: {str(action_space[best_action_idx]).replace('\n', '')}\n'
            f' - [ACTION VALUES] {action_values_attr.capitalize()}: {np.round(mean_action_values[np.arange(n_makers), best_action_idx], 4)}\n',
            f' - [CCE] Is an equilibrium: {is_cce}\n'
            f' - [NE] Is an equilibrium: {is_ne}'
        )
    
    # Plot results
    n_makers_per_experiment = {param['env']['n_makers'] for param in params}
    if len(n_makers_per_experiment) == 1:
        lw_min_cci = np.array([result[0].get_min()[:, -1] for result in results_list]).T
        lw_max_cci = np.array([result[0].get_max()[:, -1] for result in results_list]).T
        lw_mean_cci = np.array([result[0].get_mean()[:, -1] for result in results_list]).T
        lw_std_cci = np.array([result[0].get_std(sample=False)[:, -1] for result in results_list]).T
        
        fig, axis = plt.subplots(figsize=(16, 6))
        plots.plot_makers_cci(
            xlabel = 'Experiment Index',
            x = np.arange(n_parallel_runs),
            cci = lw_mean_cci,
            std = lw_std_cci,
            min = lw_min_cci,
            max = lw_max_cci,
            makers_name = [f'maker_{i}' for i in range(n_makers)],
            title = f'Mean CCI wrt. Variable Parameters - Last Window',
            ax = axis
        )
        plt.tight_layout()
        saver.save_figures({'PLOT': fig})

    end_time = time.time()
    execution_time = end_time - start_time
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    saver.print_and_save(f'Done at {current_time} | Execution time: {execution_time:.2f} seconds')
    return
