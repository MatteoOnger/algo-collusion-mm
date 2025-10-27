import matplotlib.pyplot as plt
import numpy as np
import os
import time

import algo_collusion_mm.utils.plots as plots
import algo_collusion_mm.utils.storage as storage

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Any, Callable, Dict, List, Tuple

from algo_collusion_mm.agents.makers.exp3 import MakerEXP3
from algo_collusion_mm.agents.traders.nopass import NoPassTrader
from algo_collusion_mm.envs import GMEnv
from algo_collusion_mm.utils.common import get_calvano_collusion_index
from algo_collusion_mm.utils.stats import OnlineVectorStats



BASE_PATH = os.path.join('.', 'experiments', 'exp3', 'varying_epsilon_rand')
FUNC_SCALE_REWARD = lambda r: r / 0.3
FUNC_GENERATE_VT = lambda: 0.5



def multiple_runs(
    generate_vt: Callable[[], float],
    scale_rewards: Callable[[float], float],
    agents_fixed_params: Dict[str, Any],
    agents_variable_params: Dict[str, Any],
    n_makers_u: int,
    n_makers_i: int,
    n_traders: int,
    action_space: np.ndarray,
    nash_reward: float,
    coll_reward: float,
    saver_base_path: str,
    r: int = 100,
    n: int = 10_000,
    k: int = 100
) -> Tuple[OnlineVectorStats, OnlineVectorStats, OnlineVectorStats, OnlineVectorStats]:
    """
    Run multiple independent experiments in a Glosten–Milgrom market simulation.

    This function executes `r` independent experimental runs of a Glosten–Milgrom 
    environment populated by informed/uninformed market makers and traders. 
    In each run, agents are reset and re-seeded, the environment is simulated 
    for `n` episodes, and performance metrics (including the Calvano Collusion 
    Index, or CCI) are computed over `k` rolling windows.

    Statistics of agent behavior, actions, rewards, and CCI are collected and 
    aggregated across runs. All results and metadata are saved under 
    `saver_base_path`, and summary plots are generated automatically.

    Parameters
    ----------
    generate_vt : Callable[[], float]
        Function that generates the true (fundamental) value for each episode.
    scale_rewards : Callable[[float], float]
        Function that scales or transforms agent rewards (e.g., for learning stability).
    agents_fixed_params : Dict[str, Any]
        Dictionary of hyperparameters shared across all agents of a given type 
        (e.g., learning rate, exploration strategy).
    agents_variable_params : Dict[str, Any]
        Dictionary of run- or agent-specific parameters (e.g., seed, tie-breaking rule).
    n_makers_u : int
        Number of uninformed market makers in the environment.
    n_makers_i : int
        Number of informed market makers in the environment.
    n_traders : int
        Number of traders in the environment.
    action_space : np.ndarray
        Discrete set of possible (bid, ask) price combinations for market makers.
    nash_reward : float
        Benchmark single-agent reward under Nash equilibrium, used to normalize the CCI.
    coll_reward : float
        Benchmark single-agent reward under full collusion, used to normalize the CCI.
    saver_base_path : str
        Base directory where experiment data, logs, and plots will be saved.
    r : int, default=100
        Number of independent experimental runs to execute.
    n : int, default=10_000
        Number of trading episodes per run.
    k : int, default=100
        Number of windows to divide the `n` episodes into when computing the CCI.

    Returns
    -------
    : Tuple[np.ndarray, np.ndarray]
        - Mean CCI across all runs, shape (n_makers, k)
        - Standard deviation of CCI across all runs, shape (n_makers, k)

    Notes
    -----
    - Each run creates fresh agent instances, resets their internal states, and re-seeds RNGs.
    - The Calvano Collusion Index (CCI) measures how close the agents' joint behavior 
      is to Nash equilibrium (CCI = 0) or full collusion (CCI = 1).
    - Experiment results include per-agent reward histories, frequency of chosen actions, 
      and summary statistics, all serialized by `ExperimentStorage`.
    - A plot of mean ± std CCI across windows is generated and saved automatically.
    """
    w = n // k                              # Window size
    n_makers = n_makers_u + n_makers_i      # Tot number of makers

    # To compute online statistics
    stats_cci = OnlineVectorStats((n_makers, k))
    stats_sorted_cci = OnlineVectorStats((n_makers, k))
    stats_action_freq = OnlineVectorStats((n_makers, len(action_space)))
    stats_joint_action_freq = OnlineVectorStats((len(action_space),) * n_makers)
    stats_rwd = OnlineVectorStats(n_makers)

    # To save experimental results
    saver = storage.ExperimentStorage(saver_base_path)

    start_time = time.time()
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    saver.print_and_save(f'Started at {current_time}', silent=True)

    # Create agents
    agents = {
        f'maker_u_{i}':MakerEXP3(
            name = f'maker_u_{i}',
            scale_rewards = scale_rewards,
            **agents_fixed_params['maker'],
            **agents_variable_params['maker']
        ) for i in range(n_makers_u)
    } | {
        f'trader_{i}':NoPassTrader(
            name = f'trader{i}',
            **agents_fixed_params['trader'],
            **agents_variable_params['trader']
        ) for i in range(n_traders)
    }

    # Create env
    env = GMEnv(
        generate_vt = generate_vt,
        n_episodes = n,
        n_makers_u = n_makers_u,
        n_makers_i = n_makers_i,
        n_traders = n_traders,
    )

    for i in range(r):
        if i % 10 == 0:
            saver.print_and_save(f'Running {i} ...', silent=True)

        # Reset the agent and change seed
        for agent in agents.values():
            agent.reset()
            agent.update_seed()

        # Reset env
        _, info = env.reset()

        # Play
        for agent in env.agent_iter():
            action = agents[agent].act(env.observe(agent))
            _, rewards, _, _, infos = env.step(action)

            if infos['episode_finished']:
                for a in env.possible_agents:
                    agents[a].update(rewards[a], infos[a])

        # Compute calvano collusion idex per window and agent
        cci = get_calvano_collusion_index(
            np.array([agents[name].history.get_rewards() for name in env.makers]),
            nash_reward = nash_reward,
            coll_reward = coll_reward,
            window_size = w
        )

        # Collect info
        info = {
            'parmas' : {
                'n_episodes' : n,
                'window_size' : w,
                'action_space' : str(action_space).replace('\n', ','),
                'tie_breaker' : [agents[name].tie_breaker for name in env.traders],
                'epsilon' : [agents[name].epsilon for name in env.makers],
                'seed' : {name : agent._seed for name, agent in agents.items()},
                'agent_type' : [agent.__class__.__name__ for agent in agents.values()],
            },
            'freq_actions' : {
                0 : {name : str(agents[name].history.compute_freqs(slice(0, w))).replace('\n', '') for name in env.makers},
                k : {name : str(agents[name].history.compute_freqs(slice(-w, None))).replace('\n', '') for name in env.makers},
                'global' : {name : str(agents[name].history.compute_freqs()).replace('\n', '') for name in env.makers}
            },
            'most_common_action' : {
                0 : {name : str(agents[name].history.compute_most_common(slice(0, w))) for name in env.makers},
                k : {name : str(agents[name].history.compute_most_common(slice(-w, None))) for name in env.makers},
                'global' : {name : str(agents[name].history.compute_most_common()) for name in env.makers}
            },
            'cumulative_rewards' : {
                0 : {name : round(float(agent.history.get_rewards(slice(0, w)).sum()), 3) for name, agent in agents.items()},
                k : {name : round(float(agent.history.get_rewards(slice(-w, None)).sum()), 3) for name, agent in agents.items()},
                'global' : env.cumulative_rewards
            },
            'cci' : {
                0  : {name : round(float(cci[idx, 0]), 3) for idx, name in enumerate(env.makers)},
                k  : {name : round(float(cci[idx, -1]), 3) for idx, name in enumerate(env.makers)},
                'global' : {name : round(float(cci[idx, :].mean()), 3) for idx, name in enumerate(env.makers)},
            }
        }

        # Sort agents according to the CCI of the last window
        sorted_cci = cci[np.argsort(cci[:, -1])[::-1]]

        # Joint actions frequency
        actions_comb = np.array([
            agents[name].action_to_index(agents[name].history.get_actions(slice(-w, None))) for name in env.makers
        ]).T
        unique_actions_comb, freqs = np.unique(actions_comb, return_counts=True, axis=0)
        tmp_mat = np.zeros((len(action_space),) * n_makers)

        print(unique_actions_comb)
        print(tmp_mat.shape)

        tmp_mat[tuple(unique_actions_comb.T)] = freqs / w

        # Update statistics
        stats_cci.update(cci)
        stats_sorted_cci.update(sorted_cci)
        stats_action_freq.update(np.array([agents[name].history.compute_freqs(slice(-w, None)) for name in env.makers]) / w)
        stats_joint_action_freq.update(tmp_mat)
        stats_rwd.update(np.array([env.cumulative_rewards[name] for name in env.makers]))

        # Save and print results
        dir = saver.save_experiment([env] + list(agents.values()), info=info)
        saver.print_and_save(f'{(i+1):03} {'*' if (cci[:, -1] >= 0.45).any() else ' '} -> CCI:{info['cci'][k]}'.ljust(60) + f' ({dir})', silent=True)

    end_time = time.time()
    execution_time = end_time - start_time
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    saver.print_and_save(f'Done at {current_time} | Execution time: {execution_time:.2f} seconds', silent=True)
    
    # Save plot
    fig = plots.plot_all_stats(
        w,
        {maker:agents[maker] for maker in env.makers},
        stats_cci,
        stats_sorted_cci,
        stats_action_freq,
        stats_joint_action_freq,
        title = f'Makers Stas Summary Plots - Epsilon:{list(agents.values())[0].epsilon}'
    )
    saver.save_figures({f'PLOT': fig})

    # Save and print results
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
        f'- Global:\n'
        f' - [RWD] Average: {np.round(stats_rwd.get_mean(), 4)}\n'
        f' - [RWD] Standard deviation: {np.round(stats_rwd.get_std(sample=False), 4)}',
        silent = True
    )
    return stats_cci, stats_sorted_cci, stats_action_freq, stats_joint_action_freq


def worker(
    run_id: int,
    fixed_params: Dict[str, Any],
    variable_params: List[Dict[str, Any]]
) -> Tuple[OnlineVectorStats, OnlineVectorStats, OnlineVectorStats, OnlineVectorStats]:
    """
    Worker function for executing a single run of `multiple_runs`.

    This function builds agent and environment configurations for a specific run, based on a 
    shared set of fixed parameters and a per-run dictionary of variable parameters. It supports
    running parallel experiments via `ProcessPoolExecutor`, allowing easy variation of parameters
    like agent exploration (`epsilon`) across runs.

    Agent instances (`MakerEXP3`, `NoPassTrader`) are created with a combination of:
        - `fixed_params`: shared settings for all runs
        - `variable_params[run_id]`: run-specific overrides (e.g., `epsilon`, save path)

    Environment parameters are also composed from fixed and variable sources.

    Parameters
    ----------
    run_id : int
        The index of the current parallel run, used to select the appropriate variable parameters.
    fixed_params : Dict[str, Any]
        Dictionary containing shared configuration across all runs.
        Keys: 'env', 'maker', 'trader'
    variable_params : List[Dict[str, Any]]
        List of per-run configurations. Each element overrides or extends the fixed parameters
        for its corresponding run.

    Returns
    -------
    np.ndarray
        A tuple containing:
            - Mean CCI across all internal runs: shape (n_makers, k)
            - Standard deviation of CCI across all internal runs: shape (n_makers, k)
    """
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{now}] [Run {run_id}] Starting...', flush=True)

    results = multiple_runs(
        generate_vt = FUNC_GENERATE_VT,
        scale_rewards = FUNC_SCALE_REWARD,
        agents_fixed_params = fixed_params['agent'],
        agents_variable_params = variable_params[run_id]['agent'],
        **fixed_params['env'],
        **variable_params[run_id]['env']
    )

    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{now}] [Run {run_id}] Finished', flush=True)
    return results



if __name__ == '__main__':
    saver = storage.ExperimentStorage(BASE_PATH)

    max_workers = 6
    n_parallel_runs = 93

    start_time = time.time()
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    saver.print_and_save(f'Started at {current_time}')

    n_makers_u = 3
    r, n, k = 100, 20_000, 100
    prices =  np.round(np.arange(0.0, 1.0 + 0.2, 0.2), 2)
    action_space = np.array([(ask, bid) for ask in prices for bid in prices if (ask  > bid)])

    x = MakerEXP3.compute_epsilon(len(action_space), n)
    epsilons = np.round(np.concat([
        x - np.arange(1,  8)[::-1] * 0.0005,
        np.array([x]),
        x + np.arange(1,  6) * 0.0005,
        x + 0.0025 + np.arange(1, 11) * 0.0010,
        x + 0.0125 + np.arange(1, 21) * 0.0050,
        x + 0.1125 + np.arange(1, 41) * 0.0100,
        x + 0.5125 + np.arange(1, 10) * 0.0500,
        5.0
    ]), 4)
    
    print(len(epsilons))
    print(epsilons)
    quit()

    assert len(epsilons) == n_parallel_runs

    # Experiment params
    fixed_params = {
        'env': {
            'n_makers_u': n_makers_u,
            'n_makers_i': 0,
            'n_traders': 1,
            'action_space': action_space,
            'nash_reward': 0.1,
            'coll_reward': 0.5,
            'r': r,
            'n': n,
            'k': k
        },
        'agent': {
            'maker': {
                'action_space': action_space,
            },
            'trader': {
                'tie_breaker': 'rand'
            }
        }
    }

    variable_params = [{
        'env': {
            'saver_base_path': os.path.join(BASE_PATH, str(i+1)),
        },
        'agent': {
            'maker': {
                'epsilon': epsilons[i],
            },
            'trader': {
            }
        }
    } for i in range(n_parallel_runs)]

    # Parallel run
    futures = list()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, run_id, fixed_params, variable_params) for run_id in range(n_parallel_runs)]
    results_list = [f.result() for f in futures]

    # Save results
    saver.save_objects({'results_list': results_list})

    # Print results
    for i, results in enumerate(results_list):
        stats_cci, stats_sorted_cci, stats_action_freq, stats_joint_action_freq = results
        
        min_cci, mean_cci, max_cci = stats_cci.get_min(), stats_cci.get_mean(), stats_cci.get_max()
        std_cci = stats_cci.get_std(sample=False)
        
        mean_sorted_cci = stats_sorted_cci.get_mean()
        std_sorted_cci = stats_sorted_cci.get_std(sample=False)

        mean_action_freq = stats_action_freq.get_mean()
        most_common_action_idx = np.argmax(mean_action_freq, axis=1)

        mean_joint_action_freq = stats_joint_action_freq.get_mean()
        most_common_joint_action_idx = np.unravel_index(np.argmax(mean_joint_action_freq), mean_joint_action_freq.shape)

        saver.print_and_save(
            f'{(i+1):03} {'*' if (mean_cci[:, -1] >= 0.45).any() else ' '} - epsilon: {epsilons[i]} - Last windows ({n//k}) ->\n'
            f' - [CCI] Mean and standard deviation: {np.round(mean_cci[:, -1], 4)}, {np.round(std_cci[:, -1], 4)}\n'
            f' - [CCI] Maximum and minimum: {np.round(max_cci[:, -1], 4)}, {np.round(min_cci[:, -1], 4)}\n'
            f' - [ACTION] Most common and frequency: {str(action_space[most_common_action_idx]).replace('\n', '')}, {mean_action_freq[np.arange(n_makers_u), most_common_action_idx]}\n'
            f' - [JOINT ACTION] Most common and frequency: {str(action_space[most_common_joint_action_idx, :]).replace('\n', '')}, {mean_joint_action_freq[most_common_joint_action_idx]}'
        )
    
    # # Plot results
    lwm_cci = np.array([result[0].get_mean()[:, -1] for result in results_list]).T
    lws_cci = np.array([result[0].get_std(sample=False)[:, -1] for result in results_list]).T
    lwm_sorted_cci = np.array([result[1].get_mean()[:, -1] for result in results_list]).T
    lws_sorted_cci = np.array([result[1].get_std(sample=False)[:, -1] for result in results_list]).T

    fig, axes = plt.subplots(2, 1, figsize=(18, 12))
    plots.plot_makers_cci(
        n // k,
        lwm_cci,
        lws_cci,
        x = np.arange(len(epsilons)),
        title = 'Mean CCI wrt. Epsilons - Last Window',
        xlabel = 'Epsilon',
        agents_name = [f'maker_u_{i}' for i in range(n_makers_u)],
        ax = axes[0]
    )
    plots.plot_makers_cci(
        n // k,
        lwm_sorted_cci,
        lws_sorted_cci,
        x = np.arange(len(epsilons)),
        title = 'Mean Sorted CCI wrt. Epsilons - Last Window',
        xlabel = 'Epsilon',
        agents_name = [f'maker_u_{i}' for i in range(n_makers_u)],
        ax=axes[1]
    )
    axes[0].axvline(7, ls='--', color='black', alpha=0.7)
    axes[1].axvline(7, ls='--', color='black', alpha=0.7)
    plt.tight_layout()
    saver.save_figures({'PLOT': fig})

    end_time = time.time()
    execution_time = end_time - start_time
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    saver.print_and_save(f'Done at {current_time} | Execution time: {execution_time:.2f} seconds')
