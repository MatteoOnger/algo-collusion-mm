import matplotlib.pyplot as plt
import numpy as np
import os
import time

import algo_collusion_mm.utils.plots as plots

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Any, Callable, Dict, List, Tuple

from algo_collusion_mm.agents.makers.uninformed.mlql import MakerMLQL
from algo_collusion_mm.agents.traders.nopass import NoPassTrader
from algo_collusion_mm.envs import GMEnv
from algo_collusion_mm.utils.common import get_calvano_collusion_index
from algo_collusion_mm.utils.stats import OnlineVectorStats
from algo_collusion_mm.utils.storage import ExperimentStorage



BASE_PATH = os.path.join('.', 'experiments', 'mlql')
FUNC_GENERATE_VT = lambda: 0.5



def multiple_runs(
    generate_vt: Callable[[], float],
    agents_fixed_params: Dict[str, Any],
    agents_variable_params: Dict[str, Any],
    n_makers_u: int,
    n_traders: int,
    action_space: np.ndarray,
    nash_reward: float,
    coll_reward: float,
    saver_base_path: str,
    n_episodes: int = 100,
    n_rounds: int = 10_000,
    n_windows: int = 100
) -> Tuple[OnlineVectorStats, OnlineVectorStats, OnlineVectorStats, OnlineVectorStats, OnlineVectorStats]:
    """
    Run multiple independent episodes in a Glosten-Milgrom market simulation.

    This function performs several independent runs of a simulated Glosten-Milgrom 
    financial market populated by uninformed market makers basedd on Memoryless Q-learning
    as well as traders. 
    In each run, all agents are reinitialized and reseeded before simulating a sequence 
    of trading rounds. Performance metrics such as the Calvano Collusion Index (CCI), 
    reward statistics, and action frequencies are computed over rolling time windows.

    Aggregated statistics across all runs are tracked online and summarized 
    at the end of the experiment. Detailed results (including agent histories, 
    metadata, and plots) are saved to disk under the specified base directory.

    Parameters
    ----------
    generate_vt : Callable[[], float]
        Function that generates the true value of the traded asset for each round.
    agents_fixed_params : Dict[str, Any]
        Dictionary containing hyperparameters shared across all agents of a given type 
        (e.g., learning rate, exploration strategy).
    agents_variable_params : Dict[str, Any]
        Dictionary containing per-run or per-agent parameters 
        (e.g., random seed, tie-breaking rules).
    n_makers_u : int
        Number of uninformed market makers.
    n_traders : int
        Number of traders in the environment.
    action_space : np.ndarray
        Discrete set of possible (bid, ask) price pairs available to market makers.
    nash_reward : float
        Baseline per-agent reward under Nash equilibrium, used to normalize the CCI.
    coll_reward : float
        Baseline per-agent reward under full collusion, used to normalize the CCI.
    saver_base_path : str
        Base directory where experiment results, logs, and generated plots are saved.
    n_episodes : int, default=100
        Number of independent episode repetitions.
    n_rounds : int, default=10_000
        Number of trading rounds per episode run.
    n_windows : int, default=100
        Number of rolling windows into which the `n_rounds` are divided 
        for computing windowed statistics (e.g., the CCI).

    Returns
    -------
    : tuple of OnlineVectorStats
        A tuple of five `OnlineVectorStats` objects summarizing data across all runs:
        - **stats_cci** : OnlineVectorStats  
          Mean, variance, and extrema of the Calvano Collusion Index (CCI) across all runs.  
          Shape: (n_makers_i, n_windows)
        - **stats_sorted_cci** : OnlineVectorStats  
          Same as above, but with makers sorted by their final-window CCI.
        - **stats_action_freq** : OnlineVectorStats  
          Frequency of each market maker's actions across runs and in the first and last window.
          Shape: (n_makers_i, 2, len(action_space))
        - **stats_joint_action_freq** : OnlineVectorStats  
          Joint frequency of action combinations across all market makers in
          the first and last window.
          Shape: (2,) + (len(action_space),) * n_makers_i
        - **stats_belief** : OnlineVectorStats  
          Distribution of action-selection probabilities (beliefs) for each maker after the training.
          Shape: (n_makers_i, len(action_space))

    Notes
    -----
    - Each episode run creates new agent instances, resets internal states, 
      and reseeds all random number generators for independence.
    - The Calvano Collusion Index (CCI) measures the degree of collusive behavior, 
      where `CCI = 0` corresponds to Nash equilibrium and `CCI = 1` to full collusion.
    - The function automatically saves per-agent reward histories, 
      action frequency distributions, and summary plots.
    - The final summary includes per-window averages, minima, maxima, 
      and standard deviations of all tracked metrics.

    See Also
    --------
    ExperimentStorage : Handles data persistence, logging, and metadata serialization.
    OnlineVectorStats : Tracks online means and variances for high-dimensional data.
    get_calvano_collusion_index : Computes the CCI given agents' reward histories.
    """
    window_size = n_rounds // n_windows   # Window size

    # To compute online statistics
    stats_cci = OnlineVectorStats((n_makers_u, n_windows))
    stats_sorted_cci = OnlineVectorStats((n_makers_u, n_windows))
    stats_action_freq = OnlineVectorStats((n_makers_u, 2, len(action_space)))
    stats_joint_action_freq = OnlineVectorStats((2,) + n_makers_u * (len(action_space),))
    stats_belief = OnlineVectorStats((n_makers_u, len(action_space)))
    stats_rwd = OnlineVectorStats(n_makers_u)

    # To save experimental results
    saver = ExperimentStorage(saver_base_path)

    start_time = time.time()
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    saver.print_and_save(f'Started at {current_time}', silent=True)

    # Create agents
    agents = {
        f'maker_u_{i}':MakerMLQL(
            name = f'maker_u_{i}',
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
        n_rounds = n_rounds,
        n_makers_u = n_makers_u,
        n_makers_i = 0,
        n_traders = n_traders,
    )

    for i in range(n_episodes):
        if i % 10 == 0:
            saver.print_and_save(f'Running {i} ...', silent=True)

        # Reset env
        _, info = env.reset()
        # Reset the agent and change seed
        for agent in agents.values():
            agent.reset()
            agent.update_seed()

        # Play
        for agent in env.agent_iter():
            action = agents[agent].act(env.observe(agent))
            _, rewards, _, _, infos = env.step(action)

            if infos['round_finished']:
                for a in env.possible_agents:
                    agents[a].update(rewards[a], infos[a])

        # Compute calvano collusion idex per window and agent
        cci = get_calvano_collusion_index(
            np.array([agents[name].history.get_rewards() for name in env.makers]),
            nash_reward = nash_reward,
            coll_reward = coll_reward,
            window_size = window_size
        )

        # Collect info
        info = {
            'parmas' : {
                'n_rounds' : n_rounds,
                'window_size' : window_size,
                'action_space' : str(action_space).replace('\n', ','),
                'tie_breaker' : [agents[name].tie_breaker for name in env.traders],
                'alpha' : [agents[name].alpha for name in env.makers],
                'gamma' : [agents[name].gamma for name in env.makers],
                'epsilon_scheduler' : [agents[name].epsilon_scheduler for name in env.makers],
                'epsilon_init' : [agents[name].epsilon_init for name in env.makers],
                'epsilon_decay_rate' : [agents[name].epsilon_decay_rate for name in env.makers],
                'q_init' : [agents[name].q_init for name in env.makers],
                'seed' : {name : agent._seed for name, agent in agents.items()},
                'agent_type' : [agent.__class__.__name__ for agent in agents.values()],
            },
            'belief':{
                n_windows : {name : str(agents[name].Q).replace('\n', '') for name in env.makers}
            },
            'freq_actions' : {
                0 : {name : str(agents[name].history.compute_freqs(slice(0, window_size))).replace('\n', '') for name in env.makers},
                n_windows : {name : str(agents[name].history.compute_freqs(slice(-window_size, None))).replace('\n', '') for name in env.makers},
                'global' : {name : str(agents[name].history.compute_freqs()).replace('\n', '') for name in env.makers}
            },
            'most_common_action' : {
                0 : {name : str(agents[name].history.compute_most_common(slice(0, window_size))) for name in env.makers},
                n_windows : {name : str(agents[name].history.compute_most_common(slice(-window_size, None))) for name in env.makers},
                'global' : {name : str(agents[name].history.compute_most_common()) for name in env.makers}
            },
            'cumulative_rewards' : {
                0 : {name : round(float(agent.history.get_rewards(slice(0, window_size)).sum()), 3) for name, agent in agents.items()},
                n_windows : {name : round(float(agent.history.get_rewards(slice(-window_size, None)).sum()), 3) for name, agent in agents.items()},
                'global' : env.cumulative_rewards
            },
            'cci' : {
                0  : {name : round(float(cci[idx, 0]), 3) for idx, name in enumerate(env.makers)},
                n_windows  : {name : round(float(cci[idx, -1]), 3) for idx, name in enumerate(env.makers)},
                'global' : {name : round(float(cci[idx, :].mean()), 3) for idx, name in enumerate(env.makers)},
            }
        }

        # Sort agents according to the CCI of the last window
        sorted_cci = cci[np.argsort(cci[:, -1])[::-1]]

        # Joint actions frequency first and last window
        matrix = np.zeros((2,) + n_makers_u * (len(action_space),))
        
        joint_actions = np.array([
            agents[name].history.get_actions(slice(window_size), return_index=True) for name in env.makers
        ]).T
        unique_joint_actions, freqs = np.unique(joint_actions, return_counts=True, axis=0)
        matrix[0][tuple(unique_joint_actions.T)] = freqs / window_size
        
        joint_actions = np.array([
            agents[name].history.get_actions(slice(-window_size, None), return_index=True) for name in env.makers
        ]).T
        unique_joint_actions, freqs = np.unique(joint_actions, return_counts=True, axis=0)
        matrix[1][tuple(unique_joint_actions.T)] = freqs / window_size

        # Update statistics
        stats_cci.update(cci)
        stats_sorted_cci.update(sorted_cci)
        stats_action_freq.update(np.array(
            [[
                agents[maker].history.compute_freqs(slice(window_size)),
                agents[maker].history.compute_freqs(slice(-window_size, None))
            ] for maker in env.makers]
        ) / window_size)
        stats_joint_action_freq.update(matrix)
        stats_belief.update(np.array([agents[maker].Q for maker in env.makers]))
        stats_rwd.update(np.array([env.cumulative_rewards[maker] for maker in env.makers]))

        # Save and print results
        dir = saver.save_episode(info=info)
        saver.print_and_save(f'{(i+1):03} {'*' if (cci[:, -1] >= 0.45).any() else ' '} -> CCI:{info['cci'][n_windows]}'.ljust(60) + f' ({dir})', silent=True)

    end_time = time.time()
    execution_time = end_time - start_time
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    saver.print_and_save(f'Done at {current_time} | Execution time: {execution_time:.2f} seconds', silent=True)
    
    # Save plot
    fig = plots.plot_all_stats(
        window_size,
        [agents[maker] for maker in env.makers],
        stats_cci,
        stats_sorted_cci,
        stats_action_freq,
        stats_joint_action_freq,
        stats_belief,
        title = f'MLQL - Makers Statistics Summary Plot - Epsilon:{list(agents.values())[0].epsilon}'
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
        f' - [RWD] Expected: {np.round(stats_cci.get_mean()[:, -1] * (coll_reward/n_makers_u - nash_reward/n_makers_u) + nash_reward/n_makers_u, 4)}\n'
        f'- Global:\n'
        f' - [RWD] Average: {np.round(stats_rwd.get_mean(), 4)}\n'
        f' - [RWD] Standard deviation: {np.round(stats_rwd.get_std(sample=False), 4)}',
        silent = True
    )
    return stats_cci, stats_sorted_cci, stats_action_freq, stats_joint_action_freq, stats_belief


def worker(
    run_id: int,
    fixed_params: Dict[str, Any],
    variable_params: List[Dict[str, Any]]
) -> Tuple[OnlineVectorStats, OnlineVectorStats, OnlineVectorStats, OnlineVectorStats, OnlineVectorStats]:
    """
    Execute a single experiment configuration for use in parallelized Glosten-Milgrom simulations.

    This worker function serves as a wrapper around `multiple_runs`, building a specific 
    environment and agent configuration for a given `run_id`. It merges shared experiment 
    settings (`fixed_params`) with per-run overrides (`variable_params[run_id]`), allowing 
    for easy parameter sweeps (e.g., different exploration rates or random seeds).

    It is typically executed within a `ProcessPoolExecutor` or similar parallel execution
    framework to run multiple experiments concurrently.

    Parameters
    ----------
    run_id : int
        Index of the current run, used to select the corresponding element from `variable_params`.
    fixed_params : Dict[str, Any]
        Dictionary containing experiment-wide settings shared by all runs.
        Expected keys:
            - `'env'` : Environment configuration parameters.
            - `'agent'` : Common agent hyperparameters (for both makers and traders).
    variable_params : List[Dict[str, Any]]
        List of dictionaries containing per-run configuration overrides.
        Each entry may redefine environment parameters (under `'env'`) and/or agent parameters 
        (under `'agent'`), such as `epsilon`, learning rate, or save paths.

    Returns
    -------
    : tuple of OnlineVectorStats
        A tuple of five `OnlineVectorStats` objects summarizing data across all runs:
        - **stats_cci** : OnlineVectorStats  
          Mean, variance, and extrema of the Calvano Collusion Index (CCI) across all runs.  
          Shape: (n_makers_i, n_windows)
        - **stats_sorted_cci** : OnlineVectorStats  
          Same as above, but with makers sorted by their final-window CCI.
        - **stats_action_freq** : OnlineVectorStats  
          Frequency of each market maker's actions across runs and in the first and last window.
          Shape: (n_makers_i, 2, len(action_space))
        - **stats_joint_action_freq** : OnlineVectorStats  
          Joint frequency of action combinations across all market makers in
          the first and last window.
          Shape: (2,) + (len(action_space),) * n_makers_i
        - **stats_belief** : OnlineVectorStats  
          Distribution of action-selection probabilities (beliefs) for each maker after the training.
          Shape: (n_makers_i, len(action_space))

    Notes
    -----
    - The function prints time-stamped start and completion messages for monitoring progress.
    - Combines fixed and variable configurations using Python's dictionary unpacking:
      `**fixed_params['env']`, `**variable_params[run_id]['env']`, etc.
    - The core experiment logic and statistics computation are delegated to `multiple_runs`.
    - Designed for use in distributed or parallelized experiment pipelines.
    """
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{now}] [Run {run_id}] Starting...', flush=True)

    results = multiple_runs(
        generate_vt = FUNC_GENERATE_VT,
        agents_fixed_params = fixed_params['agent'],
        agents_variable_params = variable_params[run_id]['agent'],
        **fixed_params['env'],
        **variable_params[run_id]['env']
    )

    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{now}] [Run {run_id}] Finished', flush=True)
    return results



if __name__ == '__main__':
    saver = ExperimentStorage(BASE_PATH)

    max_workers = 2
    n_parallel_runs = 4

    start_time = time.time()
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    saver.print_and_save(f'Started at {current_time}')

    n_makers_u = 2
    r, n, k = 10, 50_000, 100
    prices =  np.round(np.arange(0.0, 1.0 + 0.2, 0.2), 2)
    action_space = np.array([(ask, bid) for ask in prices for bid in prices if (ask  > bid)])

    # epsilons = np.round(np.arange(0, 101) / 100, 4)
    epsilons = np.array([0.001, 0.005, 0.01, 0.05])

    assert len(epsilons) == n_parallel_runs

    # Experiment params
    fixed_params = {
        'env': {
            'n_makers_u': n_makers_u,
            'n_traders': 1,
            'action_space': action_space,
            'nash_reward': 0.1,
            'coll_reward': 0.5,
            'n_episodes': r,
            'n_rounds': n,
            'n_windows': k
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
            'saver_base_path': os.path.join(BASE_PATH, f'experiment_{(i+1):03}'),
        },
        'agent': {
            'maker': {
                'epsilon_init': epsilons[i],
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
        stats_cci, stats_sorted_cci, stats_action_freq, stats_joint_action_freq, stats_belief = results
        
        min_cci, mean_cci, max_cci = stats_cci.get_min(), stats_cci.get_mean(), stats_cci.get_max()
        std_cci = stats_cci.get_std(sample=False)
        
        mean_sorted_cci = stats_sorted_cci.get_mean()
        std_sorted_cci = stats_sorted_cci.get_std(sample=False)

        mean_action_freq = stats_action_freq.get_mean()[:, 1, :]
        most_common_action_idx = np.argmax(mean_action_freq, axis=1)

        mean_joint_action_freq = stats_joint_action_freq.get_mean()[1]
        most_common_joint_action_idx = np.unravel_index(np.argmax(mean_joint_action_freq), mean_joint_action_freq.shape)

        mean_belief = stats_belief.get_mean()
        best_action_idx = np.argmax(mean_belief, axis=1)

        saver.print_and_save(
            f'{(i+1):03} {'*' if (mean_cci[:, -1] >= 0.45).any() else ' '} - epsilon: {epsilons[i]} - Last windows ({n//k}) ->\n'
            f' - [CCI] Average: {np.round(mean_cci[:, -1], 4)}\n'
            f' - [CCI] Standard deviation: {np.round(std_cci[:, -1], 4)}\n'
            f' - [CCI] Minimum and Maximum:{np.round(min_cci[:, -1], 4)}, {np.round(max_cci[:, -1], 4)}\n'
            f' - [ACTION] Most common: {str(action_space[most_common_action_idx]).replace('\n', '')}\n'
            f' - [ACTION] Relative frequency: {np.round(mean_action_freq[np.arange(n_makers_u), most_common_action_idx], 4)}\n'
            f' - [JOINT ACTION] Most common: {str(action_space[most_common_joint_action_idx, :]).replace('\n', '')}\n'
            f' - [JOINT ACTION] Relative frequency: {np.round(mean_joint_action_freq[most_common_joint_action_idx], 4)}\n'
            f' - [BELIEF] Best action: {str(action_space[best_action_idx]).replace('\n', '')}\n'
            f' - [BELIEF] Probability: {np.round(mean_belief[np.arange(n_makers_u), best_action_idx], 4)}',
        )
    
    # Plot results
    lw_min_cci = np.array([result[0].get_min()[:, -1] for result in results_list]).T
    lw_max_cci = np.array([result[0].get_max()[:, -1] for result in results_list]).T
    lw_mean_cci = np.array([result[0].get_mean()[:, -1] for result in results_list]).T
    lw_std_cci = np.array([result[0].get_std(sample=False)[:, -1] for result in results_list]).T
    
    fig, axis = plt.subplots(figsize=(16, 6))
    plots.plot_makers_cci(
        xlabel = 'Experiment Index',
        x = np.arange(len(epsilons)),
        cci = lw_mean_cci,
        std = lw_std_cci,
        min = lw_min_cci,
        max = lw_max_cci,
        makers_name = [f'maker_u_{i}' for i in range(n_makers_u)],
        title = 'EXP3 - Mean CCI wrt. Epsilons - Last Window',
        ax = axis
    )
    axis.axvline(7, ls='--', color='black', alpha=0.7)
    plt.tight_layout()
    saver.save_figures({'PLOT': fig})

    end_time = time.time()
    execution_time = end_time - start_time
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    saver.print_and_save(f'Done at {current_time} | Execution time: {execution_time:.2f} seconds')
