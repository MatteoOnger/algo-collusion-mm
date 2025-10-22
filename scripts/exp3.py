import matplotlib.pyplot as plt
import numpy as np
import os
import time

import algo_collusion_mm.utils.plots as plots
import algo_collusion_mm.utils.storage as storage

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Any, Callable, Dict, List

from algo_collusion_mm.agents.agent import Agent
from algo_collusion_mm.agents.makers.exp3 import MakerEXP3
from algo_collusion_mm.agents.traders.nopass import NoPassTrader
from algo_collusion_mm.envs import GMEnv
from algo_collusion_mm.utils.common import get_calvano_collusion_index
from algo_collusion_mm.utils.stats import OnlineVectorStats



BASE_PATH = os.path.join('.', 'experiments', 'exp3')
FUNC_SCALE_REWARD = lambda r: r / 0.3
FUNC_GENERATE_VT = lambda: 0.5



def multiple_runs(
    generate_vt: Callable[[], float],
    n_makers_u: int,
    n_makers_i: int,
    n_traders: int,
    action_space: np.ndarray,
    agents: Dict[str, Agent],
    nash_reward: float,
    coll_reward: float,
    saver_base_path: str,
    r: int = 100,
    n: int = 10_000,
    k: int = 100
) -> np.ndarray:
    """
    Run multiple independent experiments in a Glosten-Milgrom environment.

    This function executes `r` repeated experiments using the provided agents and environment
    configuration. In each experiment, the environment is simulated for `n` episodes and
    the behavior of the agents is recorded. The Calvano Collusion Index (CCI) is computed
    over rolling windows of size `n // k` for each agent.

    The function aggregates statistics across runs, saves experiment data and metadata, and
    returns summary statistics of CCI and rewards.

    Parameters
    ----------
    generate_vt : Callable[[], float]
        Function that generates the true value for each episode.
    n_makers_u : int
        Number of uninformed market makers in the environment.
    n_makers_i : int
        Number of informed market makers in the environment.
    n_traders : int
        Number of traders in the environment.
    action_space : np.ndarray
        Array of possible ask/bid price combinations for market makers.
    agents : Dict[str, Agent]
        Dictionary mapping agent names to Agent instances.
    nash_reward : float
        The benchmark reward (single-agent case) under Nash equilibrium for computing the CCI.
    coll_reward : float
        The benchmark reward (single-agent case) under full collusion for computing the CCI.
    saver_base_path : str
        Base directory where experiment results will be stored.
    r : int, default=100
        Number of independent experiment runs to execute.
    n : int, default=10_000
        Number of episodes per run.
    k : int, default=100
        Number of windows to divide the episode timeline into for CCI calculation.

    Returns
    -------
    : Tuple[np.ndarray, np.ndarray]
        - Mean CCI across all runs: shape (n_makers, k)
        - Standard deviation of CCI across all runs: shape (n_makers, k)

    Notes
    -----
    - Each agent is reset and reseeded before every run.
    """
    w = n // k                              # Window size
    n_makers = n_makers_u + n_makers_i      # Tot number of makers

    # To compute online statistics
    stats_cci = OnlineVectorStats((n_makers, k))
    stats_rwd = OnlineVectorStats(n_makers)

    # To save experimental results
    saver = storage.ExperimentStorage(saver_base_path)

    start_time = time.time()
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    saver.print_and_save(f'Started at {current_time}', silent=True)

    for i in range(r):
        if i % 10 == 0:
            saver.print_and_save(f'Running {i} ...', silent=True)

        # Reset the agent and change seed
        for agent in agents.values():
            agent.reset()
            agent.update_seed()

        env = GMEnv(
            generate_vt = generate_vt,
            n_episodes = n,
            n_makers_u = n_makers_u,
            n_makers_i = n_makers_i,
            n_traders = n_traders,
        )

        _, info = env.reset()

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
                'epsilon' : [agents[name].epsilon for name in env.makers],
                'agent_type' : [agent.__class__.__name__ for agent in agents.values()]
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
            },
            'seed' : {
                name : agent._seed for name, agent in agents.items()
            }
        }

        # Update statistics
        stats_cci.update(cci)
        stats_rwd.update(np.array([env.cumulative_rewards[name] for name in env.makers]))

        # Save and print results
        dir = saver.save_experiment([env] + list(agents.values()), info=info)
        saver.print_and_save(f'{(i+1):03} {"*" if cci[0, -1] >= 0.45 or cci[1, -1] >= 0.45 else " "} -> CCI:{info["cci"][n//w]}'.ljust(60) + f' ({dir})', silent=True)

    end_time = time.time()
    execution_time = end_time - start_time
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    saver.print_and_save(f'Done at {current_time} | Execution time: {execution_time:.2f} seconds', silent=True)
    
    # Save plot
    fig, ax = plt.subplots(figsize=(14, 6)) 
    plots.plot_makers_cci(
        episodes_per_window = w,
        cci = stats_cci.get_mean(),
        std = stats_cci.get_std(),
        title = f'Mean Calvano Collusion Index (CCI) - Epsilon: {list(agents.values())[0].epsilon}',
        agents_name = env.makers,
        ax = ax
    )
    saver.save_figures({'PLOT': fig})

    # Save and print results
    saver.print_and_save(
        f'Results:\n'
        f'- [CCI] Average in the last window: {np.round(stats_cci.get_mean()[:, -1], 4)}\n'
        f'- [CCI] Standard deviation in the last window: {np.round(stats_cci.get_std()[:, -1], 4)}\n'
        f'- [RWD] Global average: {np.round(stats_rwd.get_mean(), 4)}\n'
        f'- [RWD] Global standard deviation: {np.round(stats_rwd.get_std(), 4)}',
        silent = True
    )
    return stats_cci.get_mean(), stats_cci.get_std()


def worker(run_id: int, fixed_params: Dict[str, Any], variable_params: List[Dict[str, Any]]) -> np.ndarray:
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
    
    n_makers_u = fixed_params['env']['n_makers_u'] if 'n_makers_u' in fixed_params['env'] else variable_params[run_id]['env']['n_makers_u'] 
    n_traders = fixed_params['env']['n_traders'] if 'n_traders' in fixed_params['env'] else variable_params[run_id]['env']['n_traders'] 

    agents = {
        f'maker_u_{i}':MakerEXP3(
            name=f'maker_u_{i}', **fixed_params['maker'], **variable_params[run_id]['maker']
        ) for i in range(n_makers_u)
    } | {
        f'trader_{i}':NoPassTrader(
            name=f'trader{i}', **fixed_params['trader'], **variable_params[run_id]['trader']
        ) for i in range(n_traders)
    }

    results = multiple_runs(
        generate_vt = FUNC_GENERATE_VT,
        agents = agents,
        **fixed_params['env'],
        **variable_params[run_id]['env']
    )

    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{now}] [Run {run_id}] Finished', flush=True)
    return results



if __name__ == '__main__':
    saver = storage.ExperimentStorage(BASE_PATH)

    max_workers = 2
    n_parallel_runs = 2

    start_time = time.time()
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    saver.print_and_save(f'Started at {current_time}')

    prices =  np.round(np.arange(0.0, 1.0 + 0.2, 0.2), 2)
    action_space = np.array([(ask, bid) for ask in prices for bid in prices if (ask  > bid)])
    epsilons = [0.01, 0.10]

    # Experiment params
    fixed_params = {
        'env': {
            'n_makers_u': 2,
            'n_makers_i': 0,
            'n_traders': 1,
            'action_space': action_space,
            'nash_reward': 0.1,
            'coll_reward': 0.5,
            'r': 2,
        },
        'maker': {
            'action_space': action_space,
        },
        'trader': {
            'tie_breaker': 'rand'
        }
    }

    variable_params = [{
        'env': {
            'saver_base_path': os.path.join(BASE_PATH, str(epsilons[i])),
        },
        'maker': {
            'epsilon': epsilons[i],
        },
        'trader': {

        }
    } for i in range(n_parallel_runs)]

    # Parallel run
    futures = list()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, run_id, fixed_params, variable_params) for run_id in range(n_parallel_runs)]
    results_list = [f.result() for f in futures]

    # Print results
    for i, result in enumerate(results_list):
        cci = result[0]
        std = result[1]
        mlw_cci = float(cci[:, -1].min())
        mlw_std = float(std[:, -1].max())

        saver.print_and_save(f'{(i+1):03} {"*" if (cci[:, -1] >= 0.45).any() else " "} -> EPS:{epsilons[i]} | MIN_CCI:{mlw_cci:.3f}, MAX_STD:{mlw_std:.3f}')

    end_time = time.time()
    execution_time = end_time - start_time
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    saver.print_and_save(f'Done at {current_time} | Execution time: {execution_time:.2f} seconds')
