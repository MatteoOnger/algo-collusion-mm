import numpy as np
import os

from algo_collusion_mm.agents import NoPassTrader, MakerExp3
from runner import run_experiment_suite



if __name__ == '__main__':
    n_episodes = 10
    n_rounds = 1_000
    prices = np.round(np.arange(0.0, 1.0 + 0.2, 0.2), 2)
    action_space = np.array([(ask, bid) for ask in prices for bid in prices if (ask  > bid)])

    epsilons = [0.001, 0.01, 0.1]

    fixed_params = {
        'env': {
            'generate_vt': 'lambda: 0.5',
            'info_level': 'partial',
            'n_arms': len(action_space),
            'n_makers': 2,
            'n_traders': 1,
            'nash_reward': 0.1,
            'coll_reward': 0.5,
            'decimal_places': 3,
            'n_episodes': n_episodes,
            'n_rounds': n_rounds,
            'n_windows': 100,
        },
        'agent': {
            'maker': {
                'meta': {
                    'prefix': 'maker_u',
                    'class': MakerExp3
                },
                'params': {
                    'action_space': action_space,
                    'scale_rewards': 'lambda x: x / 0.3',
                }
            },
            'trader': {
                'meta':{
                    'prefix': 'trader',
                    'class': NoPassTrader,
                },
                'params': {
                    'tie_breaker': 'rand'
                }
            }
        }
    }

    variable_params = [
        {
            'env': {},
            'agent': {
                'maker': {
                    'meta': {},
                    'params': {
                        'epsilon': epsilons[i]
                    }
                },
                'trader': {
                    'meta':{},
                    'params':{}
                }
            }
        }
    for i in range(len(epsilons))]

    run_experiment_suite(
        fixed_params = fixed_params,
        variable_params = variable_params,
        base_path = os.path.join('experiments', 'exp3')
    )
