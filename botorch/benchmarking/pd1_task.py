
# warnings.filterwarnings('ignore')
from ConfigSpace import Configuration
import sys
import os
from os.path import join, dirname, abspath
from botorch.test_functions.base import BaseTestProblem
import mfpbench

import numpy as np
import torch
from torch import Tensor


DATADIR = join(dirname(dirname(dirname(__file__))), 'mf-prior-bench/data')
BENCHMARKS = [
    'translatewmt_xformer_64',
    'lm1b_transformer_2048',
    'cifar100_wideresnet_2048',
]

def format_arguments(config_space, args):
    hparams = config_space.get_hyperparameters_dict()
    formatted_hparams = {}
    for param, param_values in hparams.items():
        formatted_hparams[param] = param_values.sequence[args[param]]
    return formatted_hparams


def rescale(value, low, high, log=1):

    if log == 2:
        low_ = np.log2(low)
        high_ = np.log2(high)
    elif log == 10:
        low_ = np.log10(low)
        high_ = np.log10(high)
    else:
        low_ = low
        high_ = high
    scale = low_ + value * (high_ - low_)

    if log > 1:
        return np.clip(np.power(log, scale), low, high)
    else:
        return np.clip(scale, low, high)


def rescale_arguments(args, benchmark):
    lr_decay_factor, lr_initial, lr_power, opt_momentum = args

    # 3 values, nearest alternative
    # depth = np.round((depth + 1e-10) * 2.999999999 + 0.5)
    if benchmark == 'translatewmt_xformer_64':
        lr_decay_factor_range = 0.0100221257, 0.988565263
        lr_initial_range = 1.00276e-05, 9.8422475735
        lr_power_range = 0.1004250993, 1.9985927056
        opt_momentum_range = 5.86114e-05, 0.9989999746
    elif benchmark == 'lm1b_transformer_2048':
        lr_decay_factor_range = 0.010543, 9.885653e-01
        lr_initial_range = 0.000010, 9.986256
        lr_power_range = 0.100811, 1.999659
        opt_momentum_range = 0.0000591, 9.9899859e-01
    elif benchmark == 'cifar100_wideresnet_2048':
        lr_decay_factor_range = 0.010093, 0.989012
        lr_initial_range = 0.000010, 9.779176
        lr_power_range = 0.100708, 1.999376
        opt_momentum_range = 0.000059, 0.998993    

    config = {
        'lr_decay_factor': rescale(lr_decay_factor, lr_decay_factor_range[0], lr_decay_factor_range[1], 1),
        'lr_initial': rescale(lr_initial, lr_initial_range[0], lr_initial_range[1], 10),
        'lr_power': rescale(lr_power, lr_power_range[0], lr_power_range[1], 1),
        'opt_momentum': rescale(opt_momentum, opt_momentum_range[0], opt_momentum_range[1], 10),
    }
    return config


class PD1Function(BaseTestProblem):

    def __init__(self, task_id: str, noise_std: float = None, negate: bool = False, seed: int = 42):
        if task_id not in BENCHMARKS:
            return
        self.seed = seed
        self.dim = 4
        self._bounds = [(0.0, 1.0) for _ in range(self.dim)]
        self.bench = mfpbench.get(task_id, seed=seed, datadir=DATADIR)
        self.task_id = task_id
        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_true(self, X: Tensor, seed=None) -> Tensor:
        args_dict = rescale_arguments(tuple(*X.detach().numpy()), self.task_id)
        config = Configuration(self.bench._configspace, args_dict)
        result = self.bench.query(config)

        # this value should be maximized
        val = result.score

        return Tensor([val])

