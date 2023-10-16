import os
from os.path import abspath, dirname, join
import sys
import yaml
import json

import torch
from torch import Tensor
from botorch.models import FixedNoiseGP
from botorch.fit import fit_gpytorch_mll
from torch.quasirandom import SobolEngine
from benchmarking.mappings import get_test_function
from botorch.utils.transforms import unnormalize, normalize, standardize
from gpytorch.mlls import ExactMarginalLogLikelihood


benchmark = sys.argv[1]
seed = int(sys.argv[2])
path_to_file = os.path.join(dirname(dirname(abspath(__file__))),
                            f'configs/benchmark/{benchmark}.yaml')

with open(path_to_file, 'r') as f:
    benchmark = yaml.load(f, Loader=yaml.Loader)

function = get_test_function(benchmark['name'], benchmark['noise_std'])
bounds = Tensor(benchmark['bounds'])

sobol = SobolEngine(dimension=len(bounds), scramble=True, seed=seed)
train_X = sobol.draw(int(1e4)).to(torch.float64)
train_X_unnorm = unnormalize(train_X, bounds.T)
train_Y_unnorm = function.evaluate_true(train_X_unnorm).unsqueeze(-1)
print(train_Y_unnorm)
train_Y = standardize(train_Y_unnorm)

train_Y_std = train_Y_unnorm.std()
norm_Yvar = torch.pow(Tensor([benchmark['noise_std']]) / train_Y_std, 2)


print(norm_Yvar, benchmark['noise_std'], train_Y_std)
#raise SystemExit
train_Yvar = 1e-12 * torch.ones_like(train_Y)
model = FixedNoiseGP(train_X=train_X, train_Y=train_Y, train_Yvar=train_Yvar)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)


def d(t): return t.detach().numpy().tolist()

hyperparameters = {}
hyperparameters['lengthscales'] = d(model.covar_module.base_kernel.lengthscale)
hyperparameters['outputscale'] = d(model.covar_module.outputscale)
hyperparameters['noise'] = d(norm_Yvar)

save_path = os.path.join(dirname(abspath(__file__)),
                         f'hyperparameters/{benchmark["name"]}_hps_seed{seed}.json')
with open(save_path, 'w') as f:
    json.dump(hyperparameters, f)
