import torch
import numpy as np
from gpytorch.kernels import ScaleKernel
from gpytorch.means import ConstantMean
from botorch.models.transforms import Warp
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import AdditiveKernel
from botorch.utils.transforms import unnormalize, normalize

from botorch.acquisition.monte_carlo import (
    qNoisyExpectedImprovement,
    qUpperConfidenceBound
)
from botorch.acquisition.max_value_entropy_search import (
    qLowerBoundMaxValueEntropy
)
from botorch.acquisition.prior_monte_carlo import (
    qPriorExpectedImprovement,
    qPriorUpperConfidenceBound,
    qPriorMaxValueEntropySearch
)


def get_model_hyperparameters(model, current_data, scale_hyperparameters=True, objective=None, acquisition=None):

    has_outputscale = isinstance(model.covar_module, ScaleKernel)
    has_mean = isinstance(model.mean_module, ConstantMean)

    def tolist(l): return l.detach().to(torch.float32).numpy().tolist()
    def d(l): return l.detach().to(torch.float32).numpy().flatten()
    hp_dict = {}

    data_mean = current_data.mean()

    training_data = model.train_inputs[0]

    from torch.quasirandom import SobolEngine
    draws = SobolEngine(dimension=training_data.shape[-1]).draw(512)
    if scale_hyperparameters:
        data_variance = current_data.var()
    # print('Data variance', data_variance)
    else:
        data_variance = torch.Tensor([1])
    if has_outputscale:
        hp_dict['outputscale'] = tolist(model.covar_module.outputscale * data_variance)
        if isinstance(model.covar_module.base_kernel, AdditiveKernel):
            hp_dict['lengthscales'] = tolist(
                model.covar_module.base_kernel.kernels[0].lengthscale)
            hp_dict['active_groups'] = []
            for kernel in model.covar_module.base_kernel.kernels:
                hp_dict['active_groups'].append(tolist(kernel.active_dims))
        else:
            hp_dict['lengthscales'] = tolist(model.covar_module.base_kernel.lengthscale)
        hp_dict['noise'] = tolist(model.likelihood.noise * data_variance)
    else:
        if isinstance(model.covar_module, AdditiveKernel):
            hp_dict['active_groups'] = []
            hp_dict['lengthscales'] = tolist(model.covar_module.kernels[0].lengthscale)
            hp_dict['noise'] = tolist(model.likelihood.noise)
            for kernel in model.covar_module.kernels:
                hp_dict['active_groups'].append(tolist(kernel.active_dims))
        else:
            hp_dict['lengthscales'] = tolist(model.covar_module.lengthscale)
            hp_dict['noise'] = tolist(model.likelihood.noise)

    if has_mean:
        hp_dict['mean'] = tolist(model.mean_module.constant
                                 * data_variance ** 0.5 - data_mean)

    return hp_dict


def compute_ws_and_acq(ax_client, test_samples, objective, objective_name, local=False, local_frac=0.1):

    model = ax_client._generation_strategy.model.model.surrogate.model
    ts = test_samples.unsqueeze(-2)
    if local:
        best = model.train_targets.argmax()
        best_X = model.train_inputs[0][best].unsqueeze(0).unsqueeze(1)
        ts = best_X + ts * local_frac

    btmodel = ax_client._generation_strategy.model.model
    acqf = btmodel._acqf
    if isinstance(acqf, qPriorMaxValueEntropySearch):
        real_acqf = qLowerBoundMaxValueEntropy(
            model, candidate_set=test_samples, input_max_values=acqf.optimal_outputs)
    elif isinstance(acqf, qPriorExpectedImprovement):
        real_acqf = qNoisyExpectedImprovement(model, X_baseline=model.train_inputs[0])
    elif isinstance(acqf, qPriorUpperConfidenceBound):
        beta = 0.2 * math.log(len(model.train_targets) ** 2)
        real_acqf = qUpperConfidenceBound(model, beta)
    else:
        raise ValueError

    sampling_model = acqf.sampling_model
    real_posterior = model.posterior(ts)
    sample_posterior = sampling_model.posterior(ts)

    real_mean = real_posterior.mean
    real_var = real_posterior.variance
    sample_mean = sample_posterior.mean
    sample_var = sample_posterior.variance
    real_acq_value = real_acqf(ts)
    prior_acq_value = acqf(ts)
    ws = torch.pow(real_mean - sample_mean, 2) + real_var + \
        sample_var - 2 * (real_var * sample_var).sqrt()
    # Log (prior acq value / real acq value?) med fillna?
    return ws.sqrt(), torch.pow(prior_acq_value - real_acq_value, 2).mean(-1).sqrt()


def compute_rmse_and_nmll(ax_client, test_samples, objective, objective_name, location=None):
    TS_SPLIT = 10
    LOC_CONST = 0.025
    split_len = int(len(test_samples) / TS_SPLIT)
    results = {'global': {'MLL': 0, 'RMSE': 0}, 'local': {'MLL': 0, 'RMSE': 0}}
    location = normalize(torch.Tensor(location).unsqueeze(0), objective.bounds)

    test_location = location.shape[-1] == test_samples.shape[-1]

    for version, loc_const in zip(['global', 'local'], [1, LOC_CONST]):
        for split_idx in range(TS_SPLIT):
            split_idx_low, split_idx_high = split_idx * \
                split_len, (1 + split_idx) * split_len
            if version == 'local' and test_location:
                test_sample_batch = location + test_samples[split_idx_low:split_idx_high] * \
                    (2 * loc_const) - loc_const

            else:
                test_sample_batch = test_samples[split_idx_low:split_idx_high]

            output = - \
                objective.evaluate_true(unnormalize(
                    test_sample_batch, objective.bounds))
            y_transform = ax_client._generation_strategy.model.transforms['StandardizeY']
            y_mean, y_std = y_transform.Ymean[objective_name], y_transform.Ystd[objective_name]

            mu, cov = ax_client._generation_strategy.model.model.predict(
                test_sample_batch)
            mu_true = (mu * y_std + y_mean).flatten()

            model = ax_client._generation_strategy.model.model.surrogate.model
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            model.eval()
            preds = model(test_sample_batch)
            norm_yvalid = (output - y_mean) / y_std

            norm_yvalid = norm_yvalid.flatten()
            # marg_dist = MultivariateNormal(predmean, predcov)
            # joint_loglik = -mll(marg_dist, norm_yvalid).mean()
            results[version]['MLL'] = results[version]['MLL'] - \
                mll(preds, norm_yvalid).mean().item()
            results[version]['RMSE'] = results[version]['RMSE'] + \
                torch.pow(output - mu_true, 2).mean().item()

    return results['global']['MLL'] / TS_SPLIT, results['local']['MLL'] / TS_SPLIT, results['global']['RMSE'] / TS_SPLIT, results['local']['RMSE'] / TS_SPLIT


def plot_gp(model, acquisition=None, objective=None):

    if model.train_inputs[0].shape[-1] > 2:
        return

    N_GRID = 101
    import matplotlib.pyplot as plt
    X = torch.linspace(0, 1, N_GRID)
    Y = X
    grid = torch.stack(torch.meshgrid(X, Y))
    grid_flat = grid.reshape(2, -1).T
    posterior = model.posterior(grid_flat, observation_noise=False)
    mean = posterior.mean
    std = posterior.variance.sqrt()
    data = model.train_inputs[0]
    if mean.ndim > 2:
        # means that they have a batch dim, not really interesting here
        mean = mean[0]
        std = std[0]
        data = data[0]

    plot_bump = 0
    if objective is not None:
        plot_bump += 1
    if acquisition is not None:
        plot_bump += 1

    def d(x): return x.detach().numpy()
    grid = d(grid)
    data = d(data)
    fig, ax = plt.subplots(1, 2 + plot_bump, figsize=(16, 10),
                           sharex=True, sharey=True)

    print(std.max())
    cb0 = ax[0 + plot_bump].contourf(grid[0], grid[1], d(mean.reshape(N_GRID, N_GRID)))
    cb1 = ax[1 + plot_bump].contourf(grid[0], grid[1], d(std.reshape(N_GRID, N_GRID)))

    ax[0 + plot_bump].scatter(data[:, 0], data[:, 1], s=200,
                              c='white', edgecolors='k', linewidths=2)
    ax[1 + plot_bump].scatter(data[:, 0], data[:, 1], s=200,
                              c='white', edgecolors='k', linewidths=2)
    plt.colorbar(cb0)
    plt.colorbar(cb1)

    if objective is not None:
        obj = objective.evaluate_true(grid_flat)
        cb_fun = ax[0].contourf(grid[0], grid[1], d(obj.reshape(N_GRID, N_GRID)))
        ax[0].scatter(data[:, 0], data[:, 1], s=200,
                      c='white', edgecolors='k', linewidths=2)
        plt.colorbar(cb_fun)

    if acquisition is not None:
        obj = acquisition(grid_flat)
        cb_acq = ax[0].contourf(grid[0], grid[1], d(obj.reshape(N_GRID, N_GRID)))
        ax[0 + plot_bump - 1].scatter(data[:, 0], data[:, 1],
                                      s=200, c='white', edgecolors='k', linewidths=2)
        plt.colorbar(cb_acq)

    plt.tight_layout()
    plt.show()
