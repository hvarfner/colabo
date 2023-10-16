from __future__ import annotations

from typing import Any, Optional
import torch
from torch import Tensor

from botorch.models.model import Model
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.prior_monte_carlo import qPriorLogExpectedImprovement, qPriorMaxValueEntropySearch
from botorch.utils.transforms import (
    concatenate_pending_points,
    t_batch_mode_transform,
)
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.acquisition.prior_monte_carlo import (
    qPriorJointEntropySearch,
    qPriorMaxValueEntropySearch,
)
from botorch import settings
import matplotlib.pyplot as plt

from matplotlib import rcParams

MCMC_DIM = -3


def n(X):
    return X.detach().numpy()


def maybe_plot(X, acq_func, plot_number=999):

    if (X.shape[0] == plot_number) and (X.shape[-1] == 1):
        if hasattr(acq_func, 'prior_model'):
            model = getattr(acq_func, 'prior_model')
        else:
            model = getattr(acq_func, 'model')

        plot_prior = hasattr(acq_func, 'user_prior')
        import matplotlib.pyplot as plt

        def n(X):
            return X.detach().numpy()
        idcs = torch.argsort(X, axis=0).flatten()

        X_new = torch.cat((torch.zeros(1, 1, 1), X[idcs]))
        acq_values = n(acq_func.forward(X_new)).flatten()
        super_acq_values = n(acq_func.forward(X_new, custom_decay=0)).flatten()
        posterior = model.posterior(X_new)
        mean = n(posterior.mean).flatten()
        std = n(torch.sqrt(posterior.variance)).flatten() * 2
        x = n(X_new).flatten()
        fig, ax = plt.subplots(3 + plot_prior, 1, figsize=(16, 12), sharex=True)
        ax[0].plot(x, mean, c='blue')
        ax[0].fill_between(x, mean - std, mean + std, alpha=0.2, color='red')
        ax[0].scatter(model.train_inputs[0], model.train_targets)
        ax[0].set_title('GP', fontsize=20)
        ax[0].legend(fontsize=20)

        ax[1].plot(x, acq_values, label='Prior acq')
        ax[1].plot(x, super_acq_values, label='Acq')
        ax[1].set_title('Acquisition functions', fontsize=20)
        ax[1].legend(fontsize=20)
        if plot_prior:
            ax[3].plot(x, torch.exp(acq_func.user_prior.forward(X_new)),
                       label='Prior dist.')
            ax[3].legend(fontsize=20)
            ax[1].set_title('User Prior', fontsize=20)

            ax[2].scatter(
                n(acq_func.pareto_sets.flatten()),
                n(acq_func.pareto_fronts.flatten()),
                s=1 + 30 * n((torch.pow(acq_func.prior_probs, acq_func.decay_term))), label='Optimal samples')
            ax[1].set_title('Optimal samples', fontsize=20)

        else:
            ax[2].scatter(n(acq_func.pareto_sets.flatten()),
                          n(acq_func.pareto_fronts.flatten()))
        ax[2].legend(fontsize=20)
        plt.legend()
        plt.show()


# FOR THE ACTIVE LEARNING PLOTTING

# A rather ugly solution to avoid circular imports
def train_map_model(batch_model):
    # from botorch.fit import fit_gpytorch_mll
    from botorch.fit import fit_gpytorch_model
    from gpytorch.mlls import ExactMarginalLogLikelihood

    single_model = batch_model.subset_output(idcs=[0])
    mll = ExactMarginalLogLikelihood(single_model.likelihood, single_model)
    fit_gpytorch_model(mll)
    return single_model


def train_all_map(batch_model):
    from botorch.fit import fit_gpytorch_model
    mll = ExactMarginalLogLikelihood(batch_model.likelihood, batch_model)
    fit_gpytorch_model(mll)
    output_map = batch_model(batch_model.train_inputs[0])
    loss_map = -mll(output_map, batch_model.train_targets)


def train_map_model_torch(batch_model):

    # "Loss" for GPs - the marginal log likelihood
    from gpytorch.mlls import ExactMarginalLogLikelihood
    TRAINING_ITER = 200
    num_models = batch_model.train_targets.shape[0]
    for model_idx in range(num_models):
        model = batch_model.subset_output(idcs=[model_idx])
        # Includes GaussianLikelihood parameters
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        train_x = model.train_inputs[0]
        train_y = model.train_targets
        for i in range(TRAINING_ITER):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()


def compute_gaussian_entropy(variance: Tensor) -> Tensor:
    return 0.5 * (torch.log(2 * pi * variance) + 1)


def plot_acq(batch_model, pred_model, X, acq, acq_rev):
    fig, ax = plt.subplots(3, 4, figsize=(32, 20))
    batch_evals = batch_model.posterior(X)
    pred_evals = pred_model.posterior(X)

    average_mean = torch.mean(batch_evals.mean, dim=MCMC_DIM)
    average_std = torch.sqrt(torch.mean(batch_evals.variance, dim=MCMC_DIM))
    map_mean = pred_evals.mean
    map_std = torch.sqrt(pred_evals.variance)

    data = pred_model.train_inputs[0]
    output = pred_model.train_targets
    try:
        ls_map = pred_model.covar_module.lengthscale.flatten().detach().numpy()
        noise_map = pred_model.likelihood.noise.flatten().detach().numpy()
        bm = batch_model.models[0]
        ls_mcmc = bm.covar_module.lengthscale.flatten().detach().numpy()
        noise_mcmc = bm.likelihood.noise.flatten().detach().numpy()
    except:
        ls_map = pred_model.covar_module.base_kernel.lengthscale.flatten().detach().numpy()
        noise_map = pred_model.likelihood.noise.flatten().detach().numpy()
        bm = batch_model.models[0]
        ls_mcmc = bm.covar_module.base_kernel.lengthscale.flatten().detach().numpy()
        noise_mcmc = bm.likelihood.noise.flatten().detach().numpy()

    sort = torch.argsort(X.flatten())
    sp = X.flatten()[sort].detach().numpy()
    a_m = average_mean.flatten()[sort].detach().numpy()
    a_s = average_std.flatten()[sort].detach().numpy()
    m_m = map_mean.flatten()[sort].detach().numpy()
    m_s = map_std.flatten()[sort].detach().numpy()

    ax[0, 0].plot(sp, a_m, color='blue')
    ax[0, 0].fill_between(sp,
                          a_m - 2 * a_s,
                          a_m + 2 * a_s, color='blue', alpha=0.2)

    ax[0, 0].plot(sp, m_m, color='red')
    ax[0, 0].fill_between(sp,
                          m_m - 2 * m_s,
                          m_m + 2 * m_s, color='red', alpha=0.2)

    ax[0, 2].scatter(ls_mcmc, noise_mcmc, color='red', label='MC samples', s=30)
    ax[0, 2].scatter(ls_map, noise_map, color='blue', label='MAP', s=100)

    ax[1, 0].plot(sp, (acq / acq.max()).flatten().detach().numpy(),
                  color='green', label='KL')
    ax[1, 0].plot(sp, (acq_rev / acq_rev.max()).flatten().detach().numpy(),
                  color='purple', label='KL')

    plt.show()


def plot_acq_and_gp(batch_model, X, acq, acq_rev, acq_total):

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig, ax = plt.subplots(3, 4, figsize=(32, 20))
    batch_evals = batch_model.posterior(X, observation_noise=True)
    batch_evals_nonoise = batch_model.posterior(X, observation_noise=True)
    average_mean = torch.mean(batch_evals.mean, dim=MCMC_DIM)
    average_std = torch.sqrt(torch.mean(batch_evals.variance, dim=MCMC_DIM))
    average_std_nonoise = torch.sqrt(torch.mean(
        batch_evals_nonoise.variance, dim=MCMC_DIM))
    num_models = batch_model.train_targets.shape[0]

    # data = pred_model.train_inputs[0]
    # output = pred_model.train_targets
    try:
        # ls_map = pred_model.covar_module.lengthscale.flatten().detach().numpy()
        # noise_map = pred_model.likelihood.noise.flatten().detach().numpy()

        ls_mcmc = batch_model.covar_module.lengthscale.flatten().detach().numpy()
        noise_mcmc = batch_model.likelihood.noise.flatten().detach().numpy()
    except:
        # ls_map = pred_model.covar_module.base_kernel.lengthscale.flatten().detach().numpy()
        # noise_map = pred_model.likelihood.noise.flatten().detach().numpy()
        ls_mcmc = batch_model.covar_module.base_kernel.lengthscale.flatten().detach().numpy()
        noise_mcmc = batch_model.likelihood.noise.flatten().detach().numpy()

    sort = torch.argsort(X.flatten())
    sp = X.flatten()[sort].detach().numpy()
    a_m = average_mean.flatten()[sort].detach().numpy()
    a_s = average_std.flatten()[sort].detach().numpy()
    a_sn = average_std_nonoise.flatten()[sort].detach().numpy()
    data = (batch_model.train_inputs[0][0], batch_model.train_targets[0])
    cmap = [
        '#377eb8', '#ff7f00', '#4daf4a',
        '#f781bf', '#a65628', '#984ea3',
        '#999999', '#e41a1c', '#dede00',
    ]
    for model in range(int(num_models)):
        m_m = batch_evals.mean[sort, model, :, :].detach().numpy().flatten()
        m_s = torch.sqrt(batch_evals.variance)[
            sort, model, :, :].detach().numpy().flatten()
        m_sn = torch.sqrt(batch_evals_nonoise.variance)[
            sort, model, :, :].detach().numpy().flatten()
        current_ax = ax[int(model * 3 / num_models), 1 + model % 3]
        divider = make_axes_locatable(current_ax)

        current_ax.plot(sp, a_m, color='k')
        try:
            current_ax.axhline(
                y=batch_model.eta_value[model], linestyle=':', color='k', linewidth='2')
        except:
            pass
        current_ax.fill_between(sp,
                                a_m - 2 * a_s,
                                a_m + 2 * a_s, color='k', alpha=0.15)

        current_ax.plot(sp, m_m, color=cmap[model])
        current_ax.fill_between(sp,
                                m_m - 2 * m_s,
                                m_m + 2 * m_s, color=cmap[model], alpha=0.4)
        # current_ax.fill_between(sp,
        #                        m_m - 2 * m_sn,
        #                        m_m + 2 * m_sn, color=cmap[model], alpha=0.3)
        current_ax.scatter(data[0], data[1], color='k')
        axShallow = divider.append_axes(
            "bottom", size="50%", pad=0.01, sharex=current_ax)
        axShallow.plot(sp, (acq[:, model, :]).flatten().detach().numpy(),
                       color='dodgerblue', label='mean term')
        axShallow.plot(sp, (acq_rev[:, model, :]).flatten().detach().numpy(),
                       color='darkgoldenrod', label='var term')
        # axShallow.legend()
        ax[2, 0].scatter(ls_mcmc[model], noise_mcmc[model], color=cmap[model], s=40)

    ax[0, 0].plot(sp, a_m, color='k')
    ax[0, 0].fill_between(sp,
                          a_m - 2 * a_s,
                          a_m + 2 * a_s, color='k', alpha=0.2)
    ax[2, 0].set_xscale('log')
    ax[2, 0].set_yscale('log')
    ax[0, 0].scatter(data[0], data[1], color='k')
    # ax[2, 0].scatter(ls_map, noise_map, color='blue', label='MAP', s=100)

    ax[1, 0].plot(sp, (acq.mean(MCMC_DIM)).flatten().detach().numpy(),
                  color='dodgerblue', label='mean term')
    ax[1, 0].plot(sp, (acq_rev.mean(MCMC_DIM)).flatten().detach().numpy(),
                  color='darkgoldenrod', label='var term')

    ax[1, 0].plot(sp, (acq_total).flatten().detach().numpy(),
                  color='k', label='acq')

    ax[1, 0].legend()
    ax[0, 0].set_title('Total posterior')
    ax[1, 0].set_title('Total acqusition function')
    ax[2, 0].set_title('Log Hyperparameter values')
    ax[2, 0].set_ylabel('Log noise')  # this is the lengthscale
    ax[2, 0].set_xlabel('Log lengthscale')  # this is the lengthscale
    ax[2, 0].set_xlim([1e-3, 1e1])  # this is the lengthscale
    ax[2, 0].set_ylim([1e-3, 1e1])  # this is the lengthscale
    plt.tight_layout(pad=5)
    iteration = batch_model.train_targets.shape[1]
    plt.savefig(f'plot_{iteration}.png')
    # import pandas as pd
    # pd.DataFrame(batch_model.eta_value.detach().numpy().flatten()).to_csv(f'eta_{iteration}.csv')
    plt.show()


def plot_acq_and_gp_with_values(
        batch_model,
        X,
        mean_truncated,
        marg_mean,
        cond_means,
        var_truncated,
        marg_variance,
        cond_variances,
        acq_mean,
        acq_var,
        train_X,
        train_Y,
        optimal_inputs,
        optimal_outputs,
        acq_1_name='mean term',
        acq_2_name='var term',

):
    acq_total = (acq_mean + acq_var)
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig, ax = plt.subplots(3, 4, figsize=(32, 20))

    try:
        # ls_map = pred_model.covar_module.lengthscale.flatten().detach().numpy()
        # noise_map = pred_model.likelihood.noise.flatten().detach().numpy()
        if isinstance(batch_model, ModelListGP):
            ls_mcmc = batch_model.models[0].covar_module.lengthscale.flatten(
            ).detach().numpy()
            noise_mcmc = batch_model.models[0].likelihood[0].noise.flatten(
            ).detach().numpy()
        else:
            ls_mcmc = batch_model.covar_module.lengthscale.flatten().detach().numpy()
            noise_mcmc = batch_model.likelihood.noise.flatten().detach().numpy()
    except:
        if isinstance(batch_model, ModelListGP):
            ls_mcmc = batch_model.models[0].covar_module.base_kernel.lengthscale.flatten(
            ).detach().numpy()
            noise_mcmc = batch_model.models[0].likelihood.noise.flatten(
            ).detach().numpy()
        else:
            ls_mcmc = batch_model.covar_module.base_kernel.lengthscale.flatten().detach().numpy()
            noise_mcmc = batch_model.likelihood.noise.flatten().detach().numpy()

    sort = torch.argsort(X.flatten())
    sp = X.flatten()[sort].detach().numpy()

    # just pick one of the num_samples realiations
    a_m = marg_mean.mean(dim=-4)[:, 0, 0].detach().numpy()
    a_s = torch.sqrt(marg_variance.mean(dim=-4)[:, 0, 0]).detach().numpy()
    data = (train_X, train_Y)
    cmap = [
        '#377eb8', '#ff7f00', '#4daf4a',
        '#f781bf', '#a65628', '#984ea3',
        '#999999', '#e41a1c', '#dede00',
    ]
    if isinstance(batch_model, ModelListGP):
        num_models = 9
        num_samples = len(batch_model.models)
    else:
        num_models = 9
        num_samples = 1

    for model in range(9):
        current_ax_group = ax[int(model * 3 / num_models), 1 + model % 3]
        divider = make_axes_locatable(current_ax_group)
        for sample in range(num_samples):
            current_ax = divider.append_axes(
                "top", size="200%", pad=0.01, sharex=current_ax_group)

            current_ax.get_xaxis().set_visible(False)

            m_m = mean_truncated[sort, model, :, sample].detach().numpy().flatten()
            m_s = torch.sqrt(var_truncated)[
                sort, model, :, sample].detach().numpy().flatten()

            current_ax.plot(sp, a_m, color='k')
            try:
                current_ax.axhline(
                    y=batch_model.eta_value[model], linestyle=':', color='k', linewidth='2')
            except:
                pass
            current_ax.fill_between(sp,
                                    a_m - 2 * a_s,
                                    a_m + 2 * a_s, color='k', alpha=0.15)

            current_ax.plot(sp, m_m, color=cmap[model])
            current_ax.fill_between(sp,
                                    m_m - 2 * m_s,
                                    m_m + 2 * m_s, color=cmap[model], alpha=0.4)

            # current_ax.fill_between(sp,
            #                        m_m - 2 * m_sn,
            #                        m_m + 2 * m_sn, color=cmap[model], alpha=0.3)
            if optimal_inputs is not None:
                current_ax.scatter(optimal_inputs[model, sample].flatten(
                ), optimal_outputs[model, sample].flatten(), marker='x', c='k', s=75)
            current_ax.scatter(data[0], data[1], color='k')

        current_ax_group.plot(sp, (acq_total[:, model, :]).detach().squeeze(1).numpy(),
                              color='k', label=acq_1_name)

        # axShallow.legend()
        ax[2, 0].scatter(ls_mcmc[model], noise_mcmc[model], color=cmap[model], s=40)

    ax[0, 0].plot(sp, a_m, color='k')

    ax[0, 0].fill_between(sp,
                          a_m - 2 * a_s,
                          a_m + 2 * a_s, color='k', alpha=0.2)
    ax[2, 0].set_xscale('log')
    ax[2, 0].set_yscale('log')
    ax[0, 0].scatter(data[0], data[1], color='k')
    # ax[2, 0].scatter(ls_map, noise_map, color='blue', label='MAP', s=100)
    ax[1, 0].plot(sp, (acq_mean.mean(MCMC_DIM).mean(-1)).flatten().detach().numpy(),
                  color='dodgerblue', label=acq_1_name)
    ax[1, 0].plot(sp, (acq_var.mean(MCMC_DIM).mean(-1)).flatten().detach().numpy(),
                  color='darkgoldenrod', label=acq_2_name)
    ax[1, 0].plot(sp, (acq_total).mean(MCMC_DIM).mean(-1).flatten().detach().numpy(),
                  color='k', label='acq')

    ax[1, 0].legend()
    ax[0, 0].set_title('Total posterior')
    ax[1, 0].set_title('Total acqusition function')
    ax[2, 0].set_title('Log Hyperparameter values')
    ax[2, 0].set_ylabel('Log noise')  # this is the lengthscale
    ax[2, 0].set_xlabel('Log lengthscale')  # this is the lengthscale
    ax[2, 0].set_xlim([1e-3, 1e1])  # this is the lengthscale
    ax[2, 0].set_ylim([1e-3, 1e1])  # this is the lengthscale
    plt.tight_layout(pad=2)
    # iteration = batch_model.train_targets.shape[1]
    # plt.savefig(f'plot_{iteration}.png')
    # import pandas as pd
    # pd.DataFrame(batch_model.eta_value.detach().numpy().flatten()).to_csv(f'eta_{iteration}.csv')
    plt.show()


def plot_optimal(paths, optimal_inputs, optimal_outputs, batch_model):
    from botorch.models.fully_bayesian import normal_log_likelihood
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    path_points = paths(torch.linspace(0, 1, 201).unsqueeze(-1))

    X = torch.linspace(0, 1, 201).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    fig, ax = plt.subplots(3, 4, figsize=(32, 20))
    batch_evals = batch_model.posterior(X, observation_noise=True)
    batch_evals_nonoise = batch_model.posterior(X, observation_noise=True)
    average_mean = torch.mean(batch_evals.mean.squeeze(1), dim=MCMC_DIM)
    average_std = torch.sqrt(torch.mean(batch_evals.variance.squeeze(1), dim=MCMC_DIM))
    average_std_nonoise = torch.sqrt(torch.mean(
        batch_evals_nonoise.variance.squeeze(1), dim=MCMC_DIM))
    num_models = 9
    # data = pred_model.train_inputs[0]
    # output = pred_model.train_targets
    try:
        # ls_map = pred_model.covar_module.lengthscale.flatten().detach().numpy()
        # noise_map = pred_model.likelihood.noise.flatten().detach().numpy()

        ls_mcmc = batch_model.covar_module.lengthscale.flatten().detach().numpy()
        noise_mcmc = batch_model.likelihood.noise.flatten().detach().numpy()
    except:
        # ls_map = pred_model.covar_module.base_kernel.lengthscale.flatten().detach().numpy()
        # noise_map = pred_model.likelihood.noise.flatten().detach().numpy()
        ls_mcmc = batch_model.covar_module.base_kernel.lengthscale.flatten().detach().numpy()
        noise_mcmc = batch_model.likelihood.noise.flatten().detach().numpy()
        outputscale_mcmc = batch_model.covar_module.outputscale.flatten().detach().numpy()

    noise_mcmc = batch_model.likelihood.noise.flatten().detach().numpy()

    sort = torch.argsort(X.flatten())
    sp = X.flatten()[sort].detach().numpy()
    a_m = average_mean.flatten()[sort].detach().numpy()
    a_s = average_std.flatten()[sort].detach().numpy()
    a_sn = average_std_nonoise.flatten()[sort].detach().numpy()
    data = (batch_model.train_inputs[0][0], batch_model.train_targets[0])
    cmap = [
        '#377eb8', '#ff7f00', '#4daf4a',
        '#f781bf', '#a65628', '#984ea3',
        '#999999', '#e41a1c', '#dede00',
    ]

    train_X = batch_model.train_inputs[0][0]
    train_y = batch_model.train_targets[0].unsqueeze(-1)
    for model in range(9):
        lik = normal_log_likelihood(
            train_X, train_y, outputscale_mcmc[model], noise_mcmc[model], ls_mcmc[model])
        m_m = batch_evals.mean[sort, model, :, :].detach().numpy().flatten()
        m_s = torch.sqrt(batch_evals.variance)[
            sort, model, :, :].detach().numpy().flatten()
        m_sn = torch.sqrt(batch_evals_nonoise.variance)[
            sort, model, :, :].detach().numpy().flatten()
        current_ax = ax[int(model * 3 / num_models), 1 + model % 3]
        current_ax.plot(sp, a_m, color='k')

        current_ax.fill_between(sp,
                                a_m - 2 * a_s,
                                a_m + 2 * a_s, color='k', alpha=0.15)

        current_ax.plot(sp, m_m, color=cmap[model])
        current_ax.fill_between(sp,
                                m_m - 2 * m_s,
                                m_m + 2 * m_s, color=cmap[model], alpha=0.4)
        current_ax.plot(sp, path_points[:, model].T.detach(
        ).numpy(), color='k', linestyle=':', linewidth=1)
        # , markersize=5)
        current_ax.scatter(optimal_inputs[:, model],
                           optimal_outputs[:, model], marker='x', c='k', s=75)
        current_ax.set_title(str(lik.item()))
        # current_ax.fill_between(sp,
        #                        m_m - 2 * m_sn,
        #                        m_m + 2 * m_sn, color=cmap[model], alpha=0.3)
        current_ax.scatter(data[0], data[1], color='k')

        ax[2, 0].scatter(ls_mcmc[model], noise_mcmc[model], color=cmap[model], s=40)

    ax[0, 0].plot(sp, a_m, color='k')
    ax[0, 0].fill_between(sp,
                          a_m - 2 * a_s,
                          a_m + 2 * a_s, color='k', alpha=0.2)
    ax[2, 0].set_xscale('log')
    ax[2, 0].set_yscale('log')
    ax[0, 0].scatter(data[0], data[1], color='k')
    # ax[2, 0].scatter(ls_map, noise_map, color='blue', label='MAP', s=100)

    # ax[1, 0].plot(sp, (acq.mean(MCMC_DIM)).flatten().detach().numpy(),
    #              color='dodgerblue', label='mean term')
    # ax[1, 0].plot(sp, (acq_rev.mean(MCMC_DIM)).flatten().detach().numpy(),
    #              color='darkgoldenrod', label='var term')
    # ax[1, 0].plot(sp, (acq_total).flatten().detach().numpy(),
    #              color='k', label='acq')

    ax[1, 0].legend()
    ax[0, 0].set_title('Total posterior')
    ax[1, 0].set_title('Total acqusition function')
    ax[2, 0].set_title('Log Hyperparameter values')
    ax[2, 0].set_ylabel('Log noise')  # this is the lengthscale
    ax[2, 0].set_xlabel('Log lengthscale')  # this is the lengthscale
    ax[2, 0].set_xlim([1e-3, 1e1])  # this is the lengthscale
    ax[2, 0].set_ylim([1e-3, 1e1])  # this is the lengthscale
    plt.tight_layout(pad=5)
    iteration = batch_model.train_targets.shape[1]
    plt.savefig(f'plot_{iteration}.png')
    # import pandas as pd
    # pd.DataFrame(batch_model.eta_value.detach().numpy().flatten()).to_csv(f'eta_{iteration}.csv')
    plt.show()


def plot_optimal_paper(paths, optimal_inputs, optimal_outputs, batch_model):

    rcParams['font.family'] = 'serif'
    from botorch.models.fully_bayesian import normal_log_likelihood
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    path_points = paths(torch.linspace(0, 1, 201).unsqueeze(-1))

    X = torch.linspace(0, 1, 201).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    fig, ax = plt.subplots(1, 4, figsize=(16, 4), sharey=True, sharex=True)
    batch_evals = batch_model.posterior(X, observation_noise=True)
    batch_evals_nonoise = batch_model.posterior(X, observation_noise=True)
    average_mean = torch.mean(batch_evals.mean.squeeze(1), dim=MCMC_DIM)
    average_std = torch.sqrt(torch.mean(batch_evals.variance.squeeze(1), dim=MCMC_DIM))
    average_std_nonoise = torch.sqrt(torch.mean(
        batch_evals_nonoise.variance.squeeze(1), dim=MCMC_DIM))
    num_models = 3
    # data = pred_model.train_inputs[0]
    # output = pred_model.train_targets
    try:
        # ls_map = pred_model.covar_module.lengthscale.flatten().detach().numpy()
        # noise_map = pred_model.likelihood.noise.flatten().detach().numpy()

        ls_mcmc = batch_model.covar_module.lengthscale.flatten().detach().numpy()
        noise_mcmc = batch_model.likelihood.noise.flatten().detach().numpy()
    except:
        # ls_map = pred_model.covar_module.base_kernel.lengthscale.flatten().detach().numpy()
        # noise_map = pred_model.likelihood.noise.flatten().detach().numpy()
        ls_mcmc = batch_model.covar_module.base_kernel.lengthscale.flatten().detach().numpy()
        noise_mcmc = batch_model.likelihood.noise.flatten().detach().numpy()
        outputscale_mcmc = batch_model.covar_module.outputscale.flatten().detach().numpy()

    noise_mcmc = batch_model.likelihood.noise.flatten().detach().numpy()

    sort = torch.argsort(X.flatten())
    sp = X.flatten()[sort].detach().numpy()
    a_m = average_mean.flatten()[sort].detach().numpy()
    a_s = average_std.flatten()[sort].detach().numpy()
    a_sn = average_std_nonoise.flatten()[sort].detach().numpy()
    data = (batch_model.train_inputs[0][0], batch_model.train_targets[0])
    cmap = [
        '#377eb8', '#ff7f00', '#4daf4a',
        '#f781bf', '#a65628', '#984ea3',
        '#999999', '#e41a1c', '#dede00',
    ]

    train_X = batch_model.train_inputs[0][0]
    train_y = batch_model.train_targets[0].unsqueeze(-1)
    for model in range(3):
        lik = normal_log_likelihood(
            train_X, train_y, outputscale_mcmc[model], noise_mcmc[model], ls_mcmc[model])
        m_m = batch_evals.mean[sort, model, :, :].detach().numpy().flatten()
        m_s = torch.sqrt(batch_evals.variance)[
            sort, model, :, :].detach().numpy().flatten()
        m_sn = torch.sqrt(batch_evals_nonoise.variance)[
            sort, model, :, :].detach().numpy().flatten()
        current_ax = ax[1 + model]
        current_ax.plot(sp, a_m, color='k')

        current_ax.fill_between(sp,
                                a_m - 2 * a_s,
                                a_m + 2 * a_s, color='k', alpha=0.15)
        # Borders of marginal posterior uncertainty
        current_ax.plot(sp, a_m - 2 * a_s, color='k', label='__nolabel__', alpha=0.2)
        current_ax.plot(sp, a_m + 2 * a_s, color='k', label='__nolabel__', alpha=0.2)

        current_ax.plot(sp, m_m, color=cmap[model])
        current_ax.fill_between(sp,
                                m_m - 2 * m_s,
                                m_m + 2 * m_s, color=cmap[model], alpha=0.4)

        # Borders of sample posterior uncertainty
        current_ax.plot(sp, m_m - 2 * m_s,
                        color=cmap[model], label='__nolabel__', alpha=0.5)
        current_ax.plot(sp, m_m + 2 * m_s,
                        color=cmap[model], label='__nolabel__', alpha=0.5)
        if model == 0:
            current_ax.plot(sp, path_points[:, model].T.detach(
            ).numpy(), color='k', linestyle=':', linewidth=1.5, label='Posterior sample')
            # , markersize=5)
            current_ax.scatter(optimal_inputs[:, model],
                               optimal_outputs[:, model], marker='x', c='k', s=100, label='Optimum')
        else:
            current_ax.plot(sp, path_points[:, model].T.detach(
            ).numpy(), color='k', linestyle=':', linewidth=1.5, label='__nolabel__')
            # , markersize=5)
            current_ax.scatter(optimal_inputs[:, model],
                               optimal_outputs[:, model], marker='x', c='k', s=100, label='__nolabel__')

        # current_ax.set_title(str(lik.item()))
        # current_ax.fill_between(sp,
        #                        m_m - 2 * m_sn,
        #                        m_m + 2 * m_sn, color=cmap[model], alpha=0.3)
        current_ax.scatter(data[0], data[1], color='k', s=80)

        # ax[2, 0].scatter(ls_mcmc[model], noise_mcmc[model], color=cmap[model], s=40)

    ax[0].plot(sp, a_m, color='k', label='Posterior mean')
    ax[0].fill_between(sp,
                       a_m - 2 * a_s,
                       a_m + 2 * a_s, color='k', alpha=0.2, label='Posterior uncertainty')
    ax[0].plot(sp, a_m - 2 * a_s, color='k', label='__nolabel__', alpha=0.3)
    ax[0].plot(sp, a_m + 2 * a_s, color='k', label='__nolabel__', alpha=0.3)
    # ax[2, 0].set_xscale('log')
    # ax[2, 0].set_yscale('log')
    ax[0].scatter(data[0], data[1], color='k', s=80, label='Observed data')
    # ax[2, 0].scatter(ls_map, noise_map, color='blue', label='MAP', s=100)

    # ax[1, 0].plot(sp, (acq.mean(MCMC_DIM)).flatten().detach().numpy(),
    #              color='dodgerblue', label='mean term')
    # ax[1, 0].plot(sp, (acq_rev.mean(MCMC_DIM)).flatten().detach().numpy(),
    #              color='darkgoldenrod', label='var term')
    # ax[1, 0].plot(sp, (acq_total).flatten().detach().numpy(),
    #              color='k', label='acq')

    # ax[1, 0].legend()
    fig.legend(loc='upper left', fontsize=14)
    # ax[1, 0].set_title('Total acqusition function')
    # ax[2, 0].set_title('Log Hyperparameter values')
    # ax[2, 0].set_ylabel('Log noise')  # this is the lengthscale
    # ax[2, 0].set_xlabel('Log lengthscale')  # this is the lengthscale
    # ax[2, 0].set_xlim([1e-3, 1e1])  # this is the lengthscale
    # ax[2, 0].set_ylim([1e-3, 1e1])  # this is the lengthscale
    plt.tight_layout(pad=0.5)
    iteration = batch_model.train_targets.shape[1]
    plt.savefig(f'plot_{iteration}.pdf')
    # import pandas as pd
    # pd.DataFrame(batch_model.eta_value.detach().numpy().flatten()).to_csv(f'eta_{iteration}.csv')
    plt.show()


def plot_acq_and_gp_with_values_paper(
        batch_model,
        X,
        mean_truncated,
        cond_means,
        var_truncated,
        cond_variances,
        acq_bo,
        acq_al,
        train_X,
        train_Y,
        optimal_inputs,
        optimal_outputs,
        acq_1_name='mean term',
        acq_2_name='var term',
        cond_type='J',
        plot_sample=True,
        plot_acq=True,
        paths=None,
        dist_mc=None,
        plot_only_gp=False,
):
    ylim_model = [-2.6, 3.3]
    ylim_acq = [0, 0.85]

    show_opt = [1, 0, 1]
    if cond_type == 'W':
        show_opt = [0, 0, 0]

    plot_proportionality = 2
    plot_prop, inv_plot_prop = int(
        100 * (plot_proportionality)), int(100 * (1 / plot_proportionality))
    optimal_inputs = optimal_inputs.detach()
    optimal_outputs = optimal_outputs.detach()
    rcParams['font.family'] = 'serif'
    if (cond_type == 'M') or (cond_type == 'J'):
        plot_mean = mean_truncated
        plot_var = var_truncated
        acq_total = acq_bo

    elif cond_type == 'W':
        plot_mean = cond_means
        plot_var = cond_variances
        acq_total = acq_al

    ref_mean = cond_means.mean(-3, keepdim=True).mean(-1, keepdim=True)
    ref_var = cond_variances.mean(-3, keepdim=True).mean(-1, keepdim=True)

    if paths is not None:
        path_points = paths(torch.linspace(0, 1, 201).unsqueeze(-1))

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    # Was 4.3 in the paper, changing to 5.5 for the presentation
    fig, ax = plt.subplots(1, 4, figsize=(16, 5.5))

    ax[0].set_title('Marginal Posterior', fontsize=17)

    try:
        # ls_map = pred_model.covar_module.lengthscale.flatten().detach().numpy()
        # noise_map = pred_model.likelihood.noise.flatten().detach().numpy()
        if isinstance(batch_model, ModelListGP):
            ls_mcmc = batch_model.models[0].covar_module.lengthscale.flatten(
            ).detach().numpy()
            noise_mcmc = batch_model.models[0].likelihood[0].noise.flatten(
            ).detach().numpy()
        else:
            ls_mcmc = batch_model.covar_module.lengthscale.flatten().detach().numpy()
            noise_mcmc = batch_model.likelihood.noise.flatten().detach().numpy()
    except:
        if isinstance(batch_model, ModelListGP):
            ls_mcmc = batch_model.models[0].covar_module.base_kernel.lengthscale.flatten(
            ).detach().numpy()
            noise_mcmc = batch_model.models[0].likelihood.noise.flatten(
            ).detach().numpy()
        else:
            ls_mcmc = batch_model.covar_module.base_kernel.lengthscale.flatten().detach().numpy()
            noise_mcmc = batch_model.likelihood.noise.flatten().detach().numpy()

    sort = torch.argsort(X.flatten())
    sp = X.flatten()[sort].detach().numpy()
    # just pick one of the num_samples realiations
    a_m = ref_mean.mean(dim=MCMC_DIM)[:, 0, 0].detach().numpy()
    a_s = torch.sqrt(ref_var.mean(dim=MCMC_DIM))[:, 0, 0].detach().numpy()
    data = (train_X, train_Y)
    cmap = [
        '#377eb8', '#ff7f00', '#4daf4a',
        '#f781bf', '#a65628', '#984ea3',
        '#999999', '#e41a1c', '#dede00',
    ]
    if isinstance(batch_model, ModelListGP):
        num_models = 3
        num_samples = len(batch_model.models)
    else:
        num_models = 3
        num_samples = ref_mean.shape[-4]

    sample = 0
    for model in range(3):
        current_ax_group = ax[1 + model]
        current_ax_group.set_ylim(ylim_acq)

        divider = make_axes_locatable(current_ax_group)
        for sample in range(1):
            current_ax = divider.append_axes(
                "top", size=f"{plot_prop}%", pad=0.08, sharex=current_ax_group)

            current_ax.get_xaxis().set_visible(False)

            m_m = plot_mean[sort, show_opt[model], model, :].detach().numpy().flatten()
            m_s = torch.sqrt(plot_var)[
                sort, [show_opt[model]], model, :].detach().numpy().flatten()

            # current_ax.plot(sp, a_m, color='k')
            try:
                current_ax.axhline(
                    y=batch_model.eta_value[model], linestyle=':', color='k', linewidth='2')
            except:
                pass
            a_m = a_m.flatten()
            a_s = a_s.flatten()
            # current_ax.fill_between(sp,
            #                        a_m - 2 * a_s,
            #                        a_m + 2 * a_s, color = 'k', alpha = 0.15)

            # current_ax.plot(sp, a_m - 2 * a_s, color='k', alpha=0.2)
            # current_ax.plot(sp, a_m + 2 * a_s, color='k', alpha=0.2)

            current_ax.plot(sp, m_m, color=cmap[model])
            current_ax.fill_between(sp,
                                    m_m - 2 * m_s,
                                    m_m + 2 * m_s, color=cmap[model], alpha=0.2)
            current_ax.plot(sp, m_m - 2 * m_s, color=cmap[model], alpha=0.6)
            current_ax.plot(sp, m_m + 2 * m_s, color=cmap[model], alpha=0.6)

            # current_ax.fill_between(sp,
            #                        m_m - 2 * m_sn,
            #                        m_m + 2 * m_sn, color=cmap[model], alpha=0.3)
        if model == 0:
            current_ax.set_title('Conditional, large $\sigma_f$', fontsize=18)
        elif model == 1:
            current_ax.set_title(
                'Conditional, large $\sigma_\\varepsilon$', fontsize=18)
        elif model == 2:
            current_ax.set_title('Conditional, small $\ell$', fontsize=18)

        if model == 0:
            if plot_sample:
                current_ax.plot(sp, path_points[:, model].T.detach(
                ).numpy(), color='k', linestyle=':', linewidth=1.5, label='Posterior sample')
            # , markersize=5)

                current_ax.scatter(optimal_inputs[:, model],
                                   optimal_outputs[:, model], marker='*', c='k', s=210, label='Optimum', edgecolor='black')
        else:
            if plot_sample:
                current_ax.plot(sp, path_points[:, model].T.detach(
                ).numpy(), color='k', linestyle=':', linewidth=1.5, label='__nolabel__')
            # , markersize=5)
                current_ax.scatter(optimal_inputs[:, model],
                                   optimal_outputs[:, model], marker='*', c='k', s=210, label='Optimum', edgecolor='black')

        current_ax.scatter(data[0], data[1], color='k', s=100,
                           edgecolor='white', linewidths=1.5)

        current_ax_group.plot(sp, (acq_total[:, :, model, 0, 0]).detach().numpy(),
                              color=cmap[model], label=acq_1_name)

        if dist_mc is not None:
            current_ax_group.plot(sp, (dist_mc[:, :, model, 0, 0]).detach().numpy(),
                                  color=cmap[model], linestyle='dotted', label=acq_1_name + ' MC')
        current_ax.set_ylim(ylim_model)
        # axShallow.legend()
    ax[0].set_ylim(ylim_model)
    ax[0].plot(sp, a_m, color='k', label='Posterior mean')
    ax[0].fill_between(sp,
                       a_m - 2 * a_s,
                       a_m + 2 * a_s, color='k', alpha=0.2, label='Posterior uncertainty')
    ax[0].plot(sp, a_m - 2 * a_s, color='k', label='__nolabel__', alpha=0.3)
    ax[0].plot(sp, a_m + 2 * a_s, color='k', label='__nolabel__', alpha=0.3)
    # ax[2, 0].set_xscale('log')
    # ax[2, 0].set_yscale('log')
    ax[0].scatter(data[0], data[1], color='k', s=90,
                  edgecolor='white', label='Observed data')
    # ax[2, 0].scatter(ls_map, noise_map, color='blue', label='MAP', s=100)

    # ACQ PLOTTING

    divider = make_axes_locatable(ax[0])

    current_ax = divider.append_axes(
        "bottom", size=f"{inv_plot_prop}%", pad=0.08, sharex=ax[0])

    for model in range(num_models):
        current_ax.plot(sp, (acq_total[:, :, model, 0, 0]).detach().numpy(),
                        color=cmap[model], label='__nolabel__', alpha=0.5)

        if dist_mc is not None:
            current_ax.plot(sp, (dist_mc[:, :, model, 0, 0]).detach().numpy(),
                            color=cmap[model], label='__nolabel__', alpha=0.5, linestyle='dotted')

    current_ax.plot(sp, (acq_total).mean(MCMC_DIM, keepdim=True).mean(-4).flatten().detach().numpy(),
                    color='k', label='Avg. acquisition function')
    current_ax.set_ylim(ylim_acq)

    if dist_mc is not None:
        current_ax.plot(sp, (dist_mc).mean(MCMC_DIM, keepdim=True).mean(-4).flatten().detach().numpy(),
                        color='k', linestyle='dotted', label='Avg. acquisition function MC')

    argmax = sp[torch.argmax((acq_total).mean(
        MCMC_DIM).mean(-3).flatten().detach()).numpy()]
    if not plot_only_gp:
        ax[0].axvline(argmax, c='k', linestyle='--')
        current_ax.axvline(argmax, c='k', linestyle='--')

    current_ax.set_ylabel('Acq. value', fontsize=17)
    ax[0].set_ylabel('Function value', fontsize=17)
    ax[0].legend()

    plt.tight_layout(pad=0.5)
    iteration = batch_model.train_targets.shape[1]
    if plot_only_gp:
        plt.savefig(f'plot_{iteration}{cond_type}{plot_sample}gp.pdf')

    else:
        plt.savefig(f'plot_{iteration}{cond_type}{plot_sample}.pdf')

    # plt.show()


def plot_prior(prior_acq, subset: float = 0.0025 / 10, show_acq: bool = True):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    ACQ_BATCHES = 20
    NON_PRIOR_ACQ_NBR = 2048
    rcParams['font.family'] = 'serif'
    def dt(d): return d.detach().numpy().flatten()
    X = torch.linspace(0, 1, 251).unsqueeze(-1).to(torch.float)

    # prior_acq.evaluate(X.unsqueeze(-1), bounds=Tensor([[0, 1]]).to(torch.double).T)
    paths = prior_acq.sampling_model.paths
    old_paths = prior_acq.old_paths
    prior = prior_acq.user_prior

    fig, axes = plt.subplots(1, 2, figsize=(21, 7.5))

    prior_divider = make_axes_locatable(axes[0])
    prior_ax = prior_divider.append_axes("top", size="100%", pad=0.6, sharex=axes[0], sharey=axes[0])
    posterior_divider = make_axes_locatable(axes[1])
    posterior_ax = posterior_divider.append_axes("top", size="100%", pad=0.6, sharex=axes[1], sharey=axes[1])
    prior_moment_ax = axes[0]
    posterior_moment_ax = axes[1]
    old_samples = paths(X)
    samples = paths(X, subset=prior_acq.indices)

    num_subset = int(round(subset * len(old_samples)))

    prior_values = torch.exp(prior_acq.user_prior.forward(X))
    prior_values = torch.pow(prior_values, prior_acq.decay_factor).clamp_min(prior_acq.prior_floor)


    prior_significant = X[(prior_values > prior_values.mean() / 3).flatten()]
    
    old_prior_samples = paths.paths.prior_paths(X)
    prior_samples = paths.paths.prior_paths(X, subset=prior_acq.indices)
    idcs = prior_acq.indices
    plot_idcs = prior_acq.indices[prior_acq.indices < num_subset * 5]
    # logprobs = torch.log(probs)
    for idx in range(len(old_prior_samples[:num_subset])):
        if idx == 0:
            prior_ax.plot(dt(X), dt(old_prior_samples[idx]), alpha=0.6, c='dodgerblue', linewidth=0.3, label='$f_i \sim p(f)$')
        else:
            prior_ax.plot(dt(X), dt(old_prior_samples[idx]), alpha=0.6, c='dodgerblue', linewidth=0.3, label='__nolabel__')

    for i, idx in enumerate(plot_idcs):
        if i == 0:
            prior_ax.plot(dt(X), dt(prior_samples[i]), alpha=0.6, c='navy', linewidth=2, label='$f_i \sim p(f|\pi)$')
        else:
            prior_ax.plot(dt(X), dt(prior_samples[i]), alpha=0.6, c='navy', linewidth=2, label='__nolabel__')

    for idx in range(len(old_samples[:num_subset])):
        if idx == 0:
            posterior_ax.plot(dt(X), dt(old_samples[idx]), alpha=0.6, c='dodgerblue', linewidth=0.3, label='$f_i \sim p(f|D)$')
        else:
            posterior_ax.plot(dt(X), dt(old_samples[idx]), alpha=0.6, c='dodgerblue', linewidth=0.3, label='__nolabel__')

    for i, idx in enumerate(plot_idcs):
        if i == 0:
            posterior_ax.plot(dt(X), dt(samples[i]), alpha=0.6, c='navy', linewidth=2, label='$f_i \sim p(f|D, \pi)$')
        else:
            posterior_ax.plot(dt(X), dt(samples[i]), alpha=0.6, c='navy', linewidth=2, label='__nolabel__')


    prior_ax.plot(X, 1.5 * prior_values.flatten() / prior_values.max() + prior_ax.get_ylim()[0], color='green', label='User Prior', linewidth=2)

    prior_ax.fill_between(dt(prior_significant), *prior_ax.get_ylim(), color='green', alpha=0.15, label='Good region under $\pi$')
    posterior_ax.fill_between(dt(prior_significant), *posterior_ax.get_ylim(), color='green', alpha=0.15, label='Good region under $\pi$')
    prior_ax.grid(visible=True)
    posterior_ax.grid(visible=True)
    
    def remove_border(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim([0, 1])
        ax.spines.left.set_position(('axes', -0.01))
        
        ax.tick_params(axis='x', labelsize=17)
        ax.tick_params(axis='y', labelsize=17)      
        # ax.spines['bottom'].set_visible(False)
        # ax.spines['left'].set_visible(False)
    remove_border(prior_ax)
    remove_border(posterior_ax)
    posterior_ax.scatter(dt(prior_acq.sampling_model.train_inputs[0]), dt(
        prior_acq.sampling_model.train_targets), c='r', marker='.', s=200, label='Observations', zorder=10)
    posterior_moment_ax.scatter(dt(prior_acq.sampling_model.train_inputs[0]), dt(
        prior_acq.sampling_model.train_targets), c='r', marker='.', s=200, label='__nolabel__', zorder=10)
    
    # ax.scatter(dt(prior_acq.optimal_inputs.flatten()), dt(
    #    prior_acq.optimal_outputs.flatten()), c='k', marker='x', s=100)
    prior_ax.set_title('Samples from prior', fontsize=25)
    posterior_ax.set_title('Pathwise updated posterior samples' , fontsize=25)

    prior_moment_ax.set_title('Moments of prior', fontsize=25)
    posterior_moment_ax.set_title('Moments of posterior' , fontsize=25)
   
    old_prior_moments = old_prior_samples.mean(dim=0), old_prior_samples.std(dim=0)
    prior_moments = prior_samples.mean(dim=0), prior_samples.std(dim=0)
    prior_mean = prior_moments[0]
    prior_std_low = (torch.pow(torch.clamp_min(prior_mean - prior_samples, 0), 2) / ((prior_mean - prior_samples) > 0).sum(dim=0)).sum(dim=0).sqrt()
    prior_std_high = (torch.pow(torch.clamp_min(prior_samples - prior_mean, 0), 2) / ((prior_samples - prior_mean) > 0).sum(dim=0)).sum(dim=0).sqrt()
    
    old_posterior_moments = old_samples.mean(dim=0), old_samples.std(dim=0)
    
    
    posterior_moments = samples.mean(dim=0), samples.std(dim=0)
    posterior_mean = posterior_moments[0]
    
    posterior_std_low = (torch.pow(torch.clamp_min(posterior_mean - samples, 0), 2) / ((posterior_mean - samples) > 0).sum(dim=0)).sum(dim=0).sqrt()
    posterior_std_high = (torch.pow(torch.clamp_min(samples - posterior_mean, 0), 2) / ((samples - posterior_mean) > 0).sum(dim=0)).sum(dim=0).sqrt()
    # The old samples before resampling
    prior_moment_ax.plot(dt(X), (dt(old_prior_moments[0])), color='dodgerblue', linestyle='dashed', linewidth=2, label='Prior moments of $p(f)$')
    posterior_moment_ax.plot(dt(X), (dt(old_posterior_moments[0])), color='dodgerblue', linestyle='dashed', linewidth=2, label='Posterior moments of $p(f|D)$')

    prior_moment_ax.plot(dt(X), (dt(old_prior_moments[0] - 2 * old_prior_moments[1])), color='dodgerblue', linestyle='dashed')
    posterior_moment_ax.plot(dt(X), (dt(old_posterior_moments[0] - 2 * old_posterior_moments[1])), color='dodgerblue', linestyle='dashed')

    prior_moment_ax.plot(dt(X), (dt(old_prior_moments[0] + 2 * old_prior_moments[1])), color='dodgerblue', linestyle='dashed')
    posterior_moment_ax.plot(dt(X), (dt(old_posterior_moments[0] + 2 * old_posterior_moments[1])), color='dodgerblue', linestyle='dashed')

    # and after resampling
    prior_moment_ax.plot(dt(X), (dt(prior_moments[0])), color='navy', label='Prior moments of $p(f|\pi)$')
    posterior_moment_ax.plot(dt(X), (dt(posterior_moments[0])), color='navy', label='Posterior moments of $p(f|D, \pi)$')

    prior_moment_ax.fill_between(dt(X), dt(prior_moments[0] - 2 * prior_std_low), 
        dt(prior_moments[0] + 2 * prior_std_high), color='navy', alpha=0.2)
    posterior_moment_ax.fill_between(dt(X), dt(posterior_moments[0] - 2 * posterior_std_low), 
        dt(posterior_moments[0] + 2 * posterior_std_high), color='navy', alpha=0.2)

    prior_moment_ax.grid(visible=True)
    posterior_moment_ax.grid(visible=True)


    prior_moment_ax.fill_between(dt(prior_significant), *prior_ax.get_ylim(), color='green', alpha=0.15)
    posterior_moment_ax.fill_between(dt(prior_significant), *posterior_ax.get_ylim(), color='green', alpha=0.15)
    remove_border(prior_moment_ax)
    remove_border(posterior_moment_ax)

    prior_ax.legend(fontsize=17, loc='lower right')
    posterior_ax.legend(fontsize=17, loc='lower right')
    prior_moment_ax.legend(fontsize=17, loc='lower right')
    posterior_moment_ax.legend(fontsize=17, loc='lower right')


    if show_acq:
        fig2, axes2 = plt.subplots(2, figsize=(10.5, 7.3))
        

        posterior_acq_ax = axes2[1]
        posterior_ax = axes2[0]
                             
        for idx in range(len(old_samples[:num_subset])):
            if idx == 0:
                posterior_ax.plot(dt(X), dt(old_samples[idx]), alpha=0.6, c='dodgerblue', linewidth=0.3, label='$f_i \sim p(f|D)$')
            else:
                posterior_ax.plot(dt(X), dt(old_samples[idx]), alpha=0.6, c='dodgerblue', linewidth=0.3, label='__nolabel__')

        for i, idx in enumerate(plot_idcs):
            if i == 0:
                posterior_ax.plot(dt(X), dt(samples[i]), alpha=0.6, c='navy', linewidth=2, label='$f_i \sim p(f|D, \pi)$')
            else:
                posterior_ax.plot(dt(X), dt(samples[i]), alpha=0.6, c='navy', linewidth=2, label='__nolabel__')
        posterior_ax.fill_between(dt(prior_significant), *posterior_ax.get_ylim(), color='green', alpha=0.15, label='Good region under $\pi$')
    
        posterior_ax.scatter(dt(prior_acq.sampling_model.train_inputs[0]), dt(
            prior_acq.sampling_model.train_targets), c='r', marker='.', s=200, label='Observations', zorder=10)
            
        
        acq = torch.empty(0)
        import numpy as np
        for batch in range(ACQ_BATCHES):
            split_idx = np.floor(batch * (len(X) / ACQ_BATCHES)).astype(int), np.floor((batch + 1) * (len(X) / ACQ_BATCHES)).astype(int)
            X_batch = X[split_idx[0]: split_idx[1]].unsqueeze(-1)        
            acq = torch.cat((acq, torch.exp(prior_acq(X_batch))))
            #prior_ax.remove()
            #prior_moment_ax.remove()
            #prior_acq_ax.remove()

        prior_acq.sampling_model.set_paths(prior_acq.old_paths, torch.arange(NON_PRIOR_ACQ_NBR))
        prior_acq.sample_probs = torch.ones(NON_PRIOR_ACQ_NBR).unsqueeze(-1).unsqueeze(-1)
        non_prior_acq = torch.empty(0)
        for batch in range(ACQ_BATCHES):
            split_idx = np.floor(batch * (len(X) / ACQ_BATCHES)).astype(int), np.floor((batch + 1) * (len(X) / ACQ_BATCHES)).astype(int)
            X_batch = X[split_idx[0]: split_idx[1]].unsqueeze(-1)        
            non_prior_acq = torch.cat((non_prior_acq, torch.exp(prior_acq(X_batch))))
        
        posterior_acq_ax.plot(dt(X), dt(acq), color='navy', linewidth=2, label='ColaBO EI')
        posterior_acq_ax.plot(dt(X), dt(non_prior_acq), color='dodgerblue', linewidth=2, label='Vanilla EI', linestyle='dashed')
        posterior_acq_ax.fill_between(dt(prior_significant), *posterior_acq_ax.get_ylim(), color='green', alpha=0.15)
        posterior_acq_ax.legend(fontsize=17, loc='lower right')

        remove_border(posterior_acq_ax)
        remove_border(posterior_ax)
        posterior_acq_ax.grid(visible=True)
        posterior_ax.grid(visible=True)
        posterior_ax.legend(fontsize=17, loc='lower right')
        posterior_acq_ax.set_title('Effect on Acquisition Function' , fontsize=25)
        posterior_ax.set_title('Samples from posterior' , fontsize=25)

        fig2.tight_layout()
        fig2.savefig(f'fig_iter{len(prior_acq.sampling_model.train_targets)}0_acq.pdf')
        #fig2.show()
        #fig.delaxes(prior_acq_ax)
        #fig.delaxes(prior_ax)
        #fig.delaxes(prior_moment_ax)
        #fig.delaxes(posterior_moment_ax)

    #
    fig.tight_layout()
    fig.savefig(f'fig_iter{len(prior_acq.sampling_model.train_targets)}0.pdf')
    plt.show()


def plot_paper(prior_acq):
    from mpl_toolkits.axes_grid1 import make_axes_locatable    
    def dt(d): return d.detach().numpy()
    X = torch.linspace(0, 1, 201).unsqueeze(-1).to(torch.double)

    # prior_acq.evaluate(X.unsqueeze(-1), bounds=Tensor([[0, 1]]).to(torch.double).T)
    paths = prior_acq.sampling_model.paths
    prior = prior_acq.user_prior

    fig, axes = plt.subplots(1, 2, figsize=(36, 12))

    for ax_idx in range(2):
        prior_on = ax_idx == 1
        ax = axes[ax_idx]

        divider = make_axes_locatable(ax)
        prior_ax = divider.append_axes("top", size="200%", pad=0.6, sharex=ax)
        acq_ax = divider.append_axes("bottom", size="200%", pad=0.6, sharex=ax)

        samples = paths(X)
        prior_paths = paths.paths.prior_paths
        prior_samples = paths.paths.prior_paths(X)
        if prior is not None:
            probs = prior.compute_norm_probs(
                paths.paths.prior_paths, prior_acq.decay_factor * int(prior_on))
        else:
            probs = torch.ones(torch.Size([1, len(samples)]))

        max_prob = probs.max()
        probs = probs / max_prob

        if prior_acq.user_prior is not None:
            prior_probs = torch.exp(prior_acq.user_prior.forward(X).flatten())
        else:
            prior_probs = torch.ones_like(X).flatten()
        prior_probs = prior_probs / prior_probs.mean()
        # logprobs = torch.log(probs)
        for idx in range(len(samples)):
            prior_ax.plot(dt(X), dt(prior_samples[idx]), alpha=dt(
                probs)[0, idx], c='blue', linewidth=0.5)
        ax.plot(dt(X).flatten(), dt(prior_probs))
        for idx in range(len(samples)):
            acq_ax.plot(dt(X), dt(samples[idx]), alpha=dt(
                probs)[0, idx], c='blue', linewidth=0.5)

    plt.tight_layout()
    # plt.savefig(f'fig_iter{len(prior_acq.sampling_model.train_targets)}.pdf')
    plt.show()


def plot_surface(acq):
    """[summary]

    Args:
        function_name (str): The function to plot.
        seed (int, optional): Which run of the specified function to plot. Defaults to 1.
        num_iters (int, optional): How many iterations of the run to plot. Defaults to -1 (all).
    """
    import numpy as np

    from botorch.acquisition.logei import qLogNoisyExpectedImprovement
    from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy 
    from botorch.acquisition.prior_monte_carlo import (
        qPriorLogExpectedImprovement,
        qPriorMaxValueEntropySearch
    )
    if isinstance(acq, qPriorLogExpectedImprovement):
        bench_acq = qLogNoisyExpectedImprovement(acq.old_model, acq.X_baseline)
    if isinstance(acq, qPriorMaxValueEntropySearch):
        bench_acq = qPriorMaxValueEntropySearch(acq.old_model)
    
    MESH_RESOLUTION = 41
    LEVELS = 26
    fig, ax = plt.subplots(2, 2, figsize=(16, 10))

    if hasattr(acq, 'exploit'):
        exploit = acq.exploit
        acq.exploit = False
    else:
        exploit = False

    X_norm = np.linspace(0, 1, MESH_RESOLUTION)
    bounds = np.array([[0, 0], [1, 1]]).T
    X1 = X_norm * (bounds[0, 1] - bounds[0, 0]) + bounds[0, 0]
    X2 = X_norm * (bounds[1, 1] - bounds[1, 0]) + bounds[1, 0]

    X1, X2 = np.meshgrid(X1, X2)
    X_flat = torch.cat(
        (Tensor(X1).reshape(-1, 1), Tensor(X2).reshape(-1, 1)), dim=1).unsqueeze(-2)

    mean = np.empty((MESH_RESOLUTION, MESH_RESOLUTION))
    std = np.empty((MESH_RESOLUTION, MESH_RESOLUTION))
    acqval = np.empty((MESH_RESOLUTION, MESH_RESOLUTION))
    print('Inside plot')
    for idx in range(MESH_RESOLUTION):
        X_flat_batch = X_flat[idx * MESH_RESOLUTION: (idx + 1) * MESH_RESOLUTION]
        posterior = acq.sampling_model.posterior(X_flat_batch)

        mean[idx, :] = posterior.mean.flatten().detach().numpy()
        std[idx, :] = np.log(posterior.variance.sqrt().flatten().detach().numpy())
        if isinstance(acq, qPriorLogExpectedImprovement):
            acqval[idx, :] = torch.exp(acq(X_flat_batch).flatten()).detach().numpy()

        else:
            acqval[idx, :] = np.log(acq(X_flat_batch).flatten().detach().numpy())

    upcoming = acqval.argmax()
    upcoming_X1 = X1.flatten()[upcoming]
    upcoming_X2 = X2.flatten()[upcoming]
    argmax = mean.argmax()
    argmax_X1 = X1.flatten()[argmax]
    argmax_X2 = X2.flatten()[argmax]

    bench_acqval = torch.exp(bench_acq(X_flat)).reshape(MESH_RESOLUTION, MESH_RESOLUTION).detach().numpy()
    upcoming_bench = bench_acqval.argmax()
    upcoming_X1_bench = X1.flatten()[upcoming_bench]
    upcoming_X2_bench = X2.flatten()[upcoming_bench]
    cm = plt.cm.get_cmap('RdBu')
    ax[0, 0].contourf(X1, X2, mean, levels=LEVELS)

    # TODO plot log std?
    ax[0, 1].contourf(X1, X2, std, levels=LEVELS)

    ax[1, 0].contourf(X1, X2, bench_acqval, levels=LEVELS)
    ax[1, 0].set_title('Bench acq.')
    ax[1, 1].contourf(X1, X2, acqval, levels=LEVELS)
    ax[1, 1].set_title('Prior acq.')

    X1_data = acq.model.train_inputs[0][:, 0]
    X2_data = acq.model.train_inputs[0][:, 1]
    for i in range(2):
        for j in range(2):
            sc = ax[i, j].scatter(X1_data, X2_data, c=range(len(X2_data)),
                            vmin=0, vmax=len(X2_data), s=350, cmap=cm, edgecolors='k', linewidths=2, alpha=0.5)
            sc2 = ax[i, j].scatter(upcoming_X1, upcoming_X2, s=300, c='dodgerblue',
                                edgecolors='k', linewidths=2, alpha=1, marker='*')

            sc3 = ax[i, j].scatter(upcoming_X1_bench, upcoming_X2_bench, s=300, c='pink', edgecolors='k', marker='v', linewidths=2, alpha=0.5)

    if hasattr(acq, 'exploit'):
        acq.exploit = exploit
    if exploit:
        fig.suptitle('Exploit', fontsize=24)
    else:
        fig.suptitle('Normal', fontsize=24)

    fig.tight_layout()
    plt.savefig(f'ackleyiter{len(acq.model.train_inputs[0])}.pdf')
    # plt.show()
    return torch.Tensor([upcoming_X1, upcoming_X2]).unsqueeze(0)


def plot_ves(ves):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    def dt(d): return d.detach().numpy()
    X = torch.linspace(0, 1, 201).unsqueeze(-1).unsqueeze(-1).to(torch.double)

    fig, axes = plt.subplots(3, 1, figsize=(20, 10))

    posterior = ves.model.posterior(X)
    m = dt(posterior.mean)
    s = dt(posterior.variance.sqrt())

    samples = dt(ves.sampling_model.posterior(X).rsample())
    opts_in = dt(ves.optimal_inputs)
    opts_out = dt(ves.optimal_outputs)
    # for i in range(len(opts_in)):
    #    print(opts_out[i] - samples[i].max())
    #plt.plot(dt(X).flatten(), samples[i])
    #plt.scatter(opts_in[i], opts_out[i])
    # plt.show()

    b = 1.1, 1.1
    best_params = Tensor([b[0]]), Tensor([[b[1]]])
    betas = 1, 2, 4
    gammas = 1.001, 1.01, 1.1

    axes[0].plot(dt(X).flatten(), m.flatten())
    axes[0].scatter(dt(ves.model.train_inputs[0]), dt(ves.model.train_targets))
    axes[0].fill_between(dt(X).flatten(), (m - 2 * s).flatten(), (m + 2 * s).flatten(), alpha=0.2)

    c = ['navy', 'forestgreen', 'brown']
    alpha = [0.3, 0.7, 1]
    for b_idx, beta in enumerate(betas):
        for g_idx, gamma in enumerate(gammas):
            tensorized_params = (Tensor([beta]), Tensor([gamma]))
            res = dt(ves(X, *tensorized_params))
            axes[1].plot(
                X.flatten(), res, label=f'Beta: {beta}, k: {gamma}', alpha=alpha[g_idx], color=c[b_idx])
            axes[1].axvline(
                X.flatten()[res.argmax()], label=f'__nolabel__', alpha=alpha[g_idx], color=c[b_idx])

    reg, ei, mv = ves(X, *best_params, split=True)
    axes[2].plot(X.flatten(), dt(reg.flatten() * torch.ones_like(X.flatten())), label=f'Reg', color='purple')
    axes[2].plot(X.flatten(), dt(ei).flatten(), label=f'ei'.upper(), color='orange')
    axes[2].plot(X.flatten(), dt(mv).flatten(), label=f'mv'.upper(), color='teal')
    axes[2].plot(X.flatten(), dt(reg + ei + mv).flatten(), label=f'Total', color='black')
    axes[2].set_title('Beta: ' + str(b[0]) + ',   k: ' + str(b[1]), fontsize=18)


    plt.tight_layout()
    axes[1].legend()
    axes[2].legend()
    #plt.savefig(f'fig_iter{len(prior_acq.sampling_model.train_targets)}.pdf')
    plt.show()





def plot_vanilla(acq):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    def dt(d): return d.detach().numpy()
    X = torch.linspace(0, 1, 501).unsqueeze(-1).unsqueeze(-1).to(torch.double).requires_grad_(True)

    fig, axes = plt.subplots(3, 1, figsize=(20, 10), sharex=True)

    posterior = acq.model.posterior(X)
    m = dt(posterior.mean)
    s = dt(posterior.variance.sqrt())


    axes[0].plot(dt(X).flatten(), m.flatten())
    axes[0].fill_between(dt(X).flatten(), (m - 2 * s).flatten(), (m + 2 * s).flatten(), alpha=0.2)
    axes[0].scatter(dt(acq.model.train_inputs[0]), dt(acq.model.train_targets))

    #from torch.func import vmap, jacrev
    #def acq_func(X):
    #    return acq(X).unsqueeze(-1).unsqueeze(-1)
    #breakpoint()
    #res = dt(acq(X))
    res_grad = dt(torch.autograd.grad(acq(X).sum(), X)[0])
    
    breakpoint()
    axes[1].plot(X.flatten().detach().numpy(), res.flatten(), label=f'Acq', color='k')
    axes[2].plot(X.flatten().detach().numpy(), res_grad.flatten(), label=f'Acq', color='k')
    #axes[3].plot(X.flatten().detach().numpy(), res_grad2.flatten(), label=f'Acq', color='k')
    for ax in axes[1:]:
        for X in acq.model.train_inputs[0]:
            ax.axvline(dt(X.flatten()), *ax.get_ylim())

    plt.tight_layout()
    plt.legend()
    # plt.savefig(f'fig_iter{len(prior_acq.sampling_model.train_targets)}.pdf')
    plt.show()

