import numpy as np
import torch
from torch import Tensor
import matplotlib.pyplot as plt
from torch.quasirandom import SobolEngine

from botorch.fit import fit_gpytorch_model
from botorch.optim import optimize_acqf
from botorch.models import SingleTaskGP, FixedNoiseGP
from botorch.optim.fit import fit_gpytorch_torch
from botorch.utils.prior import DefaultPrior, UserPrior
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.transforms import normalize, standardize, unnormalize
from botorch.test_functions import *
from botorch.acquisition.joint_entropy_search import (
    qLowerBoundJointEntropySearch,
    qExploitLowerBoundJointEntropySearch,
)
from botorch.acquisition.user_prior_entropy import (
    qPriorLowerBoundJointEntropySearch,
    qPriorExploitLowerBoundJointEntropySearch,
)
from botorch.utils.prior import UserPrior
from botorch.sampling.pathwise import draw_matheron_paths
from botorch.utils.sampling import optimize_posterior_samples


def bayesian_optimization(objective, iterations, dim, bounds, num_optima=100):
    user_prior = DefaultPrior(bounds=bounds, parameter_deufalts=Tensor([-2.1415]), confidence=0.1)      
    doe = SobolEngine(dim)
    init_samples = dim + 3
    train_X = unnormalize(doe.draw(init_samples), bounds).double()
    train_y = torch.zeros(len(train_X)).double()
    
    for i in range(len(train_X)):
        train_y[i] = objective(train_X[i, :])
    train_y = train_y.unsqueeze(-1)

    # No need to get the first guess - we want as many guesses as queries anyway
    guesses = train_X
        
    for i in range(iterations):
        norm_X = normalize(train_X, bounds)
        norm_y = standardize(train_y)

        gp = FixedNoiseGP(norm_X, norm_y, train_Yvar=torch.ones_like(norm_y) * 1e-3)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
         
        fit_gpytorch_model(mll, optimizer=fit_gpytorch_torch,
                           options={'disp': False})


        paths = draw_matheron_paths(
            gp, sample_shape=torch.Size([num_optima]))

        
        optimal_inputs, optimal_outputs = optimize_posterior_samples(
            paths, bounds=Tensor(bounds).T.to(gp.train_targets), candidates=gp.train_inputs[0])
        probs = torch.exp(user_prior.forward(optimal_inputs))
        probsum = torch.sum(probs)
        choices = np.random.choice(len(probs), size=int(num_optima), p=(probs / probsum).detach().numpy(), replace=True)
        
        optimal_inputs, optimal_outputs = optimal_inputs.squeeze(-2), optimal_outputs.squeeze(-2)
        optimal_inputs, optimal_outputs = optimal_inputs.to(gp.train_targets), optimal_outputs.to(gp.train_targets)
        
        plot_paths(paths, gp, user_prior, choices)


        acq_function = qLowerBoundJointEntropySearch(
            model=gp,
            optimal_inputs=optimal_inputs,
            optimal_outputs=optimal_outputs,
        )
        
        norm_point, val = optimize_acqf(
            acq_function=acq_function,
            bounds=torch.Tensor([[0, 1] for d in range(dim)]).T,
            q=1,
            num_restarts=20,
            raw_samples=2048,
            options={'nonnegative': False, 'sample_around_best': True, 'maxiter': 50},
        )
        
        new_point = unnormalize(norm_point, bounds)
        new_eval = torch.Tensor([objective(new_point)]).reshape(1, 1)
        print(new_point, '--------', new_eval)
        train_X = torch.cat([train_X, new_point])
        train_y = torch.cat([train_y, new_eval])
    # TODO - fix so that some of the raw samples come from X_opt - these points are obviously the promising ones

def plot_paths(sample_paths, model: SingleTaskGP, prior: UserPrior, choices: np.ndarray,  num_points: int = 51, filter_factor: int = 1):
    X = torch.linspace(0, 1, num_points).unsqueeze(-1)
    y = sample_paths.forward(X)
    posterior = model.posterior(X, observation_noise=False)
    mean = n(posterior.mean).flatten()
    std = n(torch.sqrt(posterior.variance)).flatten() * 2
    
    num_optima = len(choices) * filter_factor
    #doe = SobolEngine(dimension=1, shuffle=True).draw(num_points * len(sample_paths))
    fig, ax = plt.subplots(2, 3, figsize=(16, 12), sharex=True)
    ax[0, 0].plot(X, mean, c='blue')
    ax[0, 0].fill_between(X.flatten(), mean - std * 1, mean + std * 1, alpha=0.2, color='red')
    ax[0, 0].scatter(model.train_inputs[0], model.train_targets)
    
    ax[1, 0].plot(X, 2 * n(torch.exp(prior.forward(X))).T / n(torch.exp(prior.forward(X))).max(), c='green', label='Prior')
    
    candidates = SobolEngine(dimension=1, scramble=True).draw(num_optima * filter_factor * num_points)
    candidates_per_sample = candidates.reshape(filter_factor * num_optima, 1, num_points, 1) 
    y_sample_org = sample_paths.forward(candidates_per_sample)
    
    y_sample = y_sample_org.squeeze(1).flatten()
    y_sample_prior_org = sample_paths.forward(candidates_per_sample)[choices]
    y_sample_prior = y_sample_prior_org.squeeze(1).flatten()
    candidates_prior = candidates_per_sample[choices].flatten()
    
    ax[0, 1].plot(X, mean, c='blue')
    ax[0, 1].fill_between(X.flatten(), mean - std * 1, mean + std * 1, alpha=0.2, color='red')
    ax[0, 1].scatter(n(candidates), n(y_sample), color='grey', linewidth=1, s=0.1)
    ax[1, 1].scatter(n(candidates_prior), n(y_sample_prior), color='grey', linewidth=1, s=0.1)
    ax[1, 1].plot(X, mean, c='blue')
    ax[1, 1].fill_between(X.flatten(), mean - std * 1, mean + std * 1, alpha=0.2, color='red')
    ax[1, 1].scatter(model.train_inputs[0], model.train_targets)
    
    best = candidates_per_sample[torch.arange(num_optima).to(torch.long), :, torch.argmax(y_sample_org, dim=2).flatten(), :]
    best_prior = candidates_per_sample[choices][torch.arange(num_optima).to(torch.long), :, torch.argmax(y_sample_prior_org, dim=2).flatten(), :]

    ax[0, 2].hist(n(best).flatten(), bins=np.linspace(0, 1, 51))
    ax[1, 2].hist(n(best_prior).flatten(), bins=np.linspace(0, 1, 51))
    
    ax[0, 0].get_shared_y_axes().join(ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1])
    fig.tight_layout()
    plt.show()

def n(X):
    return X.detach().numpy()

def filter_samples():
    pass


if __name__ == '__main__':

    test_objective = Branin(negate=True)

    def oned_objective(x):
        return test_objective(torch.Tensor([x, 2.275]).unsqueeze(0))

    bounds_1d = test_objective.bounds[:, 0].unsqueeze(1)
    dim_1d = 1

    best_X, best_y, best_guess = bayesian_optimization(
        objective=oned_objective, 
        iterations=200, 
        dim=1, 
        bounds=bounds_1d,
        num_optima=4000,
    )
