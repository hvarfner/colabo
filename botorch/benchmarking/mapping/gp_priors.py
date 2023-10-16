import math

from typing import Optional, Dict
import torch
from torch import Tensor

from gpytorch.priors.torch_priors import GammaPrior
from botorch.models import (
    FixedNoiseGP,
    SingleTaskGP
)
from gpytorch.means import (
    ConstantMean
)

from gpytorch.kernels import (
    ScaleKernel,
    MaternKernel,
    RBFKernel
)
from gpytorch.priors import (
    NormalPrior,
    GammaPrior,
    LogNormalPrior
)
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.likelihoods import GaussianLikelihood

from botorch.models.gp_regression import MIN_INFERRED_NOISE_LEVEL
MAP_KWARGS = ['likelihood', 'mean_module', 'covar_module']
PYRO_KWARGS = ['pyro_model', 'disable_progbar', 'warmup', 'num_samples', 'thinning']
MODELS = {
    'FixedNoiseGP': FixedNoiseGP,
    'SingleTaskGP': SingleTaskGP,
}

def parse_hyperparameters(gp_params, dims, prior_location: Optional[Tensor] = None, dim_scaling: bool = False):
    if prior_location is not None:
        poly_loc = convert_prior_to_mean(prior_location)
    else:
        poly_loc = torch.zeros((dims, 2))

    const_params = gp_params.get('const', {})
    ls_params = gp_params.get('ls', {})
    ops_params = gp_params.get('ops', {})
    noise_params = gp_params.get('noise', {})
    if dim_scaling:
        ls_params['loc'] = ls_params['loc'] + math.log(dims)

    
    p_params = gp_params.get('poly', {'magnitude': 0, 'scale': 1})
    poly_params = {'loc': poly_loc * p_params['magnitude'], 'scale': p_params['scale']}

    return const_params, poly_params, ls_params, ops_params, noise_params


def parse_constraints(gp_constraints):
    ls_constraint = gp_constraints.get('ls', 1e-4)
    scale_constraint = gp_constraints.get('scale', 1e-4)
    noise_constraint = gp_constraints.get('noise', 1e-4)

    return ls_constraint, scale_constraint, noise_constraint


def get_covar_module(model_name, dims, prior_location: Tensor = None, gp_params: Dict = None, gp_constraints: Dict = {}):

    const_params, poly_params, ls_params, ops_params, noise_params = parse_hyperparameters(
        gp_params, dims, prior_location=prior_location, dim_scaling=('highdim' in model_name))
    ls_constraint, scale_constraint, noise_constraint = parse_constraints(
        gp_constraints)
    
    COVAR_MODULES = {
        'singletask_default':
        {
            'covar_module': None,
            'likelihood': None,
        },
        'singletask_lognormal':
        {
            'mean_module': ConstantMean(NormalPrior(**const_params)),
            'covar_module': ScaleKernel(
                base_kernel=MaternKernel(
                    ard_num_dims=dims,
                    lengthscale_prior=LogNormalPrior(**ls_params),
                    lengthscale_constraint=GreaterThan(ls_constraint)
                ),
                outputscale_prior=LogNormalPrior(**ops_params),
                outputscale_constraint=GreaterThan(scale_constraint)
            ),
            'likelihood': GaussianLikelihood(LogNormalPrior(**noise_params),
                                             noise_constraint=GreaterThan(
                                                 noise_constraint)
                                             ),
        },
    }
    return COVAR_MODULES[model_name]


def convert_prior_to_mean(prior_location, quad_norm: float = 0.25):
    dim = len(prior_location)
    unnorm_quad = torch.ones((dim, 1))
    unnorm_linear = -prior_location.reshape(-1, 1) * 2
    mean_prior = -torch.cat((unnorm_linear, unnorm_quad), dim=1)
    return mean_prior
