# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Tuple

import pyro
import torch
from ax.models.torch.botorch_defaults import _get_model, MIN_OBSERVED_NOISE_LEVEL
from botorch.models.fully_bayesian import MIN_INFERRED_NOISE_LEVEL
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ZeroMean
from torch import Tensor
from torch.distributions import MultivariateNormal


def _get_rbf_kernel(num_samples: int, dim: int) -> ScaleKernel:
    return ScaleKernel(
        base_kernel=RBFKernel(ard_num_dims=dim, batch_shape=torch.Size([num_samples])),
        batch_shape=torch.Size([num_samples]),
    )


def _get_rbf_noscale_kernel(num_samples: int, dim: int) -> RBFKernel:
    return RBFKernel(ard_num_dims=dim, batch_shape=torch.Size([num_samples]))


def _get_single_task_gpytorch_model(
    Xs: List[Tensor],
    Ys: List[Tensor],
    Yvars: List[Tensor],
    task_features: List[int],
    fidelity_features: List[int],
    state_dict: Optional[Dict[str, Tensor]] = None,
    num_samples: int = 512,
    thinning: int = 16,
    use_input_warping: bool = False,
    gp_kernel: str = "matern",
    **kwargs: Any,
) -> ModelListGP:
    r"""Instantiates a batched GPyTorchModel(ModelListGP) based on the given data.
    The model fitting is based on MCMC and is run separately using pyro. The MCMC
    samples will be loaded into the model instantiated here afterwards.

    Returns:
        A ModelListGP.
    """
    if len(task_features) > 0:
        raise NotImplementedError("Currently do not support MT-GP models with MCMC!")
    if len(fidelity_features) > 0:
        raise NotImplementedError(
            "Fidelity MF-GP models are not currently supported with MCMC!"
        )
    num_mcmc_samples = num_samples // thinning
    covar_modules = [
        _get_rbf_kernel(num_samples=num_mcmc_samples, dim=Xs[0].shape[-1])
        if gp_kernel == "rbf"
        else None
        for _ in range(len(Xs))
    ]

    models = [
        _get_model(
            X=X.unsqueeze(0).expand(num_mcmc_samples, X.shape[0], -1),
            Y=Y.unsqueeze(0).expand(num_mcmc_samples, Y.shape[0], -1),
            Yvar=Yvar.unsqueeze(0).expand(num_mcmc_samples, Yvar.shape[0], -1),
            fidelity_features=fidelity_features,
            use_input_warping=use_input_warping,
            covar_module=covar_module,
            **kwargs,
        )
        for X, Y, Yvar, covar_module in zip(Xs, Ys, Yvars, covar_modules)
    ]
    model = ModelListGP(*models)
    model.to(Xs[0])
    return model


def _get_active_learning_gpytorch_model(
    Xs: List[Tensor],
    Ys: List[Tensor],
    Yvars: List[Tensor],
    task_features: List[int],
    fidelity_features: List[int],
    state_dict: Optional[Dict[str, Tensor]] = None,
    num_samples: int = 512,
    thinning: int = 16,
    use_input_warping: bool = False,
    gp_kernel: str = "rbf",
    **kwargs: Any,
) -> ModelListGP:
    r"""Instantiates a batched GPyTorchModel(ModelListGP) based on the given data.
    The model fitting is based on MCMC and is run separately using pyro. The MCMC
    samples will be loaded into the model instantiated here afterwards.

    Returns:
        A ModelListGP.
    """
    if len(task_features) > 0:
        raise NotImplementedError("Currently do not support MT-GP models with MCMC!")
    if len(fidelity_features) > 0:
        raise NotImplementedError(
            "Fidelity MF-GP models are not currently supported with MCMC!"
        )
    num_mcmc_samples = num_samples // thinning
    covar_modules = [
        _get_rbf_noscale_kernel(num_samples=num_mcmc_samples, dim=Xs[0].shape[-1])
        if gp_kernel == "rbf"
        else None
        for _ in range(len(Xs))
    ]

    models = [
        _get_model(
            X=X.unsqueeze(0).expand(num_mcmc_samples, X.shape[0], -1),
            Y=Y.unsqueeze(0).expand(num_mcmc_samples, Y.shape[0], -1),
            Yvar=Yvar.unsqueeze(0).expand(num_mcmc_samples, Yvar.shape[0], -1),
            fidelity_features=fidelity_features,
            use_input_warping=use_input_warping,
            covar_module=covar_module,
            mean_module=ZeroMean(),
            **kwargs,
        )
        for X, Y, Yvar, covar_module in zip(Xs, Ys, Yvars, covar_modules)
    ]
    for model in models:
        # to make model.subset_output() work
        model._subset_batch_dict = {
            "likelihood.noise_covar.raw_noise": -2,
            "covar_module.raw_lengthscale": -3,
        }
    model = ModelListGP(*models)
    model.to(Xs[0])
    return model


def _get_square_root_gpytorch_model(
    Xs: List[Tensor],
    Ys: List[Tensor],
    Yvars: List[Tensor],
    task_features: List[int],
    fidelity_features: List[int],
    state_dict: Optional[Dict[str, Tensor]] = None,
    num_samples: int = 512,
    thinning: int = 16,
    use_input_warping: bool = False,
    gp_kernel: str = "matern",
    **kwargs: Any,
) -> ModelListGP:
    r"""Instantiates a batched GPyTorchModel(ModelListGP) based on the given data.
    The model fitting is based on MCMC and is run separately using pyro. The MCMC
    samples will be loaded into the model instantiated here afterwards.

    Returns:
        A ModelListGP.
    """
    if len(task_features) > 0:
        raise NotImplementedError("Currently do not support MT-GP models with MCMC!")
    if len(fidelity_features) > 0:
        raise NotImplementedError(
            "Fidelity MF-GP models are not currently supported with MCMC!"
        )
    num_mcmc_samples = num_samples // thinning
    covar_modules = [
        _get_rbf_kernel(num_samples=num_mcmc_samples, dim=Xs[0].shape[-1])
        if gp_kernel == "rbf"
        else None
        for _ in range(len(Xs))
    ]

    models = [
        _get_square_root_model(
            X=X.unsqueeze(0).expand(num_mcmc_samples, X.shape[0], -1),
            Y=Y.unsqueeze(0).expand(num_mcmc_samples, Y.shape[0], -1),
            Yvar=Yvar.unsqueeze(0).expand(num_mcmc_samples, Yvar.shape[0], -1),
            fidelity_features=fidelity_features,
            use_input_warping=use_input_warping,
            covar_module=covar_module,
            **kwargs,
        )
        for X, Y, Yvar, covar_module in zip(Xs, Ys, Yvars, covar_modules)
    ]
    model = ModelListGP(*models)
    model.to(Xs[0])
    return model


def _get_active_learning_square_root_gpytorch_model(
    Xs: List[Tensor],
    Ys: List[Tensor],
    Yvars: List[Tensor],
    task_features: List[int],
    fidelity_features: List[int],
    state_dict: Optional[Dict[str, Tensor]] = None,
    num_samples: int = 512,
    thinning: int = 16,
    use_input_warping: bool = False,
    gp_kernel: str = "rbf",
    **kwargs: Any,
) -> ModelListGP:
    r"""Instantiates a batched GPyTorchModel(ModelListGP) based on the given data.
    The model fitting is based on MCMC and is run separately using pyro. The MCMC
    samples will be loaded into the model instantiated here afterwards.

    Returns:
        A ModelListGP.
    """
    if len(task_features) > 0:
        raise NotImplementedError("Currently do not support MT-GP models with MCMC!")
    if len(fidelity_features) > 0:
        raise NotImplementedError(
            "Fidelity MF-GP models are not currently supported with MCMC!"
        )
    num_mcmc_samples = num_samples // thinning
    covar_modules = [
        _get_rbf_noscale_kernel(num_samples=num_mcmc_samples, dim=Xs[0].shape[-1])
        if gp_kernel == "rbf"
        else None
        for _ in range(len(Xs))
    ]

    models = [
        _get_square_root_model(
            X=X.unsqueeze(0).expand(num_mcmc_samples, X.shape[0], -1),
            Y=Y.unsqueeze(0).expand(num_mcmc_samples, Y.shape[0], -1),
            Yvar=Yvar.unsqueeze(0).expand(num_mcmc_samples, Yvar.shape[0], -1),
            fidelity_features=fidelity_features,
            use_input_warping=use_input_warping,
            covar_module=covar_module,
            **kwargs,
        )
        for X, Y, Yvar, covar_module in zip(Xs, Ys, Yvars, covar_modules)
    ]
    model = ModelListGP(*models)
    model.to(Xs[0])
    return model

def _get_square_root_model(
    X: Tensor,
    Y: Tensor,
    Yvar: Tensor,
    task_feature: Optional[int] = None,
    fidelity_features: Optional[List[int]] = None,
    use_input_warping: bool = False,
    **kwargs: Any,
) -> GPyTorchModel:
    """Instantiate a model of type depending on the input data.

    Args:
        X: A `n x d` tensor of input features.
        Y: A `n x m` tensor of input observations.
        Yvar: A `n x m` tensor of input variances (NaN if unobserved).
        task_feature: The index of the column pertaining to the task feature
            (if present).
        fidelity_features: List of columns of X that are fidelity parameters.

    Returns:
        A GPyTorchModel (unfitted).
    """
    Yvar = Yvar.clamp_min(MIN_OBSERVED_NOISE_LEVEL)
    is_nan = torch.isnan(Yvar)
    any_nan_Yvar = torch.any(is_nan)
    all_nan_Yvar = torch.all(is_nan)
    if any_nan_Yvar and not all_nan_Yvar:
        if task_feature:
            # TODO (jej): Replace with inferred noise before making perf judgements.
            Yvar[Yvar != Yvar] = MIN_OBSERVED_NOISE_LEVEL
        else:
            raise ValueError(
                "Mix of known and unknown variances indicates valuation function "
                "errors. Variances should all be specified, or none should be."
            )
    if use_input_warping:
        warp_tf = get_warping_transform(
            d=X.shape[-1],
            task_feature=task_feature,
            batch_shape=X.shape[:-2],
        )
    else:
        warp_tf = None
    if len(fidelity_features) > 0:
        raise ValueError('SquareRootGP is not available with multi-fidelity.')
    elif task_feature is None and all_nan_Yvar:
        gp = SquareRootSingleTaskGP(
            train_X=X, train_Y=Y, input_transform=warp_tf, **kwargs)
    elif task_feature is None:
        raise ValueError('SquareRootGP is not available with fixed noise.')
    return gp


def pyro_sample_outputscale(
    concentration: float = 2.0,
    rate: float = 0.15,
    **tkwargs: Any,
) -> Tensor:

    return pyro.sample(
        "outputscale",
        # pyre-fixme[16]: Module `distributions` has no attribute `Gamma`
        pyro.distributions.Gamma(
            torch.tensor(concentration, **tkwargs),
            torch.tensor(rate, **tkwargs),
        ),
    )


def pyro_sample_mean(**tkwargs: Any) -> Tensor:

    return pyro.sample(
        "mean",
        # pyre-fixme[16]: Module `distributions` has no attribute `Normal`.
        pyro.distributions.Normal(
            torch.tensor(0.0, **tkwargs),
            torch.tensor(1.0, **tkwargs),
        ),
    )


def pyro_sample_noise(**tkwargs: Any) -> Tensor:

    # this prefers small noise but has heavy tails
    return pyro.sample(
        "noise",
        # pyre-fixme[16]: Module `distributions` has no attribute `Gamma`.
        pyro.distributions.Gamma(
            torch.tensor(0.9, **tkwargs),
            torch.tensor(10.0, **tkwargs),
        ),
    )


def pyro_sample_saas_lengthscales(
    dim: int,
    alpha: float = 0.1,
    **tkwargs: Any,
) -> Tensor:

    tausq = pyro.sample(
        "kernel_tausq",
        # pyre-fixme[16]: Module `distributions` has no attribute `HalfCauchy`.
        pyro.distributions.HalfCauchy(torch.tensor(alpha, **tkwargs)),
    )
    inv_length_sq = pyro.sample(
        "_kernel_inv_length_sq",
        # pyre-fixme[16]: Module `distributions` has no attribute `HalfCauchy`.
        pyro.distributions.HalfCauchy(torch.ones(dim, **tkwargs)),
    )
    inv_length_sq = pyro.deterministic("kernel_inv_length_sq", tausq * inv_length_sq)
    lengthscale = pyro.deterministic(
        "lengthscale",
        (1.0 / inv_length_sq).sqrt(),  # pyre-ignore [16]
    )
    return lengthscale


def pyro_sample_input_warping(
    dim: int,
    **tkwargs: Any,
) -> Tuple[Tensor, Tensor]:

    c0 = pyro.sample(
        "c0",
        # pyre-fixme[16]: Module `distributions` has no attribute `LogNormal`.
        pyro.distributions.LogNormal(
            torch.tensor([0.0] * dim, **tkwargs),
            torch.tensor([0.75**0.5] * dim, **tkwargs),
        ),
    )
    c1 = pyro.sample(
        "c1",
        # pyre-fixme[16]: Module `distributions` has no attribute `LogNormal`.
        pyro.distributions.LogNormal(
            torch.tensor([0.0] * dim, **tkwargs),
            torch.tensor([0.75**0.5] * dim, **tkwargs),
        ),
    )
    return c0, c1


def postprocess_saas_samples(samples: Dict[str, Tensor]) -> Dict[str, Tensor]:
    inv_length_sq = (
        samples["kernel_tausq"].unsqueeze(-1) * samples["_kernel_inv_length_sq"]
    )
    samples["lengthscale"] = (1.0 / inv_length_sq).sqrt()  # pyre-ignore [16]
    del samples["kernel_tausq"], samples["_kernel_inv_length_sq"]
    # this prints the summary

    return samples


def postprocess_squareroot_gp_samples(samples: Dict[str, Tensor]) -> Dict[str, Tensor]:
    return samples


def postprocess_bayesian_al_samples(samples: Dict[str, Tensor]) -> Dict[str, Tensor]:
    return samples


# pyre-fixme[24]: Generic type `dict` expects 2 type parameters, use `typing.Dict`
#  to avoid runtime subscripting errors.
def load_mcmc_samples_to_model(model: GPyTorchModel, mcmc_samples: Dict) -> None:
    """Load MCMC samples into GPyTorchModel."""
    if "delta_eta" in mcmc_samples:
        Y_max = torch.max(model.train_targets)
        delta_eta_onedim = mcmc_samples["delta_eta"].detach().clone()
        delta_eta = delta_eta_onedim.unsqueeze(-1).unsqueeze(-1)
        model.set_eta(delta_eta + Y_max)
    if "noise" in mcmc_samples:
        model.likelihood.noise_covar.noise = (
            mcmc_samples["noise"]
            .detach()
            .clone()
            .view(model.likelihood.noise_covar.noise.shape)
            .clamp_min(MIN_INFERRED_NOISE_LEVEL)
        )
    if hasattr(model.covar_module, 'base_kernel'):
        model.covar_module.base_kernel.lengthscale = (
            mcmc_samples["lengthscale"]
            .detach()
            .clone()
            .view(model.covar_module.base_kernel.lengthscale.shape)  # pyre-ignore
        )
    else:
        model.covar_module.lengthscale = (
            mcmc_samples["lengthscale"]
            .detach()
            .clone()
            .view(model.covar_module.lengthscale.shape)  # pyre-ignore
        )
    if "outputscale" in mcmc_samples:
        model.covar_module.outputscale = (  # pyre-ignore
            mcmc_samples["outputscale"]
            .detach()
            .clone()
            # pyre-fixme[16]: Item `Tensor` of `Union[Tensor, Module]` has no attribute
            #  `outputscale`.
            .view(model.covar_module.outputscale.shape)
        )
    if "mean" in mcmc_samples:
        model.mean_module.constant.data = (
            mcmc_samples["mean"]
            .detach()
            .clone()
            .view(model.mean_module.constant.shape)  # pyre-ignore
        )
    if "c0" in mcmc_samples:
        model.input_transform._set_concentration(  # pyre-ignore
            i=0,
            value=mcmc_samples["c0"]
            .detach()
            .clone()
            .view(model.input_transform.concentration0.shape),  # pyre-ignore
        )
        # pyre-fixme[16]: Item `Tensor` of `Union[Tensor, Module]` has no attribute
        #  `_set_concentration`.
        model.input_transform._set_concentration(
            i=1,
            value=mcmc_samples["c1"]
            .detach()
            .clone()
            .view(model.input_transform.concentration1.shape),  # pyre-ignore
        )


def pyro_sample_delta_eta(mu: float = -1, var: float = 0.25, **tkwargs: Any) -> Tensor:
    return pyro.sample(
        "delta_eta",
        # pyre-fixme[16]: Module `distributions` has no attribute `Gamma`.
        pyro.distributions.LogNormal(
            torch.tensor(mu, **tkwargs),
            torch.tensor(var ** 0.5, **tkwargs),
        ),
    )


def pyro_sample_al_noise(mu: float = 0, var: float = 3.0, **tkwargs: Any) -> Tensor:
    return pyro.sample(
        "noise",
        # pyre-fixme[16]: Module `distributions` has no attribute `Gamma`.
        pyro.distributions.LogNormal(
            torch.tensor(mu, **tkwargs),
            torch.tensor(var ** 0.5, **tkwargs),
        ),
    )


def pyro_sample_al_lengthscales(dim, mu: float = 0, var: float = 3.0, **tkwargs: Any) -> Tensor:
    return pyro.sample(
        "lengthscale",
        # pyre-fixme[16]: Module `distributions` has no attribute `Gamma`.
        pyro.distributions.LogNormal(
            torch.ones(dim, **tkwargs) * mu,
            torch.ones(dim, **tkwargs) * var ** 0.5
        ),
    )


def pyro_sample_bo_lengthscales(dim, **tkwargs: Any) -> Tensor:
    return pyro.sample(
        "lengthscale",
        # pyre-fixme[16]: Module `distributions` has no attribute `Gamma`.
        pyro.distributions.Gamma(
            torch.tensor([3.0] * dim, **tkwargs),
            torch.tensor([6.0] * dim, **tkwargs),
        ),
    )


def pyro_sample_bo_noise(**tkwargs: Any) -> Tensor:
    return pyro.sample(
        "noise",
        # pyre-fixme[16]: Module `distributions` has no attribute `Gamma`.
        pyro.distributions.Gamma(
            torch.tensor(1.1, **tkwargs),
            torch.tensor(0.05, **tkwargs),
        ),
    )


def slice_bo_prior(dim):
    outputscale_mean = torch.Tensor([0])
    noise_mean = torch.Tensor([0])
    lengthscale_mean = torch.zeros(dim)
    outputscale_variance = torch.Tensor([3])
    noise_variance = torch.Tensor([3])
    lengthscale_variance = torch.ones_like(lengthscale_mean) * 3

    means = torch.cat((outputscale_mean, noise_mean, lengthscale_mean), dim=0)
    variances = torch.cat((outputscale_variance, noise_variance, lengthscale_variance), dim=0)
    dist = MultivariateNormal(means, torch.diag(variances))
    return dist


def slice_al_prior(dim):
    noise_mean = torch.Tensor([0])
    lengthscale_mean = torch.zeros(dim)
    noise_variance = torch.Tensor([3])
    lengthscale_variance = torch.ones_like(lengthscale_mean) * 3

    means = torch.cat((noise_mean, lengthscale_mean), dim=0)
    variances = torch.cat((noise_variance, lengthscale_variance), dim=0)
    dist = MultivariateNormal(means, torch.diag(variances))
    return dist


def slice_warping_prior(dim):
    mu_mean = torch.Tensor([0])
    outputscale_mean = torch.Tensor([0])
    noise_mean = torch.Tensor([0])
    lengthscale_mean = torch.zeros(dim)
    alpha_mean = torch.zeros(dim)
    beta_mean = torch.zeros(dim)
    mu_variance = torch.Tensor([1])
    outputscale_variance = torch.Tensor([3])
    noise_variance = torch.Tensor([3])
    lengthscale_variance = torch.ones_like(lengthscale_mean) * 3
    alpha_variance = torch.ones_like(lengthscale_mean) * 1
    beta_variance = torch.ones_like(lengthscale_mean) * 1

    means = torch.cat(
        (
            mu_mean,
            outputscale_mean,
            noise_mean,
            lengthscale_mean,
            alpha_mean,
            beta_mean,
        ), 
        dim=0
    )
    variances = torch.cat(
        (
            mu_variance,
            outputscale_variance,
            noise_variance,
            lengthscale_variance,
            alpha_variance,
            beta_variance,
        ), 
        dim=0
    )
    dist = MultivariateNormal(means, torch.diag(variances))
    return dist


def slice_sqrt_prior(dim):
    eta_mean = torch.Tensor([-1])
    outputscale_mean = torch.Tensor([0])
    noise_mean = torch.Tensor([0])
    lengthscale_mean = torch.zeros(dim)
    eta_variance = torch.Tensor([0.25])
    outputscale_variance = torch.Tensor([3])
    noise_variance = torch.Tensor([3])
    lengthscale_variance = torch.ones_like(lengthscale_mean) * 3

    means = torch.cat((eta_mean, outputscale_mean, noise_mean, lengthscale_mean), dim=0)
    variances = torch.cat((eta_variance, outputscale_variance, noise_variance, lengthscale_variance), dim=0)
    dist = MultivariateNormal(means, torch.diag(variances))
    return dist


def slice_noscale_sqrt_prior(dim):
    eta_mean = torch.Tensor([-1])
    noise_mean = torch.Tensor([0])
    lengthscale_mean = torch.zeros(dim)
    eta_variance = torch.Tensor([0.25])
    noise_variance = torch.Tensor([3])
    lengthscale_variance = torch.ones_like(lengthscale_mean) * 3

    means = torch.cat((eta_mean, noise_mean, lengthscale_mean), dim=0)
    variances = torch.cat((eta_variance, noise_variance, lengthscale_variance), dim=0)
    dist = MultivariateNormal(means, torch.diag(variances))
    return dist


def postprocess_noscale_sqrt_slice(hp_tensor):
    samples = {}
    samples['delta_eta'] = hp_tensor[:, 0]
    samples['noise'] = hp_tensor[:, 1]
    samples['lengthscale'] = hp_tensor[:, 2:]
    return samples

def postprocess_warping(hp_tensor):
    samples = {}
    dim  = int((hp_tensor.shape[1] - 3) / 3)
    samples['mean'] = hp_tensor[:, 0]
    samples['outputscale'] = hp_tensor[:, 1]
    samples['noise'] = hp_tensor[:, 2]
    samples['lengthscale'] = hp_tensor[:, 3:dim + 3]
    samples['alpha'] = hp_tensor[:, dim + 3:2 * dim + 3]
    samples['beta'] = hp_tensor[:, 2 * dim + 3:3 * dim + 3]
    return samples


def postprocess_bo_slice(hp_tensor):
    samples = {}
    samples['outputscale'] = hp_tensor[:, 0]
    samples['noise'] = hp_tensor[:, 1]
    samples['lengthscale'] = hp_tensor[:, 2:]
    return samples

    

def postprocess_al_slice(hp_tensor):
    samples = {}
    samples['noise'] = hp_tensor[:, 0]
    samples['lengthscale'] = hp_tensor[:, 1:]
    return samples

    

PRIOR_REGISTRY = {
    'SAAS': {
        'parameter_priors':
        {
            'outputscale_func': pyro_sample_outputscale,
            'mean_func': pyro_sample_mean,
            'noise_func': pyro_sample_noise,
            'lengthscale_func': pyro_sample_saas_lengthscales,
            'input_warping_func': pyro_sample_input_warping,
        },
        'postprocessing': postprocess_saas_samples
    },
    'BAL': {
        'parameter_priors':
        {
            'outputscale_func': None,
            'mean_func': None,
            'noise_func': pyro_sample_al_noise,
            'lengthscale_func': pyro_sample_al_lengthscales,
            'input_warping_func': None,
        },
        'postprocessing': postprocess_bayesian_al_samples
    },
    'BO': {
        'parameter_priors':
        {
            'outputscale_func': pyro_sample_outputscale,
            'mean_func': pyro_sample_mean,
            'noise_func': pyro_sample_al_noise,
            'lengthscale_func': pyro_sample_al_lengthscales,
            'input_warping_func': None,
        },
        'postprocessing': postprocess_bayesian_al_samples
    },
    'SCoreBO': {
        'parameter_priors':
        {
            'outputscale_func': pyro_sample_outputscale,
            'mean_func': None,
            'noise_func': pyro_sample_al_noise,
            'lengthscale_func': pyro_sample_al_lengthscales,
            'eta_func': pyro_sample_delta_eta,
            'input_warping_func': None,
        },
        'postprocessing': postprocess_squareroot_gp_samples
    },
    'SCoreBO_slice': {
        'parameter_priors':
        {
            'joint': slice_sqrt_prior, 
        },
        'postprocessing': postprocess_bo_slice
    },
    'BO_slice_warp':
    {
        'parameter_priors':
        {
            'joint': slice_warping_prior, 
        },
        'postprocessing': postprocess_warping
    },
    'SCoreBO_AL_slice': {
        'parameter_priors':
        {
            'joint': slice_noscale_sqrt_prior, 
        },
        'postprocessing': postprocess_noscale_sqrt_slice
    },
    'BO_slice': {
        'parameter_priors':
        {
            'joint': slice_bo_prior, 
        },
        'postprocessing': postprocess_bo_slice
    },
    'AL_slice': {
        'parameter_priors':
        {
            'joint': slice_al_prior, 
        },
        'postprocessing': postprocess_al_slice
    },
}
