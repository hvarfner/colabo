# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


r"""Gaussian Process Regression models with fully Bayesian inference.

Fully Bayesian models use Bayesian inference over model hyperparameters, such
as lengthscales and noise variance, learning a posterior distribution for the
hyperparameters using the No-U-Turn-Sampler (NUTS). This is followed by
sampling a small set of hyperparameters (often ~16) from the posterior
that we will use for model predictions and for computing acquisition function
values. By contrast, our “standard” models (e.g.
`SingleTaskGP`) learn only a single best value for each hyperparameter using
MAP. The fully Bayesian method generally results in a better and more
well-calibrated model, but is more computationally intensive. For a full
description, see [Eriksson2021saasbo].

We use a lightweight PyTorch implementation of a Matern-5/2 kernel as there are
some performance issues with running NUTS on top of standard GPyTorch models.
The resulting hyperparameter samples are loaded into a batched GPyTorch model
after fitting.

References:

.. [Eriksson2021saasbo]
    D. Eriksson, M. Jankowiak. High-Dimensional Bayesian Optimization
    with Sparse Axis-Aligned Subspaces. Proceedings of the Thirty-
    Seventh Conference on Uncertainty in Artificial Intelligence, 2021.
"""


import math
from abc import abstractmethod
from typing import Any, Dict, List, Mapping, Optional, Tuple, Callable, Union

import pyro
import numpy as np
import torch
from torch.distributions import Kumaraswamy
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.models.utils import validate_input_scaling
from botorch.sampling import MCSampler
from botorch.posteriors.fully_bayesian import FullyBayesianPosterior, MCMC_DIM
from botorch import settings
from botorch.models.utils import fantasize as fantasize_flag, validate_input_scaling
from gpytorch.constraints import GreaterThan
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel
from gpytorch.kernels.kernel import dist, Kernel
from gpytorch.likelihoods.gaussian_likelihood import (
    FixedNoiseGaussianLikelihood,
    GaussianLikelihood,
)
import gpytorch
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.means.constant_mean import ConstantMean
from gpytorch.means.mean import Mean
from gpytorch.means import ZeroMean
from gpytorch.models.exact_gp import ExactGP
from pyro.ops.integrator import register_exception_handler
from torch import Tensor


MIN_INFERRED_NOISE_LEVEL = 1e-6
FIXED_MIN_NOISE = 1e-4
_sqrt5 = math.sqrt(5)


def _handle_torch_linalg(exception: Exception) -> bool:
    return type(exception) == torch.linalg.LinAlgError


def _handle_valerr_in_dist_init(exception: Exception) -> bool:
    if not type(exception) == ValueError:
        return False
    return "satisfy the constraint PositiveDefinite()" in str(exception)


register_exception_handler("torch_linalg", _handle_torch_linalg)
register_exception_handler("valerr_in_dist_init", _handle_valerr_in_dist_init)


def compute_mean(X: Tensor, const: int, poly: Tensor) -> Tensor:
    degrees = torch.arange(1, poly.shape[-1] + 1)
    x_poly = torch.pow(X.unsqueeze(-1), degrees)
    return const + torch.sum(x_poly * poly, dim=[-1, -2])


# TODO exponential kernel
def sqexp_kernel(X: Tensor, lengthscale: Tensor) -> Tensor:
    """Squared exponential kernel."""

    dist = compute_dists(X=X, lengthscale=lengthscale)
    exp_component = torch.exp(-torch.pow(dist, 2) / 2)
    return exp_component


def matern52_kernel(X: Tensor, lengthscale: Tensor) -> Tensor:
    """Matern-5/2 kernel."""
    dist = compute_dists(X=X, lengthscale=lengthscale)
    sqrt5_dist = _sqrt5 * dist
    return sqrt5_dist.add(1 + 5 / 3 * (dist**2)) * torch.exp(-sqrt5_dist)



def compute_dists(X: Tensor, lengthscale: Tensor) -> Tensor:
    """Compute kernel distances."""
    scaled_X = X / lengthscale
    return dist(scaled_X, scaled_X, x1_eq_x2=True)


def reshape_and_detach(target: Tensor, new_value: Tensor) -> None:
    """Detach and reshape `new_value` to match `target`."""
    return new_value.detach().clone().view(target.shape).to(target)


class PyroModel:
    r"""
    Base class for a Pyro model; used to assist in learning hyperparameters.

    This class and its subclasses are not a standard BoTorch models; instead
    the subclasses are used as inputs to a `SaasFullyBayesianSingleTaskGP`,
    which should then have its hyperparameters fit with
    `fit_fully_bayesian_model_nuts`. (By default, its subclass `SaasPyroModel`
    is used).  A `PyroModel`’s `sample` method should specify lightweight
    PyTorch functionality, which will be used for fast model fitting with NUTS.
    The utility of `PyroModel` is in enabling fast fitting with NUTS, since we
    would otherwise need to use GPyTorch, which is computationally infeasible
    in combination with Pyro.

    :meta private:
    """

    def set_warm_start_state(self, warm_start_state):
        self.warm_start_state = warm_start_state

    def set_inputs(
        self, train_X: Tensor, train_Y: Tensor, train_Yvar: Optional[Tensor] = None
    ):
        """Set the training data.

        Args:
            train_X: Training inputs (n x d)
            train_Y: Training targets (n x 1)
            train_Yvar: Observed noise variance (n x 1). Inferred if None.
        """
        self.custom_fit = False
        self.train_X = train_X
        self.train_Y = train_Y
        self.train_Yvar = train_Yvar

    @abstractmethod
    def sample(self) -> None:
        r"""Sample from the model."""
        pass  # pragma: no cover

    @abstractmethod
    def postprocess_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor], **kwargs: Any
    ) -> Dict[str, Tensor]:
        """Post-process the final MCMC samples."""
        pass  # pragma: no cover

    @abstractmethod
    def load_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor]
    ) -> Tuple[Mean, Kernel, Likelihood]:
        pass  # pragma: no cover


class SaasPyroModel(PyroModel):
    r"""Implementation of the sparse axis-aligned subspace priors (SAAS) model.

    The SAAS model uses sparsity-inducing priors to identify the most important
    parameters. This model is suitable for high-dimensional BO with potentially
    hundreds of tunable parameters. See [Eriksson2021saasbo]_ for more details.

    `SaasPyroModel` is not a standard BoTorch model; instead, it is used as
    an input to `SaasFullyBayesianSingleTaskGP`. It is used as a default keyword
    argument, and end users are not likely to need to instantiate or modify a
    `SaasPyroModel` unless they want to customize its attributes (such as
    `covar_module`).
    """

    def set_inputs(
        self, train_X: Tensor, train_Y: Tensor, train_Yvar: Optional[Tensor] = None
    ):
        super().set_inputs(train_X, train_Y, train_Yvar)
        self.ard_num_dims = self.train_X.shape[-1]

    def sample(self) -> None:
        r"""Sample from the SAAS model.

        This samples the mean, noise variance, outputscale, and lengthscales according
        to the SAAS prior.
        """
        tkwargs = {"dtype": self.train_X.dtype, "device": self.train_X.device}
        outputscale = self.sample_outputscale(concentration=2.0, rate=0.15, **tkwargs)
        mean = self.sample_mean(**tkwargs)
        noise = self.sample_noise(**tkwargs)
        lengthscale = self.sample_lengthscale(dim=self.ard_num_dims, **tkwargs)
        K = matern52_kernel(X=self.train_X, lengthscale=lengthscale)
        K = outputscale * K + noise * torch.eye(self.train_X.shape[-2], **tkwargs)
        pyro.sample(
            "Y",
            pyro.distributions.MultivariateNormal(
                loc=mean.view(-1).expand(self.train_X.shape[-2]),
                covariance_matrix=K,
            ),
            obs=self.train_Y.squeeze(-1),
        )

    def sample_outputscale(
        self, concentration: float = 2.0, rate: float = 0.15, **tkwargs: Any
    ) -> Tensor:
        r"""Sample the outputscale."""
        return pyro.sample(
            "outputscale",
            pyro.distributions.Gamma(
                torch.tensor(concentration, **tkwargs),
                torch.tensor(rate, **tkwargs),
            ),
        )

    def sample_mean(self, **tkwargs: Any) -> Tensor:
        r"""Sample the mean constant."""
        return pyro.sample(
            "mean",
            pyro.distributions.Normal(
                torch.tensor(0.0, **tkwargs),
                torch.tensor(1.0, **tkwargs),
            ),
        )

    def sample_noise(self, **tkwargs: Any) -> Tensor:
        r"""Sample the noise variance."""
        if self.train_Yvar is None:
            return MIN_INFERRED_NOISE_LEVEL + pyro.sample(
                "noise",
                pyro.distributions.Gamma(
                    torch.tensor(0.9, **tkwargs),
                    torch.tensor(10.0, **tkwargs),
                ),
            )
        else:
            return self.train_Yvar

    def sample_lengthscale(
        self, dim: int, alpha: float = 0.1, **tkwargs: Any
    ) -> Tensor:
        r"""Sample the lengthscale."""
        tausq = pyro.sample(
            "kernel_tausq",
            pyro.distributions.HalfCauchy(torch.tensor(alpha, **tkwargs)),
        )
        inv_length_sq = pyro.sample(
            "_kernel_inv_length_sq",
            pyro.distributions.HalfCauchy(torch.ones(dim, **tkwargs)),
        )
        inv_length_sq = pyro.deterministic(
            "kernel_inv_length_sq", tausq * inv_length_sq
        )
        lengthscale = pyro.deterministic(
            "lengthscale",
            inv_length_sq.rsqrt(),
        )
        return lengthscale

    def postprocess_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        r"""Post-process the MCMC samples.

        This computes the true lengthscales and removes the inverse lengthscales and
        tausq (global shrinkage).
        """
        inv_length_sq = (
            mcmc_samples["kernel_tausq"].unsqueeze(-1)
            * mcmc_samples["_kernel_inv_length_sq"]
        )
        mcmc_samples["lengthscale"] = inv_length_sq.rsqrt()
        # Delete `kernel_tausq` and `_kernel_inv_length_sq` since they aren't loaded
        # into the final model.
        del mcmc_samples["kernel_tausq"], mcmc_samples["_kernel_inv_length_sq"]
        return mcmc_samples

    def load_mcmc_samples(
        self, mcmc_samples: Dict[str, Tensor]
    ) -> Tuple[Mean, Kernel, Likelihood]:
        r"""Load the MCMC samples into the mean_module, covar_module, and likelihood."""
        tkwargs = {"device": self.train_X.device, "dtype": self.train_X.dtype}
        num_mcmc_samples = len(mcmc_samples["mean"])
        batch_shape = torch.Size([num_mcmc_samples])

        mean_module = ConstantMean(batch_shape=batch_shape).to(**tkwargs)
        covar_module = ScaleKernel(
            base_kernel=MaternKernel(
                ard_num_dims=self.ard_num_dims,
                batch_shape=batch_shape,
            ),
            batch_shape=batch_shape,
        ).to(**tkwargs)
        if self.train_Yvar is not None:
            likelihood = FixedNoiseGaussianLikelihood(
                # Reshape to shape `num_mcmc_samples x N`
                noise=self.train_Yvar.squeeze(-1).expand(
                    num_mcmc_samples, len(self.train_Yvar)
                ),
                batch_shape=batch_shape,
            ).to(**tkwargs)
        else:
            likelihood = GaussianLikelihood(
                batch_shape=batch_shape,
                noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL),
            ).to(**tkwargs)
            likelihood.noise_covar.noise = reshape_and_detach(
                target=likelihood.noise_covar.noise,
                new_value=mcmc_samples["noise"].clamp_min(MIN_INFERRED_NOISE_LEVEL),
            )
        covar_module.base_kernel.lengthscale = reshape_and_detach(
            target=covar_module.base_kernel.lengthscale,
            new_value=mcmc_samples["lengthscale"],
        )
        covar_module.outputscale = reshape_and_detach(
            target=covar_module.outputscale,
            new_value=mcmc_samples["outputscale"],
        )
        mean_module.constant.data = reshape_and_detach(
            target=mean_module.constant.data,
            new_value=mcmc_samples["mean"],
        )
        return mean_module, covar_module, likelihood


class SaasFullyBayesianSingleTaskGP(ExactGP, BatchedMultiOutputGPyTorchModel):
    r"""A fully Bayesian single-task GP model with the SAAS prior.

    This model assumes that the inputs have been normalized to [0, 1]^d and thatcdcd
    the output has been standardized to have zero mean and unit variance. You can
    either normalize and standardize the data before constructing the model or use
    an `input_transform` and `outcome_transform`. The SAAS model [Eriksson2021saasbo]_
    with a Matern-5/2 kernel is used by default.

    You are expected to use `fit_fully_bayesian_model_nuts` to fit this model as it
    isn't compatible with `fit_gpytorch_model`.

    Example:
        >>> saas_gp = SaasFullyBayesianSingleTaskGP(train_X, train_Y)
        >>> fit_fully_bayesian_model_nuts(saas_gp)
        >>> posterior = saas_gp.posterior(test_X)
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Optional[Tensor] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        input_transform: Optional[InputTransform] = None,
        pyro_model: Optional[PyroModel] = None,
        num_samples: int = 256,
        warmup: int = 128,
        thinning: int = 16,
        disable_progbar: bool = False,
        num_groups: int = 0,
        hyperparameters: dict = None,
        warm_start_state: Optional[Any] = None,

    ) -> None:
        r"""Initialize the fully Bayesian single-task GP model.

        Args:
            train_X: Training inputs (n x d)
            train_Y: Training targets (n x 1)
            train_Yvar: Observed noise variance (n x 1). Inferred if None.
            outcome_transform: An outcome transform that is applied to the
                training data during instantiation and to the posterior during
                inference (that is, the `Posterior` obtained by calling
                `.posterior` on the model will be on the original scale).
            input_transform: An input transform that is applied in the model's
                forward pass.
            pyro_model: Optional `PyroModel`, defaults to `SaasPyroModel`.
        """
        if isinstance(train_Yvar, float):
            train_Yvar = torch.full_like(train_Y, train_Yvar)
        with torch.no_grad():
            transformed_X = self.transform_inputs(
                X=train_X, input_transform=input_transform
            )
        if outcome_transform is not None:
            train_Y, train_Yvar = outcome_transform(train_Y, train_Yvar)
        self._validate_tensor_args(X=transformed_X, Y=train_Y)
        validate_input_scaling(
            train_X=transformed_X, train_Y=train_Y, train_Yvar=train_Yvar
        )
        self._input_batch_shape, self._aug_batch_shape = self.get_batch_dimensions(
            train_X=train_X, train_Y=train_Y
        )
        num_mcmc_samples = num_samples // thinning
        if pyro_model is None:
            pyro_model = SaasPyroModel()

        pyro_model.set_inputs(
            train_X=transformed_X, train_Y=train_Y, train_Yvar=train_Yvar
        )
        self.pyro_model = pyro_model
        train_X = train_X.unsqueeze(0).expand(num_mcmc_samples, train_X.shape[0], -1)
        train_Y = train_Y.unsqueeze(0).expand(num_mcmc_samples, train_Y.shape[0], -1)
        self._num_outputs = train_Y.shape[-1]
        if train_Yvar is not None:  # Clamp after transforming
            train_Yvar = train_Yvar.clamp(MIN_INFERRED_NOISE_LEVEL)

        X_tf, Y_tf, _ = self._transform_tensor_args(X=train_X, Y=train_Y)
        super().__init__(
            train_inputs=X_tf, train_targets=Y_tf, likelihood=GaussianLikelihood()
        )
        self.mean_module = None
        self.covar_module = None
        self.likelihood = None
        self.pyro_model.set_warm_start_state(warm_start_state=warm_start_state)
        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        if input_transform is not None:
            self.input_transform = input_transform
        self.num_samples = num_samples
        self.warmup = warmup
        self.thinning = thinning
        self.disable_progbar = disable_progbar
        if num_groups > 0 and hasattr(self.pyro_model, 'num_groups'):
            self.pyro_model.set_num_groups(groups=num_groups)
            self.pyro_model.set_hyperparameters(**hyperparameters)

    def _check_if_fitted(self):
        r"""Raise an exception if the model hasn't been fitted."""
        if self.covar_module is None:
            raise RuntimeError(
                "Model has not been fitted. You need to call "
                "`fit_fully_bayesian_model_nuts` to fit the model."
            )

    @property
    def median_lengthscale(self) -> Tensor:
        r"""Median lengthscales across the MCMC samples."""
        self._check_if_fitted()
        lengthscale = self.covar_module.base_kernel.lengthscale.clone()
        return lengthscale.median(0).values.squeeze(0)

    @property
    def num_mcmc_samples(self) -> int:
        r"""Number of MCMC samples in the model."""
        self._check_if_fitted()
        return len(self.covar_module.outputscale)

    @property
    def batch_shape(self) -> torch.Size:
        r"""Batch shape of the model, equal to the number of MCMC samples.
        Note that `SaasFullyBayesianSingleTaskGP` does not support batching
        over input data at this point."""
        return torch.Size([self.num_mcmc_samples])

    def train(self, mode: bool = True) -> None:
        r"""Puts the model in `train` mode."""
        super().train(mode=mode)
        if mode:
            self.mean_module = None
            self.covar_module = None
            self.likelihood = None

    def load_mcmc_samples(self, mcmc_samples: Dict[str, Tensor]) -> None:
        r"""Load the MCMC hyperparameter samples into the model.

        This method will be called by `fit_fully_bayesian_model_nuts` when the model
        has been fitted in order to create a batched SingleTaskGP model.
        """
        (
            self.mean_module,
            self.covar_module,
            self.likelihood,
        ) = self.pyro_model.load_mcmc_samples(mcmc_samples=mcmc_samples)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        r"""Custom logic for loading the state dict.

        The standard approach of calling `load_state_dict` currently doesn't play well
        with the `SaasFullyBayesianSingleTaskGP` since the `mean_module`, `covar_module`
        and `likelihood` aren't initialized until the model has been fitted. The reason
        for this is that we don't know the number of MCMC samples until NUTS is called.
        Given the state dict, we can initialize a new model with some dummy samples and
        then load the state dict into this model. This currently only works for a
        `SaasPyroModel` and supporting more Pyro models likely requires moving the model
        construction logic into the Pyro model itself.
        """

        if not isinstance(self.pyro_model, SaasPyroModel):
            raise NotImplementedError("load_state_dict only works for SaasPyroModel")
        raw_mean = state_dict["mean_module.raw_constant"]
        num_mcmc_samples = len(raw_mean)
        dim = self.pyro_model.train_X.shape[-1]
        tkwargs = {"device": raw_mean.device, "dtype": raw_mean.dtype}
        # Load some dummy samples
        mcmc_samples = {
            "mean": torch.ones(num_mcmc_samples, **tkwargs),
            "lengthscale": torch.ones(num_mcmc_samples, dim, **tkwargs),
            "outputscale": torch.ones(num_mcmc_samples, **tkwargs),
        }
        if self.pyro_model.train_Yvar is None:
            mcmc_samples["noise"] = torch.ones(num_mcmc_samples, **tkwargs)
        (
            self.mean_module,
            self.covar_module,
            self.likelihood,
        ) = self.pyro_model.load_mcmc_samples(mcmc_samples=mcmc_samples)
        # Load the actual samples from the state dict
        super().load_state_dict(state_dict=state_dict, strict=strict)

    def forward(self, X: Tensor) -> MultivariateNormal:
        """
        Unlike in other classes' `forward` methods, there is no `if self.training`
        block, because it ought to be unreachable: If `self.train()` has been called,
        then `self.covar_module` will be None, `check_if_fitted()` will fail, and the
        rest of this method will not run.
        """
        self._check_if_fitted()
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        return MultivariateNormal(mean_x, covar_x)

    # pyre-ignore[14]: Inconsistent override
    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        posterior_transform: Optional[PosteriorTransform] = None,
        predict_per_model: bool = False,
        **kwargs: Any,
    ) -> FullyBayesianPosterior:
        r"""Computes the posterior over model outputs at the provided points.

        Args:
            X: A `(batch_shape) x q x d`-dim Tensor, where `d` is the dimension
                of the feature space and `q` is the number of points considered
                jointly.
            output_indices: A list of indices, corresponding to the outputs over
                which to compute the posterior (if the model is multi-output).
                Can be used to speed up computation if only a subset of the
                model's outputs are required for optimization. If omitted,
                computes the posterior over all model outputs.
            observation_noise: If True, add the observation noise from the
                likelihood to the posterior. If a Tensor, use it directly as the
                observation noise (must be of shape `(batch_shape) x q x m`).
            posterior_transform: An optional PosteriorTransform.

        Returns:
            A `FullyBayesianPosterior` object. Includes observation noise if specified.
        """
        self._check_if_fitted()
        # X_batch = X.unsqueeze(0).repeat(self.num_batch, 1, 1, 1)
        if predict_per_model:
            posterior = super().posterior(
                X=X,
                output_indices=output_indices,
                observation_noise=observation_noise,
                posterior_transform=posterior_transform,
                **kwargs,
            )
        else:
            posterior = super().posterior(
                X=X.unsqueeze(MCMC_DIM),
                output_indices=output_indices,
                observation_noise=observation_noise,
                posterior_transform=posterior_transform,
                **kwargs,
            )
        posterior = FullyBayesianPosterior(distribution=posterior.distribution)
        return posterior

    def fit(self) -> None:
        r"""Load the MCMC hyperparameter samples into the model.

        This method will be called by `fit_fully_bayesian_model_nuts` when the model
        has been fitted in order to create a batched SingleTaskGP model.
        """
        if self.pyro_model.custom_fit:
            from botorch.fit import fit_fully_bayesian_model_custom
            fit_fully_bayesian_model_custom(
                model=self,
                warmup_steps=self.warmup,
                num_samples=self.num_samples,
                thinning=self.thinning,
                disable_progbar=self.disable_progbar
            )
        else:
            from botorch.fit import fit_fully_bayesian_model_nuts
            fit_fully_bayesian_model_nuts(
                model=self,
                warmup_steps=self.warmup,
                num_samples=self.num_samples,
                thinning=self.thinning,
                disable_progbar=self.disable_progbar
            )

    def fantasize(
        self,
        X: Tensor,
        sampler: MCSampler,
        observation_noise: Union[bool, Tensor] = True,
        **kwargs: Any,
    ):
        r"""Construct a fantasy model.

        Constructs a fantasy model in the following fashion:
        (1) compute the model posterior at `X` (if `observation_noise=True`,
        this includes observation noise taken as the mean across the observation
        noise in the training data. If `observation_noise` is a Tensor, use
        it directly as the observation noise to add).
        (2) sample from this posterior (using `sampler`) to generate "fake"
        observations.
        (3) condition the model on the new fake observations.

        Args:
            X: A `batch_shape x n' x d`-dim Tensor, where `d` is the dimension of
                the feature space, `n'` is the number of points per batch, and
                `batch_shape` is the batch shape (must be compatible with the
                batch shape of the model).
            sampler: The sampler used for sampling from the posterior at `X`.
            observation_noise: If True, include the mean across the observation
                noise in the training data as observation noise in the posterior
                from which the samples are drawn. If a Tensor, use it directly
                as the specified measurement noise.

        Returns:
            The constructed fantasy model.
        """
        propagate_grads = kwargs.pop("propagate_grads", False)

        with fantasize_flag():
            with settings.propagate_grads(propagate_grads):
                post_X = self.posterior(
                    X, observation_noise=observation_noise, **kwargs
                )
            X = X.unsqueeze(MCMC_DIM).repeat(1, post_X.shape()[1], 1, 1)

            Y_fantasized = sampler(post_X)  # num_fantasies x batch_shape x n' x m
            # Use the mean of the previous noise values (TODO: be smarter here).
            # noise should be batch_shape x q x m when X is batch_shape x q x d, and
            # Y_fantasized is num_fantasies x batch_shape x q x m.
            noise_shape = Y_fantasized.shape[1:]
            noise = self.likelihood.noise.unsqueeze(-1).expand(noise_shape)
            return self.condition_on_observations(
                X=self.transform_inputs(X), Y=Y_fantasized, noise=noise
            )

    def get_warm_start_state(self):
        return self.pyro_model.get_warm_start_state()


class WarpingFullyBayesianSingleTaskGP(ExactGP, BatchedMultiOutputGPyTorchModel):
    r"""A fully Bayesian single-task GP model with the SAAS prior.

    This model assumes that the inputs have been normalized to [0, 1]^d and thatcdcd
    the output has been standardized to have zero mean and unit variance. You can
    either normalize and standardize the data before constructing the model or use
    an `input_transform` and `outcome_transform`. The SAAS model [Eriksson2021saasbo]_
    with a Matern-5/2 kernel is used by default.

    You are expected to use `fit_fully_bayesian_model_nuts` to fit this model as it
    isn't compatible with `fit_gpytorch_model`.

    Example:
        >>> saas_gp = SaasFullyBayesianSingleTaskGP(train_X, train_Y)
        >>> fit_fully_bayesian_model_nuts(saas_gp)
        >>> posterior = saas_gp.posterior(test_X)
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Optional[Tensor] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        input_transform: Optional[InputTransform] = None,
        pyro_model: Optional[PyroModel] = None,
        num_samples: int = 256,
        warmup: int = 128,
        thinning: int = 16,
        disable_progbar: bool = False
    ) -> None:
        r"""Initialize the fully Bayesian single-task GP model.

        Args:
            train_X: Training inputs (n x d)
            train_Y: Training targets (n x 1)
            train_Yvar: Observed noise variance (n x 1). Inferred if None.
            outcome_transform: An outcome transform that is applied to the
                training data during instantiation and to the posterior during
                inference (that is, the `Posterior` obtained by calling
                `.posterior` on the model will be on the original scale).
            input_transform: An input transform that is applied in the model's
                forward pass.
            pyro_model: Optional `PyroModel`, defaults to `SaasPyroModel`.
        """
        num_mcmc_samples = num_samples // thinning
        train_X = train_X.unsqueeze(0).expand(num_mcmc_samples, train_X.shape[0], -1)
        train_Y = train_Y.unsqueeze(0).expand(num_mcmc_samples, train_Y.shape[0], -1)

        with torch.no_grad():
            transformed_X = self.transform_inputs(
                X=train_X, input_transform=input_transform
            )
        if outcome_transform is not None:
            train_Y, train_Yvar = outcome_transform(train_Y, train_Yvar)

        self._validate_tensor_args(X=transformed_X, Y=train_Y)
        validate_input_scaling(
            train_X=transformed_X, train_Y=train_Y, train_Yvar=train_Yvar
        )
        self._input_batch_shape, self._aug_batch_shape = self.get_batch_dimensions(
            train_X=train_X, train_Y=train_Y
        )
        self._num_outputs = train_Y.shape[-1]
        if train_Yvar is not None:  # Clamp after transforming
            train_Yvar = train_Yvar.clamp(MIN_INFERRED_NOISE_LEVEL)

        X_tf, Y_tf, _ = self._transform_tensor_args(X=train_X, Y=train_Y)
        super().__init__(
            train_inputs=X_tf, train_targets=Y_tf, likelihood=GaussianLikelihood()
        )

        self.mean_module = None
        self.covar_module = None
        self.likelihood = None
        if pyro_model is None:
            pyro_model = SaasPyroModel()

        pyro_model.set_inputs(
            train_X=transformed_X, train_Y=train_Y, train_Yvar=train_Yvar
        )
        self.pyro_model = pyro_model
        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        if input_transform is not None:
            self.input_transform = input_transform
        self.num_samples = num_samples
        self.warmup = warmup
        self.thinning = thinning
        self.disable_progbar = disable_progbar
        # self.num_batch = num_mcmc_samples

    def _check_if_fitted(self):
        r"""Raise an exception if the model hasn't been fitted."""
        if self.covar_module is None:
            raise RuntimeError(
                "Model has not been fitted. You need to call "
                "`fit_fully_bayesian_model_nuts` to fit the model."
            )

    @property
    def median_lengthscale(self) -> Tensor:
        r"""Median lengthscales across the MCMC samples."""
        self._check_if_fitted()
        lengthscale = self.covar_module.base_kernel.lengthscale.clone()
        return lengthscale.median(0).values.squeeze(0)

    @property
    def num_mcmc_samples(self) -> int:
        r"""Number of MCMC samples in the model."""
        self._check_if_fitted()
        return len(self.covar_module.outputscale)

    @property
    def batch_shape(self) -> torch.Size:
        r"""Batch shape of the model, equal to the number of MCMC samples.
        Note that `SaasFullyBayesianSingleTaskGP` does not support batching
        over input data at this point."""
        return torch.Size([self.num_mcmc_samples])

    def train(self, mode: bool = True) -> None:
        r"""Puts the model in `train` mode."""
        super().train(mode=mode)
        if mode:
            self.mean_module = None
            self.covar_module = None
            self.likelihood = None

    def load_mcmc_samples(self, mcmc_samples: Dict[str, Tensor]) -> None:
        r"""Load the MCMC hyperparameter samples into the model.

        This method will be called by `fit_fully_bayesian_model_nuts` when the model
        has been fitted in order to create a batched SingleTaskGP model.
        """
        (
            self.mean_module,
            self.covar_module,
            self.likelihood,
            self.input_transform
        ) = self.pyro_model.load_mcmc_samples(mcmc_samples=mcmc_samples)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        r"""Custom logic for loading the state dict.

        The standard approach of calling `load_state_dict` currently doesn't play well
        with the `SaasFullyBayesianSingleTaskGP` since the `mean_module`, `covar_module`
        and `likelihood` aren't initialized until the model has been fitted. The reason
        for this is that we don't know the number of MCMC samples until NUTS is called.
        Given the state dict, we can initialize a new model with some dummy samples and
        then load the state dict into this model. This currently only works for a
        `SaasPyroModel` and supporting more Pyro models likely requires moving the model
        construction logic into the Pyro model itself.
        """

        if not isinstance(self.pyro_model, SaasPyroModel):
            raise NotImplementedError("load_state_dict only works for SaasPyroModel")
        raw_mean = state_dict["mean_module.raw_constant"]
        num_mcmc_samples = len(raw_mean)
        dim = self.pyro_model.train_X.shape[-1]
        tkwargs = {"device": raw_mean.device, "dtype": raw_mean.dtype}
        # Load some dummy samples
        mcmc_samples = {
            "mean": torch.ones(num_mcmc_samples, **tkwargs),
            "lengthscale": torch.ones(num_mcmc_samples, dim, **tkwargs),
            "outputscale": torch.ones(num_mcmc_samples, **tkwargs),
        }
        if self.pyro_model.train_Yvar is None:
            mcmc_samples["noise"] = torch.ones(num_mcmc_samples, **tkwargs)
        (
            self.mean_module,
            self.covar_module,
            self.likelihood,
        ) = self.pyro_model.load_mcmc_samples(mcmc_samples=mcmc_samples)
        # Load the actual samples from the state dict
        super().load_state_dict(state_dict=state_dict, strict=strict)

    def forward(self, X: Tensor) -> MultivariateNormal:
        """
        Unlike in other classes' `forward` methods, there is no `if self.training`
        block, because it ought to be unreachable: If `self.train()` has been called,
        then `self.covar_module` will be None, `check_if_fitted()` will fail, and the
        rest of this method will not run.
        """
        self._check_if_fitted()
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        return MultivariateNormal(mean_x, covar_x)

    # pyre-ignore[14]: Inconsistent override
    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        posterior_transform: Optional[PosteriorTransform] = None,
        predict_per_model: bool = False,
        **kwargs: Any,
    ) -> FullyBayesianPosterior:
        r"""Computes the posterior over model outputs at the provided points.

        Args:
            X: A `(batch_shape) x q x d`-dim Tensor, where `d` is the dimension
                of the feature space and `q` is the number of points considered
                jointly.
            output_indices: A list of indices, corresponding to the outputs over
                which to compute the posterior (if the model is multi-output).
                Can be used to speed up computation if only a subset of the
                model's outputs are required for optimization. If omitted,
                computes the posterior over all model outputs.
            observation_noise: If True, add the observation noise from the
                likelihood to the posterior. If a Tensor, use it directly as the
                observation noise (must be of shape `(batch_shape) x q x m`).
            posterior_transform: An optional PosteriorTransform.

        Returns:
            A `FullyBayesianPosterior` object. Includes observation noise if specified.
        """
        self._check_if_fitted()
        # X_batch = X.unsqueeze(0).repeat(self.num_batch, 1, 1, 1)

        if predict_per_model:
            posterior = super().posterior(
                X=X,
                output_indices=output_indices,
                observation_noise=observation_noise,
                posterior_transform=posterior_transform,
                **kwargs,
            )

        else:
            posterior = super().posterior(
                X=X.unsqueeze(MCMC_DIM),
                output_indices=output_indices,
                observation_noise=observation_noise,
                posterior_transform=posterior_transform,
                **kwargs,
            )
        posterior = FullyBayesianPosterior(distribution=posterior.distribution)
        return posterior

    def fit(self) -> None:
        r"""Load the MCMC hyperparameter samples into the model.

        This method will be called by `fit_fully_bayesian_model_nuts` when the model
        has been fitted in order to create a batched SingleTaskGP model.
        """
        from botorch.fit import fit_fully_bayesian_model_nuts
        fit_fully_bayesian_model_nuts(
            model=self,
            warmup_steps=self.warmup,
            num_samples=self.num_samples,
            thinning=self.thinning,
            disable_progbar=self.disable_progbar
        )

