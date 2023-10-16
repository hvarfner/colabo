#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Batch acquisition functions using the reparameterization trick in combination
with (quasi) Monte-Carlo sampling. See [Rezende2014reparam]_, [Wilson2017reparam]_ and
[Balandat2020botorch]_.

.. [Rezende2014reparam]
    D. J. Rezende, S. Mohamed, and D. Wierstra. Stochastic backpropagation and
    approximate inference in deep generative models. ICML 2014.

.. [Wilson2017reparam]
    J. T. Wilson, R. Moriconi, F. Hutter, and M. P. Deisenroth.
    The reparameterization trick for acquisition functions. ArXiv 2017.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Optional, Union
import warnings
from functools import partial
from botorch.models.utils import fantasize as fantasize_flag

import numpy as np
import torch
from torch.quasirandom import SobolEngine
from botorch.acquisition.acquisition import AcquisitionFunction, MCSamplerMixin, OneShotAcquisitionFunction
from botorch.acquisition.cached_cholesky import CachedCholeskyMCAcquisitionFunction
from botorch.acquisition.objective import (
    IdentityMCObjective,
    MCAcquisitionObjective,
    PosteriorTransform,
)
from botorch.acquisition.logei import _log_improvement
from botorch.utils.safe_math import (
    fatmax,
    log_fatplus,
    log_softplus,
    logmeanexp,
    smooth_amax,
)
from torch.distributions import Normal
from botorch.models.utils import check_no_nans
from botorch import settings
from botorch.acquisition.knowledge_gradient import _split_fantasy_points, _get_value_function
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.exceptions.errors import UnsupportedError
from botorch.models.model import Model
from botorch.posteriors.posterior import Posterior
from botorch.sampling.base import MCSampler
from botorch.sampling.pathwise_sampler import PathwiseSampler
from botorch.utils.transforms import (
    concatenate_pending_points,
    match_batch_shape,
    t_batch_mode_transform,
)

from botorch.acquisition.utils import (
    compute_best_feasible_objective,
    prune_inferior_points,
)
from botorch.utils.prior import UserPrior
from botorch.sampling.normal import IIDNormalSampler
from torch import Tensor

from botorch.sampling.pathwise.update_strategies import gaussian_update
from botorch.sampling.pathwise import MatheronPath
from botorch.models.gp_regression import MIN_INFERRED_NOISE_LEVEL

TRUNC_MARGIN = 10
OPT_ERROR = 1e-2

TAU_RELU = 1e-6
TAU_MAX = 1e-2


class PriorMCAcquisitionFunction(MCAcquisitionFunction):
    r"""MC-based batch Expected Improvement.

    This computes qEI by
    (1) sampling the joint posterior over q points
    (2) evaluating the improvement over the current best for each sample
    (3) maximizing over q
    (4) averaging over the samples

    `qEI(X) = E(max(max Y - best_f, 0)), Y ~ f(X), where X = (x_1,...,x_q)`

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> best_f = train_Y.max()[0]
        >>> sampler = SobolQMCNormalSampler(1024)
        >>> qEI = qExpectedImprovement(model, best_f, sampler)
        >>> qei = qEI(test_X)
    """

    def __init__(
        self,
        model: Model,
        paths: MatheronPath,
        sampler: PathwiseSampler,
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        X_pending: Optional[Tensor] = None,
        user_prior: Optional[UserPrior] = None,
        user_prior_value: Optional[UserPrior] = None,
        decay_beta: float = 10.0,
        prior_floor: float = 1e-9,
        custom_decay: Optional[float] = None,
        use_resampling: bool = True,
        resampling_fraction: float = 0.05,
        plot: bool = False,
        decay_power: float = 2.0, 
        ** kwargs: Any,
    ) -> None:
        r"""q-Expected Improvement with arbitrary user priors.

        Args:
            model: A fitted model.
            best_f: The best objective value observed so far (assumed noiseless). Can be
                a `batch_shape`-shaped tensor, which in case of a batched model
                specifies potentially different values for each element of the batch.
            sampler: The sampler used to draw base samples. See `MCAcquisitionFunction`
                more details.
            objective: The MCAcquisitionObjective under which the samples are evaluated.
                Defaults to `IdentityMCObjective()`.
            posterior_transform: A PosteriorTransform (optional).
            X_pending:  A `m x d`-dim Tensor of `m` design points that have been
                submitted for function evaluation but have not yet been evaluated.
                Concatenated into X upon forward call. Copied and set to have no Papperskorg1!
                gradient.
        """
        sampling_model = deepcopy(model)
        sampling_model.set_paths(paths)
        super().__init__(
            model=sampling_model,  # TODO or just model - check what fantasies are being used
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
        )
        if plot:
            from copy import copy
        
            self.old_paths = MatheronPath(paths.paths.prior_paths, paths.paths.update_paths)
        
        self.old_model = model
        self.sampling_model = sampling_model
        if custom_decay is not None:
            self.decay_factor = custom_decay
        else:
            self.decay_factor = decay_beta / \
                ((len(self.model.train_targets)) ** decay_power)

        self.prior_floor = torch.Tensor([prior_floor])
        self.user_prior = user_prior
        if self.user_prior is None:
            self.sample_probs = torch.ones(self.sampler.sample_shape).reshape(-1, 1, 1)
        else:
            self.user_prior.register_maxval(optval_prior=user_prior_value)
            self.sample_probs = self.user_prior.compute_norm_probs(
                self.sampling_model.paths.paths.prior_paths, self.decay_factor, self.prior_floor).reshape(-1, 1, 1).detach()

            if torch.any(torch.isnan(self.sample_probs)):
                print('Some value in sample probs is nan')
                print(torch.isnan(self.sample_probs).sum())
                raise SystemExit

            self.sample_probs = self.sample_probs / self.sample_probs.mean()

        if user_prior is not None and use_resampling:
            am = self.sample_probs.argmax()
            torch.sqrt(torch.pow(self.user_prior.optimal_inputs - 0.5
                       * torch.ones_like(self.user_prior.optimal_inputs), 2).sum(dim=-1)).min()
            _, self.sample_probs, indices = resample(
                paths, self.sample_probs, resampling_fraction)
            self.indices = indices

            # TODO switch places when doing the super() call
            self.sampling_model.set_paths(paths, self.indices)
            self.model = self.sampling_model
            self.sample_probs = (self.sample_probs / self.sample_probs.mean()).detach()
        

class qPriorExpectedImprovement(PriorMCAcquisitionFunction):
    r"""MC-based batch Expected Improvement.
    This computes qEI by
    (1) sampling the joint posterior over q points
    (2) evaluating the improvement over the current best for each sample
    (3) maximizing over q
    (4) averaging over the samples

    `qEI(X) = E(max(max Y - best_f, 0)), Y ~ f(X), where X = (x_1,...,x_q)`

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> best_f = train_Y.max()[0]
        >>> sampler = SobolQMCNormalSampler(1024)
        >>> qEI = qExpectedImprovement(model, best_f, sampler)
        >>> qei = qEI(test_X)
    """

    def __init__(
        self,
        model: Model,
        paths: MatheronPath,
        sampler: PathwiseSampler,
        X_baseline: Tensor,
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        X_pending: Optional[Tensor] = None,
        user_prior: Optional[UserPrior] = None,
        user_prior_value: Optional[UserPrior] = None,
        decay_beta: float = 10.0,
        prior_floor: float = 1e-9,
        custom_decay: Optional[float] = None,
        use_resampling: bool = True,
        resampling_fraction: float = 0.1,
        **kwargs: Any,
    ) -> None:
        r"""q-Expected Improvement with arbitrary user priors.

        Args:
            model: A fitted model.
            best_f: The best objective value observed so far (assumed noiseless). Can be
                a `batch_shape`-shaped tensor, which in case of a batched model
                specifies potentially different values for each element of the batch.
            sampler: The sampler used to draw base samples. See `MCAcquisitionFunction`
                more details.
            objective: The MCAcquisitionObjective under which the samples are evaluated.
                Defaults to `IdentityMCObjective()`.
            posterior_transform: A PosteriorTransform (optional).
            X_pending:  A `m x d`-dim Tensor of `m` design points that have been
                submitted for function evaluation but have not yet been evaluated.
                Concatenated into X upon forward call. Copied and set to have no
                gradient.
        """
        super().__init__(
            model=model,
            paths=paths,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
            user_prior=user_prior,
            user_prior_value=user_prior_value,
            decay_beta=decay_beta,
            prior_floor=prior_floor,
            custom_decay=custom_decay,
            use_resampling=use_resampling,
            resampling_fraction=resampling_fraction,
        )
        X_baseline = prune_inferior_points(
            model=model,
            X=X_baseline,
            objective=objective,
            posterior_transform=posterior_transform,
            marginalize_dim=kwargs.get("marginalize_dim"),
        )
        self.register_buffer("X_baseline", X_baseline)
        # self.register_buffer("best_f", torch.as_tensor(best_f, dtype=float))
        if kwargs.get('plot', False):
            from botorch.utils.plot_acq import plot_paper
            plot_paper(self)

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor, use_prior: bool = True) -> Tensor:
        r"""Evaluate qExpectedImprovement on the candidate set `X`.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of t-batches with `q` `d`-dim design
                points each.

        Returns:
            A `batch_shape'`-dim Tensor of Expected Improvement values at the given
            design points `X`, where `batch_shape'` is the broadcasted batch shape of
            model and input `X`.
        """
        q = X.shape[-2]
        X_full = torch.cat([match_batch_shape(self.X_baseline, X), X], dim=-2)
        # TODO: Implement more efficient way to compute posterior over both training and
        # test points in GPyTorch (https://github.com/cornellius-gp/gpytorch/issues/567)
        posterior = self.sampling_model.posterior(
            X_full, posterior_transform=self.posterior_transform
        )
        samples = self.get_posterior_samples(posterior)
        obj = self.objective(samples, X=X_full)
        diffs = obj[..., -q:].max(dim=-1).values - obj[..., :-q].max(dim=-1).values

        if use_prior:
            diffs = diffs * self.sample_probs.squeeze(-1)
        return diffs.clamp_min(0).mean(dim=0)


class qPriorLogExpectedImprovement(PriorMCAcquisitionFunction):
    r"""MC-based batch Expected Improvement.

    This computes qEI by
    (1) sampling the joint posterior over q points
    (2) evaluating the improvement over the current best for each sample
    (3) maximizing over q
    (4) averaging over the samples

    `qEI(X) = E(max(max Y - best_f, 0)), Y ~ f(X), where X = (x_1,...,x_q)`

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> best_f = train_Y.max()[0]
        >>> sampler = SobolQMCNormalSampler(1024)
        >>> qEI = qExpectedImprovement(model, best_f, sampler)
        >>> qei = qEI(test_X)
    """

    def __init__(
        self,
        model: Model,
        paths: MatheronPath,
        sampler: PathwiseSampler,
        X_baseline: Tensor,
        best_f: Optional[Tensor] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        X_pending: Optional[Tensor] = None,
        user_prior: Optional[UserPrior] = None,
        user_prior_value: Optional[UserPrior] = None,
        decay_beta: float = 10.0,
        prior_floor: float = 1e-9,
        custom_decay: Optional[float] = None,
        use_resampling: bool = True,
        resampling_fraction: float = 0.1,
        prune_baseline: bool = True,
        cache_root: bool = True,
        eta: Union[Tensor, float] = 1e-3,
        fat: bool = True,  # LogEI stuff
        tau_max: float = TAU_MAX,  # LogEI stuff
        tau_relu: float = TAU_RELU,  # LogEI stuff
        plot: bool = False,
        max_baseline: int = 5,
        decay_power: float = 2,
        **kwargs: Any,
    ) -> None:
        r"""q-Expected Improvement with arbitrary user priors.

        Args:
            model: A fitted model.
            best_f: The best objective value observed so far (assumed noiseless). Can be
                a `batch_shape`-shaped tensor, which in case of a batched model
                specifies potentially different values for each element of the batch.
            sampler: The sampler used to draw base samples. See `MCAcquisitionFunction`
                more details.
            objective: The MCAcquisitionObjective under which the samples are evaluated.
                Defaults to `IdentityMCObjective()`.
            posterior_transform: A PosteriorTransform (optional).
            X_pending:  A `m x d`-dim Tensor of `m` design points that have been
                submitted for function evaluation but have not yet been evaluated.
                Concatenated into X upon forward call. Copied and set to have no
                gradient.
        """
        self.q_reduction = partial(fatmax if fat else smooth_amax, tau=tau_max)
        self.sample_reduction = logmeanexp
        self.eta = eta
        self.fat = fat
        self.tau_max = tau_max
        self._fat = False
        self._constraints = None
        self.max_baseline = max_baseline

        # TODO - ARGUMENT Q HERE?

        super().__init__(
            model=model,
            paths=paths,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
            user_prior=user_prior,
            user_prior_value=user_prior_value,
            decay_beta=decay_beta,
            prior_floor=prior_floor,
            custom_decay=custom_decay,
            use_resampling=use_resampling,
            resampling_fraction=resampling_fraction,
            plot=plot,
            decay_power=decay_power,
        )

        self.tau_relu = tau_relu
        self._init_baseline(
            model=model,
            X_baseline=X_baseline,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            prune_baseline=prune_baseline,
            cache_root=cache_root,
            **kwargs,
        )

        # self.forward(*self.model.train_inputs)
        #self.forward(self.model.train_inputs[0] + torch.randn((25, 2)) * 0.025)
        from botorch.utils.plot_acq import plot_prior, plot_surface
        if model.train_inputs[0].shape[-1] == 2 and plot:
            opt = plot_surface(self)
        if model.train_inputs[0].shape[-1] == 1 and plot:
            plot_prior(self)

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor, use_prior: bool = True) -> Tensor:

        samples, obj = self._get_samples_and_objectives(X)
        non_reduced_acq = self._sample_forward(obj)
        if use_prior:
            non_reduced_acq = non_reduced_acq + torch.log(self.sample_probs)
        q_reduced_acq = self.q_reduction(non_reduced_acq, dim=-1)
        acq = self.sample_reduction(q_reduced_acq, dim=0)
        return acq

    def _sample_forward(self, obj: Tensor) -> Tensor:
        r"""Evaluate qLogNoisyExpectedImprovement per sample on the candidate set `X`.

        Args:   
            obj: `mc_shape x batch_shape x q`-dim Tensor of MC objective values.

        Returns:
            A `sample_shape x batch_shape x q`-dim Tensor of log noisy expected smoothed
            improvement values.
        """
        return _log_improvement(
            Y=obj,
            best_f=self.compute_best_f(obj),
            tau=self.tau_relu,
            fat=self._fat,
        )

    def _init_baseline(
        self,
        model: Model,
        X_baseline: Tensor,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        prune_baseline: bool = False,
        cache_root: bool = True,
        **kwargs: Any,
    ) -> None:
        # setup of CachedCholeskyMCAcquisitionFunction
        self._cache_root = False
        #self._setup(model=model, cache_root=cache_root)
        if prune_baseline:
            X_baseline = prune_inferior_points(
                model=model,
                X=X_baseline,
                objective=objective,
                max_frac=min(1, self.max_baseline / len(model.train_inputs[0])),
                posterior_transform=posterior_transform,
                marginalize_dim=kwargs.get("marginalize_dim"),
            )
        self.register_buffer("X_baseline", X_baseline)
        # registering buffers for _get_samples_and_objectives in the next `if` block
        self.register_buffer("baseline_samples", None)
        self.register_buffer("baseline_obj", None)
        if self._cache_root:
            self.q_in = -1
            # set baseline samples
            with torch.no_grad():  # this is _get_samples_and_objectives(X_baseline)
                posterior = self.model.posterior(
                    X_baseline, posterior_transform=self.posterior_transform
                )
                # Note: The root decomposition is cached in two different places. It
                # may be confusing to have two different caches, but this is not
                # trivial to change since each is needed for a different reason:
                # - LinearOperator caching to `posterior.mvn` allows for reuse within
                #   this function, which may be helpful if the same root decomposition
                #   is produced by the calls to `self.base_sampler` and
                #   `self._cache_root_decomposition`.
                # - self._baseline_L allows a root decomposition to be persisted outside
                #   this method.
                self.baseline_samples = self.get_posterior_samples(posterior)
                self.baseline_obj = self.objective(self.baseline_samples, X=X_baseline)

            # We make a copy here because we will write an attribute `base_samples`
            # to `self.base_sampler.base_samples`, and we don't want to mutate
            # `self.sampler`.
            self.base_sampler = deepcopy(self.sampler)
            self.register_buffer(
                "_baseline_best_f",
                self._compute_best_feasible_objective(
                    samples=self.baseline_samples, obj=self.baseline_obj
                ),
            )
            self._baseline_L = self._compute_root_decomposition(posterior=posterior)

    def compute_best_f(self, obj: Tensor) -> Tensor:
        """Computes the best (feasible) noisy objective value.

        Args:
            obj: `sample_shape x batch_shape x q`-dim Tensor of objectives in forward.

        Returns:
            A `sample_shape x batch_shape x 1`-dim Tensor of best feasible objectives.
        """
        if self._cache_root:
            val = self._baseline_best_f
        else:
            val = self._compute_best_feasible_objective(
                samples=self.baseline_samples, obj=self.baseline_obj
            )
        # ensuring shape, dtype, device compatibility with obj
        n_sample_dims = len(self.sample_shape)
        view_shape = torch.Size(
            [
                *val.shape[:n_sample_dims],  # sample dimensions
                *(1,) * (obj.ndim - val.ndim),  # pad to match obj
                *val.shape[n_sample_dims:],  # the rest
            ]
        )
        return val.view(view_shape).to(obj)

    def _get_samples_and_objectives(self, X: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Compute samples at new points, using the cached root decomposition.

        Args:
            X: A `batch_shape x q x d`-dim tensor of inputs.

        Returns:
            A two-tuple `(samples, obj)`, where `samples` is a tensor of posterior
            samples with shape `sample_shape x batch_shape x q x m`, and `obj` is a
            tensor of MC objective values with shape `sample_shape x batch_shape x q`.
        """
        n_baseline, q = self.X_baseline.shape[-2], X.shape[-2]
        X_full = torch.cat([match_batch_shape(self.X_baseline, X), X], dim=-2)
        # TODO: Implement more efficient way to compute posterior over both training and
        # test points in GPyTorch (https://github.com/cornellius-gp/gpytorch/issues/567)
        posterior = self.model.posterior(
            X_full, posterior_transform=self.posterior_transform
        )
        if not self._cache_root:
            samples_full = super().get_posterior_samples(posterior)
            obj_full = self.objective(samples_full, X=X_full)
            # assigning baseline buffers so `best_f` can be computed in _sample_forward
            self.baseline_samples, samples = samples_full.split([n_baseline, q], dim=-2)
            self.baseline_obj, obj = obj_full.split([n_baseline, q], dim=-1)
            return samples, obj

        # handle one-to-many input transforms
        n_plus_q = X_full.shape[-2]
        n_w = posterior._extended_shape()[-2] // n_plus_q
        q_in = q * n_w
        self._set_sampler(q_in=q_in, posterior=posterior)
        samples = self._get_f_X_samples(posterior=posterior, q_in=q_in)
        obj = self.objective(samples, X=X_full[..., -q:, :])
        return samples, obj

    def _compute_best_feasible_objective(self, samples: Tensor, obj: Tensor) -> Tensor:
        return compute_best_feasible_objective(
            samples=samples,
            obj=obj,
            constraints=self._constraints,
            model=self.model,
            objective=self.objective,
            posterior_transform=self.posterior_transform,
            X_baseline=self.X_baseline,
        )


class qPriorUpperConfidenceBound(PriorMCAcquisitionFunction):
    r"""MC-based batch Upper Confidence Bound.

    Uses a reparameterization to extend UCB to qUCB for q > 1 (See Appendix A
    of [Wilson2017reparam].)

    `qUCB = E(max(mu + |Y_tilde - mu|))`, where `Y_tilde ~ N(mu, beta pi/2 Sigma)`
    and `f(X)` has distribution `N(mu, Sigma)`.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> sampler = SobolQMCNormalSampler(1024)
        >>> qUCB = qUpperConfidenceBound(model, 0.1, sampler)
        >>> qucb = qUCB(test_X)
    """

    def __init__(
        self,
        model: Model,
        paths: MatheronPath,
        sampler: PathwiseSampler,
        beta: float,
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        user_prior: Optional[UserPrior] = None,
        user_prior_value: Optional[UserPrior] = None,
        decay_beta: float = 10.0,
        prior_floor: float = 1e-9,
        X_pending: Optional[Tensor] = None,
        custom_decay: Optional[float] = None,
        use_resampling: bool = True,
        resampling_fraction: float = 0.1,

    ) -> None:
        r"""q-Upper Confidence Bound.

        Args:
            model: A fitted model.
            beta: Controls tradeoff between mean and standard deviation in UCB.
            sampler: The sampler used to draw base samples. See `MCAcquisitionFunction`
                more details.
            objective: The MCAcquisitionObjective under which the samples are
                evaluated. Defaults to `IdentityMCObjective()`.
            posterior_transform: A PosteriorTransform (optional).
            X_pending: A `batch_shape x m x d`-dim Tensor of `m` design points that have
                points that have been submitted for function evaluation but have not yet
                been evaluated. Concatenated into X upon forward call. Copied and set to
                have no gradient.
        """
        super().__init__(
            model=model,
            paths=paths,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
            user_prior=user_prior,
            user_prior_value=user_prior_value,
            decay_beta=decay_beta,
            prior_floor=prior_floor,
            custom_decay=custom_decay,
            use_resampling=use_resampling,
            resampling_fraction=resampling_fraction,
        )
        self.beta_prime = math.sqrt(2 * beta * math.pi)

        # from botorch.utils.plot_acq import plot_prior
        # plot_prior(self)

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor, use_prior: bool = True) -> Tensor:
        r"""Evaluate qUpperConfidenceBound on the candidate set `X`.

        Args:
            X: A `batch_sahpe x q x d`-dim Tensor of t-batches with `q` `d`-dim design
                points each.
se
        Returns:
            A `batch_shape'`-dim Tensor of Upper Confidence Bound values at the given
            design points `X`, where `batch_shape'` is the broadcasted batch shape of
            model and input `X`.
        """
        posterior = self.sampling_model.posterior(
            X=X, posterior_transform=self.posterior_transform
        )
        samples = self.get_posterior_samples(posterior)
        obj = self.objective(samples, X=X)

        mean = obj.mean(dim=0)
        ucb_samples = mean + self.beta_prime * 2 * (obj - mean).clamp_min(0)
        if use_prior:
            ucb_samples = ucb_samples * self.sample_probs

        return ucb_samples.max(dim=-1)[0].mean(dim=0)


class qPriorMaxValueEntropySearch(PriorMCAcquisitionFunction):

    def __init__(
        self,
        model: Model,
        sampler: PathwiseSampler,
        paths: MatheronPath,
        optimal_outputs: Optional[Tensor] = None,  # can be extracted from prior
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        user_prior: Optional[UserPrior] = None,
        user_prior_value: Optional[UserPrior] = None,
        decay_beta: float = 10.0,
        prior_floor: float = 1e-9,
        X_pending: Optional[Tensor] = None,
        exploit_fraction: float = 0.1,
        num_bins: int = 0,
        maximize: bool = True,
        num_noise_samples: int = 16,
        num_optima: float = 32,
        custom_decay: Optional[float] = None,
        use_resampling: bool = True,
        resampling_fraction: float = 0.1,
        plot: bool = False,
        decay_power: float = 2,
    ) -> None:
        r"""q-Upper Confidence Bound.
targets
        Args:
            model: A fitted model.
            beta: Controls tradeoff between mean and standard deviation in UCB.
            sampler: The sampler used to draw base samples. See `MCAcquisitionFunction`
                more details.
            objective: The MCAcquisitionObjective under which the samples are
                evaluated. Defaults to `IdentityMCObjective()`.
            posterior_transform: A PosteriorTransform (optional).
            X_pending: A `batch_shape x m x d`-dim Tensor of `m` design points that have
                points that have been submitted for function evaluation but have not yet
                been evaluated. Concatenated into X upon forward call. Copied and set to
                have no gradient.
        """
        super().__init__(
            model=model,
            paths=paths,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
            user_prior=user_prior,
            user_prior_value=user_prior_value,
            decay_beta=decay_beta,
            prior_floor=prior_floor,
            custom_decay=custom_decay,
            use_resampling=use_resampling,
            resampling_fraction=resampling_fraction,
            decay_power=decay_power,
        )
        if self.user_prior is None:
            self.optimal_gp_outputs = optimal_outputs[0:num_optima]
        else:
            self.optimal_gp_outputs = self.user_prior.optimal_outputs[0:num_optima]
        self.optimal_gp_outputs = self.optimal_gp_outputs.reshape(-1, 1, 1)
        sobol = SobolEngine(dimension=1, scramble=True)
        sobol_samples = sobol.draw(num_noise_samples)
        noise_level = model.likelihood.noise[0].sqrt().item()

        base_noise_samples = torch.distributions.Normal(
            loc=0, scale=1).icdf(sobol_samples).squeeze(-1) * noise_level
        self.base_noise_samples = base_noise_samples.reshape(-1, 1, 1, 1)
        self.min_entropy = 0.5 * (1 + math.log(0.5 * math.pi * noise_level ** 2))

        # self.optimal_gp_inputs = optimal_inputs.unsqueeze(-2)
        # self.optimal_gp_outputs = optimal_outputs.unsqueeze(-2)
        self.posterior_transform = posterior_transform

        # The optima (can be maxima, can be minima) come in as the largest
        # values if we optimize, or the smallest (likely substantially negative)
        # if we minimize. Inside the acquisition function, however, we always
        # want to consider MAX-values. As such, we need to flip them if
        # we want to minimize.
        self.condition_noiseless = True
        self.initial_gp_model = model
        self.gp_crop = num_optima
        self.optimal_cropped_outputs = self.optimal_gp_outputs[0:num_optima]
        self.exploit = (torch.rand(1) < exploit_fraction).item()

        from botorch.utils.plot_acq import plot_prior, plot_surface
        if self.initial_gp_model.train_inputs[0].shape[-1] == 2 and plot:
            plot_surface(self)
        if self.initial_gp_model.train_inputs[0].shape[-1] == 1 and plot:
            plot_prior(self)

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor, use_prior: bool = True, split: bool = False, prior_entropy: bool = True, return_sample=-1) -> Tensor:
        r"""Evaluate qPriorMES
        """
        from time import time
        prev = self.sampling_model.posterior(X).rsample()
        if self.exploit:
            prior_weighted_mean = (prev * self.sample_probs).mean(0).max(-1).values
            return prior_weighted_mean
        # The other acquisition functions need to be transposed, which is rather confusing

        # optima dim is dim # 0 for the posterior, which is kind of confusing
        # Thus, we transpose so that sample dim is always first, for both
        # THis means that the expectation in the conditional entropy estimate should average
        # dim 2 and not dim zero after we have transposed

        noisy_prev_samples = (prev + self.base_noise_samples).unsqueeze(-2)
        post = self.initial_gp_model.posterior(X, observation_noise=True)
        post_noiseless = self.initial_gp_model.posterior(X, observation_noise=False)
        prev_logprobs = post.log_prob(
            noisy_prev_samples.squeeze(-2)).unsqueeze(-1).unsqueeze(-1)

        # increase the probability of those samples that are actually possible
        # The batch output has the wrong dim
        sigma_x = post_noiseless.variance.transpose(-1, -2)
        sigma_plus = post.variance.transpose(-1, -2)
        sigma_n = sigma_plus - sigma_x
        post_mean = post.mean.transpose(-1, -2)
        # some slightly more random quantity

        g = (self.optimal_cropped_outputs.transpose(0, 1) * sigma_plus - sigma_n
             * post_mean - sigma_x * noisy_prev_samples) / (sigma_x * sigma_plus * sigma_n).sqrt()
        # H - the probability of the optimal value at any point in the search space
        # If this is zero, that means that the optimum is just impossible
        # So clamping it to say that it's not impossible should be risk-free
        h = ((self.optimal_cropped_outputs.transpose(0, 1) - post_mean) / sigma_x.sqrt())
        update_ratio = Normal(0, 1).cdf(g).clamp_min(
            1e-8) / Normal(0, 1).cdf(h).clamp_min(1e-8)
        post_probs = (torch.exp(prev_logprobs) * update_ratio)

        # zero after logging and thus do not contribute to the expactation
        # The update ratio can be zero, which becomes an issue here
        post_logprobs = update_ratio * torch.log(post_probs)
        if use_prior:
            prev_logprobs = prev_logprobs * self.sample_probs.unsqueeze(-1)
            post_logprobs = post_logprobs * self.sample_probs.unsqueeze(-1)

        prev_entropy = torch.clamp(-prev_logprobs.mean([0, 1]), self.min_entropy)
        post_entropies = torch.clamp(-post_logprobs.mean([0, 1]), self.min_entropy)
        if use_prior:
            # average over both the number of optima and max over the q-batch
            info_gain = ((prev_entropy - post_entropies).clamp_min(0)
                         ).mean([-2]).max(dim=-1).values
        else:
            # average over both the number of optima and max over the q-batch
            info_gain = ((prev_entropy - post_entropies).clamp_min(0)
                         ).mean([-2]).max(dim=-1).values

        if split:
            if prior_entropy:
                return prev_entropy.squeeze(-1)
            else:
                return post_entropies.clamp_max(prev_entropy).mean(-1)

        if return_sample >= 0:

            return (prev_entropy - post_entropies)[:, return_sample]
        return info_gain


def _update_samples(model, paths, cond_inputs, cond_targets):
    from torch.nn import ModuleList
    from copy import deepcopy
    prior_paths = paths.paths.prior_paths
    path_dtype = prior_paths.weight.dtype
    temp_model = deepcopy(model).to(path_dtype)

    num_data = len(cond_targets)
    cond_inputs, cond_targets = cond_inputs.to(path_dtype), cond_targets.to(path_dtype)
    # cond_inputs = cond_inputs.transpose(0, 1)

    # There is no batch element involved in the cond targets
    # cond_targets = cond_targets.transpose(0, 1)
    # `train_inputs`` and `train_targets` are assumed pre-transformed

    # vad ska shape av prior latents vara? Vill vi inte ha 100x100?
    noise = temp_model.likelihood.noise[0]
    noise = torch.ones(cond_targets.shape[-1]) * noise
    noise[-1] = 1e-10
    noise = torch.diag(noise)

    # The unsqueeze makes us get num_samples x num_optima (or vice versa) in batch size
    prior_latents = prior_paths(cond_inputs.unsqueeze(1))

    cond_inputs = cond_inputs.unsqueeze(1)  # .transpose(0, 2)
    cond_targets = cond_targets.unsqueeze(1)  # .transpose(0, 2)
    # Unsqueeze this one too is probably smart
    prior_latents = prior_latents  # .transpose(0, 2)
    noise = noise.unsqueeze(0).unsqueeze(
        0) * torch.ones_like(prior_latents.unsqueeze(-1))
    update_paths = gaussian_update(
        model=temp_model,
        points=cond_inputs,
        target_values=cond_targets,
        sample_values=prior_latents,
        # noise_covariance=noise
    )
    # update_paths.weight = update_paths.weight.unsqueeze(0)
    return MatheronPath(prior_paths=prior_paths, update_paths=update_paths)


def resample(paths: MatheronPath, sample_probs: Tensor, resampling_fraction: float, only_once: bool = True) -> tuple(MatheronPath, Tensor):
    num_resamples = math.ceil(resampling_fraction * len(sample_probs))
    probs = (sample_probs / sample_probs.sum()).flatten().detach().numpy()
    #plot_samples(paths, probs)

    chosen_paths = np.random.choice(
        len(sample_probs), size=num_resamples, p=probs, replace=True)
    #plot_samples(paths, np.ones_like(probs))
    indices, counts = np.unique(chosen_paths, return_counts=True)

    if only_once:
        paths = filter_paths(paths, indices)
        return paths, Tensor(counts.reshape(len(counts), 1, 1)), indices

    else:
        p_paths_weight = paths.paths.prior_paths.weight[chosen_paths]
        u_paths_weight = paths.paths.update_paths.weight[chosen_paths]
        paths.paths.prior_paths.weight = p_paths_weight
        paths.paths.update_paths.weight = u_paths_weight
        return paths, torch.ones(torch.Size([num_resamples, 1, 1]))


def plot_samples(paths, probs):
    X = torch.linspace(0, 1, 201).unsqueeze(-1)
    import matplotlib.pyplot as plt
    path_vals = paths(X)
    plt.plot(X, path_vals.detach().numpy().T, c='blue', alpha=0.2)
    plt.show()


def filter_paths(paths: MatheronPath, indices: np.array):
    # paths.paths.prior_paths.weight = paths.paths.prior_paths.weight[indices]
    # paths.paths.update_paths.weight = paths.paths.update_paths.weight[indices]
    return paths


def check_tau(tau: FloatOrTensor, name: str) -> FloatOrTensor:
    """Checks the validity of the tau arguments of the functions below, and returns
    `tau` if it is valid."""
    if isinstance(tau, Tensor) and tau.numel() != 1:
        raise ValueError(name + f" is not a scalar: {tau.numel() = }.")
    if not (tau > 0):
        raise ValueError(name + f" is non-positive: {tau = }.")
    return tau

    

class qPriorKnowledgeGradient:
    pass


class qPriorJointEntropySearch:
    pass
